import numpy as np
import os
import sys
import scipy
import skimage
import argparse
import lmdb
import pickle
import matplotlib.pyplot as plt
import time

# Import custom LMDB utilities
eval_scripts_path = os.path.dirname(os.path.abspath(__file__))
view_est_corresp_path = os.path.dirname(eval_scripts_path)
sys.path.append(view_est_corresp_path)
import gen_lmdb_utils as utils
# Import global variables
render4cnn_path = os.path.dirname(view_est_corresp_path)
sys.path.append(render4cnn_path)
import global_variables as gv

# Import Caffe
sys.path.append(gv.g_pycaffe_path)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

MAX_NUM_EXAMPLES = 10
# MAX_NUM_EXAMPLES = 1e6
BATCH_SIZE = 192

### END IMPORTS ###

# Angle conversions
def deg2rad(deg_angle):
    return deg_angle * np.pi / 180.0

def rad2deg(rad_angle):
    return rad_angle * 180.0 / np.pi

# Produce the rotation matrix from the axis rotations
def angle2dcm(xRot, yRot, zRot, deg_type='deg'):
    if deg_type == 'deg':
        xRot = deg2rad(xRot)
        yRot = deg2rad(yRot)
        zRot = deg2rad(zRot)

    xMat = np.array([
        [np.cos(xRot), np.sin(xRot), 0],
        [-np.sin(xRot), np.cos(xRot), 0],
        [0, 0, 1]
    ])

    yMat = np.array([
        [np.cos(yRot), 0, -np.sin(yRot)],
        [0, 1, 0],
        [np.sin(yRot), 0, np.cos(yRot)]
    ])

    zMat = np.array([
        [1, 0, 0],
        [0, np.cos(zRot), np.sin(zRot)],
        [0, -np.sin(zRot), np.cos(zRot)]
    ])

    return np.dot(zMat, np.dot(yMat, xMat))

'''
Forward propagate a set of images and obtain viewpoint predictions. Note that this returns predictions across all
object classes and angles, so the result needs to be further processed.

Arguments:
- model_deploy_file (str): The path to the deployment model prototxt file
- model_params_file (str): The path to the model weights
- batch_size (int): How many instances to forward prop at once
- input_data (dict): The data to forward. The keys are the data layer names, and the values are dictionaries mapping from LMDB key to numpy arrays.
- output_keys (list of str): The names of the output layers whose values should be returned
- mean_file (str): The path to the ImageNet mean .npy file
- resize_dim (int): How large the images should be, aka the size of the image input to the network. 0 means no resizing.
'''
def batch_predict(model_deploy_file, model_params_file, batch_size, input_data, output_keys, mean_file=None, resize_dim=0):
    # Get LMDB keys from the first input type
    first_data = input_data[input_data.keys()[0]]
    lmdb_keys = first_data.keys()

    # set imagenet_mean
    if mean_file is None:
        imagenet_mean = np.array([104, 117, 123])
    else:
        imagenet_mean = np.load(mean_file)
        net_parameter = caffe_pb2.NetParameter()
        text_format.Merge(open(model_deploy_file, 'r').read(), net_parameter)
        print net_parameter
        print net_parameter.input_dim, imagenet_mean.shape
        ratio = resize_dim * 1.0 / imagenet_mean.shape[1]
        imagenet_mean = scipy.ndimage.zoom(imagenet_mean, (1, ratio, ratio))

    # INIT NETWORK - NEW CAFFE VERSION
    net = caffe.Net(model_deploy_file, model_params_file, caffe.TEST)
    # Initialize transformer for the image data
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # height*width*channel -> channel*height*width
    transformer.set_mean('data', imagenet_mean)  #### subtract mean ####
    transformer.set_raw_scale('data', 255)  # pixel value range
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB -> BGR

    # set test batch size
    for data_layer_name in input_data.keys():
        data_blob_shape = list(net.blobs[data_layer_name].data.shape)
        net.blobs[data_layer_name].reshape(batch_size, *data_blob_shape[1:])

    ## BATCH PREDICTS
    batch_num = int(np.ceil(len(lmdb_keys) / float(batch_size)))
    outputs = [[] for _ in range(len(output_keys))]
    for k in range(batch_num):
        start_idx = batch_size * k
        end_idx = min(batch_size * (k + 1), len(lmdb_keys))
        print 'batch: %d/%d, idx: %d to %d' % (k+1, batch_num, start_idx, end_idx)

        # prepare batch input data
        batch_data = {}
        for data_layer_name in input_data.keys():
            batch_data[data_layer_name] = []
        # iterate through instances
        for key in lmdb_keys[start_idx:end_idx]:
            # iterate through input layers
            for data_layer_name in input_data.keys():
                data = input_data[data_layer_name][key]
                # Transform data if needed
                if data_layer_name in transformer.inputs.keys():
                    data = transformer.preprocess(data_layer_name, data)
                batch_data[data_layer_name].append(data)

        # If the batch size doesn't divide the data nicely, this is needed to fill up the last batch
        for j in range(batch_size - (end_idx - start_idx)):
            for data_layer_name in input_data.keys():
                batch_data[data_layer_name].append(batch_data[data_layer_name][-1])

        # forward pass
        for data_layer_name, data in batch_data.iteritems():
            net.blobs[data_layer_name].data[...] = data
        out = net.forward()

        # extract activations
        for i, key in enumerate(output_keys):
            batch_outputs = out[output_keys[i]]
            for j in range(end_idx - start_idx):
                outputs[i].append(np.array(np.squeeze(batch_outputs[j, :])))
    return outputs

'''
Generate the prediction accuracy of the given model.

Arguments:
- model_proto: The path to the deployment model prototxt file
- model_weights: The path to the model weights
- test_root: The root directory of the LMDBs for the test data. There should be LMDBs in folders called 'image_lmdb',
    'keypoint_class_lmdb', 'keypoint_loc_lmdb', and 'viewpoint_label_lmdb' under this directory
- imagenet_mean_file: The path to the ImageNet mean .npy file
- outputFile: Where to store the accuracy results. By default, it will not save
'''
def get_model_acc(model_proto, model_weights, test_root, imagenet_mean_file=None, prediction_cache_file=None, eval_from_cache=False, blank_keypoint_img=False, blank_keypoint_class=False, use_keypoint_data=True, use_sparse_keypoint_map=True):
    ## Get data from test LMDBs
    # Images
    image_lmdb = lmdb.open(os.path.join(test_root, 'image_lmdb'), readonly=True)
    data_image = utils.getFirstNLmdbImgs(image_lmdb, MAX_NUM_EXAMPLES)
    # Viewpoint labels
    viewpoint_label_lmdb = lmdb.open(os.path.join(test_root, 'viewpoint_label_lmdb'))
    viewpoint_labels = utils.getFirstNLmdbVecs(viewpoint_label_lmdb, MAX_NUM_EXAMPLES)
    # Keypoint image
    keypoint_loc_lmdb_name = 'keypoint_loc_lmdb' if use_sparse_keypoint_map else 'gaussian_keypoint_map_lmdb_23'
    keypoint_loc_lmdb = lmdb.open(os.path.join(test_root, keypoint_loc_lmdb_name), readonly=True)
    data_keypoint_image = utils.getFirstNLmdbImgs(keypoint_loc_lmdb, MAX_NUM_EXAMPLES)
    # Keypoint class
    keypoint_class_lmdb = lmdb.open(os.path.join(test_root, 'keypoint_class_lmdb'))
    data_keypoint_class = utils.getFirstNLmdbVecs(keypoint_class_lmdb, MAX_NUM_EXAMPLES)

    ## Get keys and make sure they match across all LMDBs
    lmdb_keys = data_image.keys()
    assert(lmdb_keys == viewpoint_labels.keys())
    assert(lmdb_keys == data_keypoint_image.keys())
    assert(lmdb_keys == data_keypoint_class.keys())

    ## Process data for batch prediction
    # Convert heatmaps from 227x227 to 1x227x227
    for key in data_keypoint_image.keys():
        data_keypoint_image[key] = data_keypoint_image[key].reshape((1, 227, 227))
    # Blank out keypoint image if needed
    if blank_keypoint_img:
        for key in data_keypoint_image.keys():
            data_keypoint_image[key][...] = 0
    # Blank out keypoint class if needed
    if blank_keypoint_class:
        for key in data_keypoint_class.keys():
            data_keypoint_class[key][...] = 0

    angle_names = ['azimuth', 'elevation', 'tilt']
    output_keys = ['pred_azimuth', 'pred_elevation', 'pred_tilt']
    if not eval_from_cache:
        ## Configure data for batch_predict
        input_data = {'data': data_image}
        if use_keypoint_data:
            input_data['data_keypoint_image'] = data_keypoint_image
            input_data['data_keypoint_class'] = data_keypoint_class

        # Do forward pass to extract predictions for all classes
        full_predictions = batch_predict(model_proto, model_weights, min(MAX_NUM_EXAMPLES, BATCH_SIZE), input_data, output_keys, imagenet_mean_file, resize_dim=227)
        # Cache predictions to file
        if prediction_cache_file:
            prediction_dict = {}
            for i, angle_name in enumerate(angle_names):
                prediction_dict[angle_name] = {}
                for j, key in enumerate(lmdb_keys):
                    prediction_dict[angle_name][key] = full_predictions[i][j]
            pickle.dump(prediction_dict, open(prediction_cache_file, 'wb'))
    else:
        # Get predictions from cache file
        print('Importing predictions from cache file')
        prediction_dict = pickle.load(open(prediction_cache_file, 'rb'))
        full_predictions = []
        for angle_name in angle_names:
            arr = []
            for key in lmdb_keys:
                arr.append(prediction_dict[angle_name][key])
            full_predictions.append(arr)

    # Convert labels to numpy array for comparing against predictions (which are returned as a matrix)
    viewpoint_labels_as_mat = np.zeros((len(lmdb_keys), 4))
    for i, key in enumerate(lmdb_keys):
        viewpoint_labels_as_mat[i, :] = viewpoint_labels[key]
    # Convert keypoint class vectors to indexes for stratified evaluation
    keypoint_classes = [np.where(data_keypoint_class[lmdb_keys[k]])[0] for k in range(len(lmdb_keys))]

    # Extract the angle predictions for the correct object class
    obj_classes = viewpoint_labels_as_mat[:, 0]
    preds = activations_to_preds(full_predictions, obj_classes)
    # Compare predictions to ground truth labels
    angle_dists = compute_angle_dists(preds, viewpoint_labels_as_mat)
    # Compute accuracy and median error per object class
    class_accs, class_med_errs = compute_metrics_by_obj_class(angle_dists, obj_classes)
    # Compute accuracy and median error per keypoint class
    keypoint_class_accs, keypoint_class_mederrs = compute_metrics_by_keypt_class(angle_dists, keypoint_classes)

    return class_accs, class_med_errs, keypoint_class_accs, keypoint_class_mederrs


def activations_to_preds(full_predictions, obj_classes):
    num_angles = len(full_predictions)
    preds = np.zeros((obj_classes.shape[0], num_angles))
    for i in range(obj_classes.shape[0]):
        class_idx = int(obj_classes[i])
        # Go through each angle type
        for k in range(num_angles):
            all_class_probs = full_predictions[k][i]
            # Get predictions for given class
            gt_class_probs = all_class_probs[class_idx * 360:(class_idx + 1) * 360]
            pred = gt_class_probs.argmax() + class_idx * 360
            preds[i, k] = pred

    return preds


def compute_metrics_by_keypt_class(angle_dists, keypoint_classes):
    keypoint_class_accs = {}
    keypoint_class_mederrs = {}
    for i, keypoint_class_name in enumerate(utils.KEYPOINT_CLASSES):
        keypoint_class_indexes = [k for k in range(len(keypoint_classes)) if keypoint_classes[k] == i]
        if len(keypoint_class_indexes) == 0:
            print('Found no examples with keypoint %s, skipping' % keypoint_class_name)
            continue
        angle_dists_cur_keypoint_class = angle_dists[keypoint_class_indexes]
        keypoint_class_acc = np.sum(angle_dists_cur_keypoint_class < np.pi / 6) / float(len(keypoint_class_indexes))
        keypoint_class_mederr = rad2deg(np.median(angle_dists_cur_keypoint_class))
        keypoint_class_accs[keypoint_class_name] = keypoint_class_acc
        keypoint_class_mederrs[keypoint_class_name] = keypoint_class_mederr

    return keypoint_class_accs, keypoint_class_mederrs


def compute_metrics_by_obj_class(angle_dists, obj_classes):
    class_accs = {}
    class_med_errs = {}
    for synset, class_name in utils.synset_name_pairs:
        class_id = utils.SYNSET_OLDCLASSIDX_MAP[synset]
        obj_class_indexes = np.where(obj_classes == class_id)[0]
        if len(obj_class_indexes) == 0:
            continue
        angle_distsCurClass = angle_dists[obj_class_indexes]
        class_acc = np.sum(angle_distsCurClass < np.pi / 6) / float(len(obj_class_indexes))
        class_med_err = rad2deg(np.median(angle_distsCurClass))
        class_accs[class_name] = class_acc
        class_med_errs[class_name] = class_med_err

    return class_accs, class_med_errs

def compute_angle_dists(preds, viewpoint_labels_as_mat):
    angle_dists = np.zeros(viewpoint_labels_as_mat.shape[0])
    for i in range(viewpoint_labels_as_mat.shape[0]):
        # Get rotation matrices from prediction and ground truth angles
        predR = angle2dcm(preds[i, 0], preds[i, 1], preds[i, 2])
        gtR = angle2dcm(viewpoint_labels_as_mat[i, 1], viewpoint_labels_as_mat[i, 2], viewpoint_labels_as_mat[i, 3])
        # Get geodesic distance
        angleDist = scipy.linalg.norm(scipy.linalg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)
        angle_dists[i] = angleDist

    return angle_dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_proto', type=str, help='The path to the deployment model prototxt file')
    parser.add_argument('model_weights', type=str, help='The path to the model weights')
    parser.add_argument('test_root', type=str, help='The root directory of the LMDBs for the test data. There should be LMDBs in folders called \'image_lmdb\', \'keypoint_class_lmdb\', \'keypoint_loc_lmdb\', and \'viewpoint_label_lmdb\' under this directory')
    parser.add_argument('mode', type=str, help='The type of evaluation to be done. Possibilities are "correspondences", "r4cnn"')
    parser.add_argument('--imagenet_mean_file', type=str, default=gv.g_image_mean_file, help='The path to the ImageNet mean .npy file')
    parser.add_argument('--outputFile', type=str, default=None, help='Where to store the accuracy results. By default, it will not save')
    parser.add_argument('--predictionCacheFile', type=str, default=None, help='Where to cache the (verbose) angle predictions. By default, it will not save')
    parser.add_argument('--eval_from_cache', dest='eval_from_cache', action='store_true', help='True if evaluations should be made from the cache file')
    parser.set_defaults(eval_from_cache=False)

    args = parser.parse_args()
    class_accs, class_med_errs, keypoint_class_accs, keypoint_class_mederrs = get_model_acc(args.model_proto, args.model_weights, args.test_root, prediction_cache_file=args.predictionCacheFile, eval_from_cache=args.eval_from_cache, use_keypoint_data=(args.mode == 'correspondences'))

    # Write accuracy and median error results
    if args.outputFile:
        f = open(args.outputFile, 'wb')
    else:
        f = sys.stdout
    f.write('Results for test set at %s\n' % args.test_root)
    f.write('Model prototxt file: %s\n' % args.model_proto)
    f.write('Model weights: %s\n' % args.model_weights)
    f.write('\n')
    for class_name in sorted(class_accs.keys()):
        f.write('%s:\n' % class_name)
        f.write('\tAccuracy: %0.4f\n' % class_accs[class_name])
        f.write('\tMedErr: %0.4f\n' % class_med_errs[class_name])
    f.write('Mean accuracy: %0.4f\n' % np.mean(class_accs.values()))
    f.write('Mean medErr: %0.4f\n' % np.mean(class_med_errs.values()))
    f.write('\n')
    for keypoint_class_name in sorted(keypoint_class_accs.keys()):
        f.write('%s:\n' % keypoint_class_name)
        f.write('\tAccuracy: %0.4f\n' % keypoint_class_accs[keypoint_class_name])
        f.write('\tMedErr: %0.4f\n' % keypoint_class_mederrs[keypoint_class_name])
    if args.outputFile:
        f.close()