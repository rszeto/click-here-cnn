import numpy as np
import os
import sys
import scipy
# import skimage
import argparse
import lmdb
import cPickle as pickle
import matplotlib.pyplot as plt
import pdb
import glob

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

# MAX_NUM_EXAMPLES = 10
MAX_NUM_EXAMPLES = 1e6

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
    imagenet_mean = get_image_mean(mean_file, model_deploy_file, resize_dim)
    # INIT NETWORK - NEW CAFFE VERSION
    net = caffe.Net(model_deploy_file, model_params_file, caffe.TEST)

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
        batch_data = prepare_batch_data(batch_size, end_idx, imagenet_mean, input_data, lmdb_keys, start_idx)# forward pass
        for data_layer_name, data in batch_data.iteritems():
            net.blobs[data_layer_name].data[...] = data
        out = net.forward()

        # extract activations
        for i, key in enumerate(output_keys):
            batch_outputs = out[key]
            for j in range(end_idx - start_idx):
                outputs[i].append(np.array(np.squeeze(batch_outputs[j, :])))
    return outputs


def get_image_mean(mean_file, model_deploy_file, resize_dim):
    if mean_file is None:
        imagenet_mean = np.array([104, 117, 123])
    else:
        imagenet_mean = np.load(mean_file)
        net_parameter = caffe_pb2.NetParameter()
        text_format.Merge(open(model_deploy_file, 'r').read(), net_parameter)
        ratio = resize_dim * 1.0 / imagenet_mean.shape[1]
        imagenet_mean = scipy.ndimage.zoom(imagenet_mean, (1, ratio, ratio))

    return imagenet_mean


def prepare_batch_data(batch_size, end_idx, imagenet_mean, input_data, lmdb_keys, start_idx):
    batch_data = {}
    for data_layer_name in input_data.keys():
        batch_data[data_layer_name] = []
    # iterate through instances
    for key in lmdb_keys[start_idx:end_idx]:
        # iterate through input layers
        for data_layer_name in input_data.keys():
            data = input_data[data_layer_name][key]
            # If data is a real image, subtract ImageNet mean
            if data_layer_name == 'data':
                # Cast to float in order to allow negative values
                data = data.astype(np.float32) - imagenet_mean
            batch_data[data_layer_name].append(data)

    # If the batch size doesn't divide the data nicely, this is needed to fill up the last batch
    for j in range(batch_size - (end_idx - start_idx)):
        for data_layer_name in input_data.keys():
            batch_data[data_layer_name].append(batch_data[data_layer_name][-1])

    return batch_data


'''
Generate the prediction accuracy of the given model.

Arguments:
- model_proto: The path to the deployment model prototxt file
- model_weights: The path to the model weights
- test_root: The root directory of the LMDBs for the test data. There should be LMDBs in folders called 'image_lmdb',
    'keypoint_class_lmdb', 'keypoint_loc_lmdb', and 'viewpoint_label_lmdb' under this directory
- imagenet_mean_file: The path to the ImageNet mean .npy file
- output_file: Where to store the accuracy results. By default, it will not save
'''
def get_model_acc(lmdb_tuples, model_proto, model_weights, test_root, output_keys, imagenet_mean_file=gv.g_image_mean_file, activation_cache_file=None, eval_from_cache=False):
    '''
    lmdb_tuples (arr): Array where each entry is a triple (input_name, lmdb_name, is_image_data), where
        - input_name is the input blob name
        - lmdb_name is the name of the LMDB storing the data
        - is_image_data is a boolean indicating whether the data is images, in which case image transformations are needed
    Example format: [
        ("data", "image_lmdb", True),
        ("pool5_weight_map", "pool5_weight_maps_lmdb", False),
        ("keypoint_class", "keypoint_class_lmdb", False)
    ]
    '''
    input_data, label_data, lmdb_keys = prepare_input_data(lmdb_tuples, test_root)

    angle_names = ['azimuth', 'elevation', 'tilt']
    # output_keys = ['pred_azimuth', 'pred_elevation', 'pred_tilt']
    if not eval_from_cache:
        # Do forward pass to extract predictions for all classes
        full_activations = batch_predict(model_proto, model_weights, min(MAX_NUM_EXAMPLES, gv.g_test_batch_size), input_data, output_keys, imagenet_mean_file, resize_dim=227)
        # Cache predictions to file
        if activation_cache_file:
            print('Caching activations')
            activation_dict = {}
            for i, angle_name in enumerate(angle_names):
                activation_dict[angle_name] = {}
                for j, key in enumerate(lmdb_keys):
                    activation_dict[angle_name][key] = full_activations[i][j]
            pickle.dump(activation_dict, open(activation_cache_file, 'wb'))
    else:
        # Get activations from cache file
        print('Importing activations from cache file')
        activation_dict = pickle.load(open(activation_cache_file, 'rb'))
        full_activations = []
        for angle_name in angle_names:
            arr = []
            for key in lmdb_keys:
                arr.append(activation_dict[angle_name][key])
            full_activations.append(arr)

    # Convert labels to numpy array for comparing against activations (which are returned as a matrix)
    viewpoint_labels_as_mat = np.zeros((len(lmdb_keys), 4))
    for i, key in enumerate(lmdb_keys):
        viewpoint_labels_as_mat[i, :] = label_data[key]
    # Convert keypoint class vectors to indexes for stratified evaluation
    keypoint_classes = extract_keypoint_classes(test_root, MAX_NUM_EXAMPLES)

    # Extract the angle activations for the correct object class
    obj_classes = viewpoint_labels_as_mat[:, 0]
    preds = activations_to_preds(full_activations, obj_classes)
    # Compare predictions to ground truth labels
    angle_dists = compute_angle_dists(preds, viewpoint_labels_as_mat)
    # Compute accuracy and median error per object class
    class_accs, class_med_errs = compute_metrics_by_obj_class(angle_dists, obj_classes)
    # Compute accuracy and median error per keypoint class
    keypoint_class_accs, keypoint_class_mederrs = compute_metrics_by_keypt_class(angle_dists, keypoint_classes)

    # return class_accs, class_med_errs, keypoint_class_accs, keypoint_class_mederrs
    return class_accs, class_med_errs, keypoint_class_accs, keypoint_class_mederrs


def prepare_input_data(lmdb_tuples, test_root, max_num_examples=MAX_NUM_EXAMPLES):
    input_data = {}
    label_data = None
    lmdb_keys = None
    for lmdb_tuple in lmdb_tuples:
        input_name, lmdb_name, is_image_data = lmdb_tuple
        cur_lmdb = lmdb.open(os.path.join(test_root, lmdb_name), readonly=True, lock=False)
        if input_name == 'label':
            label_data = utils.getFirstNLmdbVecs(cur_lmdb, max_num_examples)
        else:
            if is_image_data:
                input_data[input_name] = utils.getFirstNLmdbCaffeImgs(cur_lmdb, max_num_examples)
            else:
                input_data[input_name] = utils.getFirstNLmdbVecs(cur_lmdb, max_num_examples)
                # Hack
                if input_name == 'keypoint_map':
                    for key, value in input_data[input_name].iteritems():
                        input_data[input_name][key] = value[np.newaxis, :, :]
                elif input_name == 'keypoint_class':
                    for key, value in input_data[input_name].iteritems():
                        input_data[input_name][key] = value[:, np.newaxis, np.newaxis]
                elif input_name == 'attn':
                    for key, value in input_data[input_name].iteritems():
                        input_data[input_name][key] = value[np.newaxis, :, :]

            # Compare keys to make sure they're consistent across LMDBs
            if lmdb_keys is None:
                lmdb_keys = input_data[input_name].keys()
            else:
                assert (lmdb_keys == input_data[input_name].keys())

    # Check label data was found and its keys match the input data
    assert (label_data)
    assert (lmdb_keys == label_data.keys())
    return input_data, label_data, lmdb_keys


def extract_keypoint_classes(test_root, max_num_examples):
    keypoint_class_lmdb = lmdb.open(os.path.join(test_root, 'keypoint_class_lmdb'), readonly=True, lock=False)
    keypoint_class_data = utils.getFirstNLmdbVecs(keypoint_class_lmdb, max_num_examples)
    ret = []
    for lmdb_key, keypoint_class_vec in keypoint_class_data.iteritems():
        ret.append(np.where(keypoint_class_vec)[0][0])
    return ret

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
        class_id = utils.SYNSET_CLASSIDX_MAP[synset]
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
    parser.add_argument('exp_num', type=int, help='The experiment number')
    parser.add_argument('iter_num', type=int, help='The iteration number of the snapshot')
    parser.add_argument('--cache_preds', action='store_true', help='Whether to save the predictions to disk if they don\'t exist, or use the existing cache if they do')
    args = parser.parse_args()

    print('Evaluating weights for experiment %d, iteration %d' % (args.exp_num, args.iter_num))

    # First, locate the experiment
    exp_dir_list = glob.glob(os.path.join(gv.g_experiments_root_folder, '%06d_*' % args.exp_num))
    if len(exp_dir_list) == 0:
        print('Failed to find experiment %d' % args.exp_num)
        exit()
    exp_dir = exp_dir_list[0]

    # Then, read the evaluation configuration
    eval_config_lines = [line.strip() for line in open(os.path.join(exp_dir, 'evaluation', 'evalAcc_args.txt'), 'r').readlines()]
    model_proto_path = eval_config_lines[0]
    model_weights_path = eval_config_lines[1].replace('###', str(args.iter_num))
    test_root = eval_config_lines[2]
    output_keys = eval_config_lines[3:6]
    lmdb_info = eval_config_lines[6:]
    assert(len(lmdb_info) % 3 == 0)
    lmdb_tuples = zip(lmdb_info[0::3], lmdb_info[1::3], [s == 'True' for s in lmdb_info[2::3]])

    # Generate cache path if needed
    cache_path = os.path.join(exp_dir, 'evaluation', 'cache_%d.pkl' % args.iter_num) if args.cache_preds else None
    class_accs, class_med_errs, keypoint_class_accs, keypoint_class_mederrs = get_model_acc(lmdb_tuples, model_proto_path, model_weights_path, test_root, output_keys, activation_cache_file=cache_path)

    # Write accuracy and median error results
    output_file = os.path.join(exp_dir, 'evaluation', 'acc_mederr_%d.txt' % args.iter_num)
    f = open(output_file, 'wb')
    f.write('Results for test set at %s\n' % test_root)
    f.write('Model prototxt file: %s\n' % model_proto_path)
    f.write('Model weights: %s\n' % model_weights_path)
    f.write('\n')
    for class_name in sorted(class_accs.keys()):
        f.write('%s:\n' % class_name)
        f.write('\tAccuracy: %0.4f\n' % class_accs[class_name])
        f.write('\tMedErr: %0.4f\n' % class_med_errs[class_name])
    f.write('Mean accuracy: %0.4f\n' % np.mean(class_accs.values()))
    f.write('Mean medErr: %0.4f\n' % np.mean(class_med_errs.values()))
    f.write('\n')
    if keypoint_class_accs and keypoint_class_mederrs:
        for keypoint_class_name in sorted(keypoint_class_accs.keys()):
            f.write('%s:\n' % keypoint_class_name)
            f.write('\tAccuracy: %0.4f\n' % keypoint_class_accs[keypoint_class_name])
            f.write('\tMedErr: %0.4f\n' % keypoint_class_mederrs[keypoint_class_name])
    f.close()
    # Expand read permissions
    os.chmod(output_file, 0766)
