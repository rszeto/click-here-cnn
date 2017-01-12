import numpy as np
import scipy.io as spio
import os
import sys
import re
from glob import glob
from scipy.ndimage import imread
import lmdb
import random
from scipy.misc import imresize
import matplotlib.pyplot as plt
from warnings import warn
import skimage
import time
import multiprocessing
from pprint import pprint
import h5py
import scipy.ndimage
import itertools
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imsave

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import global_variables as gv

sys.path.append(gv.g_pycaffe_path)
import caffe

RENDER4CNN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PASCAL3D_ROOT = os.path.join(RENDER4CNN_ROOT, 'datasets', 'pascal3d')
ANNOTATIONS_ROOT = os.path.join(PASCAL3D_ROOT, 'Annotations')
IMAGES_ROOT = os.path.join(PASCAL3D_ROOT, 'Images')

# synset_name_pairs = gv.g_shape_synset_name_pairs
synset_name_pairs = [('02924116', 'bus'), ('02958343', 'car'), ('03790512', 'motorbike')]

DATASET_SOURCES = ['pascal', 'imagenet']
LMDB_NAMES = ['image_lmdb', 'keypoint_class_lmdb', 'keypoint_loc_lmdb', 'viewpoint_label_lmdb']

KEYPOINT_TYPES = {
    # 'aeroplane': ['left_elevator', 'left_wing', 'noselanding', 'right_elevator', 'right_wing', 'rudder_lower', 'rudder_upper', 'tail'],
    # 'bicycle': ['head_center', 'left_back_wheel', 'left_front_wheel', 'left_handle', 'left_pedal_center', 'right_back_wheel', 'right_front_wheel', 'right_handle', 'right_pedal_center', 'seat_back', 'seat_front'],
    # 'boat': ['head', 'head_down', 'head_left', 'head_right', 'tail_left', 'tail_right', 'tail'],
    # 'bottle': ['mouth', 'body', 'body_left', 'body_right', 'bottom', 'bottom_left', 'bottom_right'],
    'bus': ['body_back_left_lower', 'body_back_left_upper', 'body_back_right_lower', 'body_back_right_upper', 'body_front_left_upper', 'body_front_right_upper', 'body_front_left_lower', 'body_front_right_lower', 'left_back_wheel', 'left_front_wheel', 'right_back_wheel', 'right_front_wheel'],
    'car': ['left_front_wheel', 'left_back_wheel', 'right_front_wheel', 'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield', 'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light', 'right_front_light', 'left_back_trunk', 'right_back_trunk'],
    # 'chair': ['back_upper_left', 'back_upper_right', 'seat_upper_left', 'seat_upper_right', 'seat_lower_left', 'seat_lower_right', 'leg_upper_left', 'leg_upper_right', 'leg_lower_left', 'leg_lower_right'],
    # 'diningtable': ['leg_upper_left', 'leg_upper_right', 'leg_lower_left', 'leg_lower_right', 'top_upper_left', 'top_upper_right', 'top_lower_left', 'top_lower_right', 'top_up', 'top_down', 'top_left', 'top_right'],
    'motorbike': ['back_seat', 'front_seat', 'head_center', 'headlight_center', 'left_back_wheel', 'left_front_wheel', 'left_handle_center', 'right_back_wheel', 'right_front_wheel', 'right_handle_center'],
    # 'sofa': ['front_bottom_left', 'front_bottom_right', 'seat_bottom_left', 'seat_bottom_right', 'left_bottom_back', 'right_bottom_back', 'top_left_corner', 'top_right_corner', 'seat_top_left', 'seat_top_right'],
    # 'train': ['head_left_bottom', 'head_left_top', 'head_right_bottom', 'head_right_top', 'head_top', 'mid1_left_bottom', 'mid1_left_top', 'mid1_right_bottom', 'mid1_right_top', 'mid2_left_bottom', 'mid2_left_top', 'mid2_right_bottom', 'mid2_right_top', 'tail_left_bottom', 'tail_left_top', 'tail_right_bottom', 'tail_right_top'],
    # 'tvmonitor': ['front_bottom_left', 'front_bottom_right', 'front_top_left', 'front_top_right', 'back_bottom_left', 'back_bottom_right', 'back_top_left', 'back_top_right']
}

CLASSNAME_SYNSET_MAP = {}
for synset, class_name in synset_name_pairs:
    CLASSNAME_SYNSET_MAP[class_name] = synset

SYNSET_OLDCLASSIDX_MAP = {}
for i in range(len(gv.g_shape_synset_name_pairs)):
    synset, _ = gv.g_shape_synset_name_pairs[i]
    SYNSET_OLDCLASSIDX_MAP[synset] = i

SYNSET_CLASSNAME_MAP = {}
for synset, class_name in synset_name_pairs:
    SYNSET_CLASSNAME_MAP[synset] = class_name

SYNSET_CLASSIDX_MAP = {}
for i in range(len(synset_name_pairs)):
    SYNSET_CLASSIDX_MAP[synset_name_pairs[i][0]] = i

KEYPOINT_CLASSES = []
for synset, class_name in synset_name_pairs:
    keypoint_names = KEYPOINT_TYPES[class_name]
    for keypoint_name in keypoint_names:
        KEYPOINT_CLASSES.append(class_name + '_' + keypoint_name)

KEYPOINTCLASS_INDEX_MAP = {}
for i in range(len(KEYPOINT_CLASSES)):
    KEYPOINTCLASS_INDEX_MAP[KEYPOINT_CLASSES[i]] = i

INFO_FILE_HEADER = 'imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n'
LINE_FORMAT = re.compile('(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*),(.*)')
DEFAULT_LMDB_SIZE = 1e13
TXN_BATCH_SIZE = 100

######### Importing .mat files ###############################################
######### Reference: http://stackoverflow.com/a/8832212 ######################

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        # Handle case where elem is an array of mat_structs
        elif isinstance(elem, np.ndarray) and len(elem) > 0 and \
                isinstance(elem[0], spio.matlab.mio5_params.mat_struct):
            dict[strg] = np.array([_todict(subelem) for subelem in elem])
        else:
            dict[strg] = elem
    return dict

######### Saving to LMDB #####################################################

def image_to_caffe(image):
    # Convert image to weird Caffe format. Fork depending on if image is color or not.
    if image.ndim == 2:
        # If grayscale, transpose to match Caffe dim ordering
        image_caffe = image.reshape((1, image.shape[0], image.shape[1]))
    else:
        # If color, switch from RGB to BGR
        image_caffe = image[:, :, ::-1]
        # Go from H*W*C to C*H*W
        image_caffe = image_caffe.transpose(2, 0, 1)
    return image_caffe

def image_to_datum(image):
    image_caffe = image_to_caffe(image)
    # Populate datum
    datum = caffe.proto.caffe_pb2.Datum()
    (datum.channels, datum.height, datum.width) = image_caffe.shape
    datum.data = image_caffe.tobytes()
    return datum

def vector_to_datum(vec):
    # Reshape vector to be in Caffe format
    vec_caffe = vec.reshape([len(vec), 1, 1])
    # Put datum into LMDB
    datum = caffe.io.array_to_datum(vec_caffe)
    return datum

def tensor_to_datum(tensor):
    if tensor.ndim == 2:
        tensor = tensor[np.newaxis, :, :]
    return caffe.io.array_to_datum(tensor)

######### Loading from LMDB (for debugging) ##################################

def caffe_to_image(image_caffe):
    # Change C*H*W to H*W*C
    image = image_caffe.transpose((1, 2, 0))
    # Squeeze extra dimension if grayscale
    image = image.squeeze()
    # Change BGR to RGB if image is in color
    if image.ndim == 3:
        image = image[:, :, ::-1]
    return image

def lmdbStrToImage(lmdbStr):
    # Get datum from LMDB
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(lmdbStr)
    # Retrieve image in weird Caffe format
    caffe_image = caffe.io.datum_to_array(datum)
    return caffe_to_image(caffe_image)

def lmdbStrToCaffeImage(lmdbStr):
    # Get datum from LMDB
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(lmdbStr)
    # Retrieve image in weird Caffe format
    caffe_image = caffe.io.datum_to_array(datum)
    return caffe_image

# Adapted from Render For CNN's caffe_utils function load_vector_from_lmdb
def lmdbStrToVec(lmdbStr):
    # Get datum from LMDB
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(lmdbStr)
    # Parse to array
    array = caffe.io.datum_to_array(datum)
    array = np.squeeze(array)
    assert (array is not None)
    return array

# Return a dict of the first N items in the given LMDB (or all if <N items exist)
# X is a function that takes an LMDB string and returns the desired item
def _getFirstNLmdbX(lmdb_obj, N, X):
    with lmdb_obj.begin() as txn:
        # Warn if N is greater than number of pairs in LMDB
        maxN = txn.stat()['entries']
        print maxN
        if N > maxN:
            warn('Only %d values in LMDB, obtaining all' % maxN)

        # Go through LMDB and populate array with images
        cursor = txn.cursor()
        ret = {}
        for i in range(min(N, maxN)):
            cursor.next()
            key = cursor.key()
            item = X(cursor.value())
            ret[key] = item
        return ret

def getFirstNLmdbImgs(lmdb_obj, N):
    return _getFirstNLmdbX(lmdb_obj, N, lmdbStrToImage)

def getFirstNLmdbCaffeImgs(lmdb_obj, N):
    return _getFirstNLmdbX(lmdb_obj, N, lmdbStrToCaffeImage)

def getFirstNLmdbVecs(lmdb_obj, N):
    return _getFirstNLmdbX(lmdb_obj, N, lmdbStrToVec)

## Other

'''
@brief:
    convert 360 view degree to view estimation label
    e.g. for bicycle with class_idx 1, label will be 360~719
'''
def view2label(degree, class_index):
  return int(degree)%360 + class_index*360

'''
@brief:
    Return true iff the point is located inside the box, inclusive.
    E.g. If point is on box corner, return true
'''

def insideBox(point, box):
    return point[0] >= box[0] and point[0] <= box[2] \
            and point[1] >= box[1] and point[1] <= box[3]

def keypointInfo2Str(fullImagePath, bbox, keyptLoc, keyptClass, viewptLabel):
    return '%s,%d,%d,%d,%d,%f,%f,%d,%d,%d,%d,%d\n' % (
        fullImagePath,
        bbox[0], bbox[1], bbox[2], bbox[3],
        keyptLoc[0], keyptLoc[1],
        keyptClass,
        viewptLabel[0], viewptLabel[1], viewptLabel[2], viewptLabel[3]
    )

def padImage(image, pad_size, pad_value):
    image_tr = image.transpose((2, 0, 1))
    padded_tr = np.pad(image_tr, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)), 'constant', constant_values=pad_value)
    return padded_tr.transpose(1, 2, 0)

def createSynKeypointCsv():
    if not os.path.exists(gv.g_corresp_folder):
        os.mkdir(gv.g_corresp_folder)
    infoFile = open(os.path.join(gv.g_corresp_folder, 'syn_corresp_lmdb_info.csv'), 'w')
    infoFile.write(INFO_FILE_HEADER)
    imgNamePattern = re.compile('(.*)_(.*)_a(.*)_e(.*)_t(.*)_d(.*).jpg')
    for synset in os.listdir(gv.g_syn_images_bkg_overlaid_folder):
        for md5 in os.listdir(os.path.join(gv.g_syn_images_bkg_overlaid_folder, synset)):
            for image_path in glob(os.path.join(gv.g_syn_images_bkg_overlaid_folder, synset, md5, '*.jpg')):
                # Get bounding box for object (in this case, whole image)
                image = imread(image_path)
                bbox = np.array([0, 0, image.shape[1]-1, image.shape[0]-1])

                # Get class index from synset
                # object_class = SYNSET_CLASSIDX_MAP[synset]
                object_class = SYNSET_OLDCLASSIDX_MAP[synset]
                class_name = SYNSET_CLASSNAME_MAP[synset]

                # Extract azimuth, elevation, rotation from file name
                image_name = os.path.basename(image_path)
                m = re.match(imgNamePattern, image_name)
                azimuth, elevation, tilt = m.group(3, 4, 5)

                # Get CSV that lists keypoint locations in the image
                csvPath = image_path.replace('.jpg', '_keypoint2d.csv')
                with open(csvPath, 'r') as f:
                    for line in f.readlines():
                        m = re.match('(.*),(.*),(.*)', line)
                        keypoint_name, x, y = m.group(1, 2, 3)

                        keypoint_loc = (float(x), float(y))
                        keypoint_class = KEYPOINTCLASS_INDEX_MAP[class_name + '_' + keypoint_name]
                        finalLabel = (
                            object_class,
                            view2label(azimuth, object_class),
                            view2label(elevation, object_class),
                            view2label(tilt, object_class)
                        )
                        keyptStr = keypointInfo2Str(image_path, bbox, keypoint_loc, keypoint_class, finalLabel)
                        infoFile.write(keyptStr)

    infoFile.close()

def createPascalKeypointCsv():
    # Generate train and test lists and store in file
    matlab_cmd = 'addpath(\'%s\'); getPascalTrainValImgs' % BASE_DIR
    os.system('matlab -nodisplay -r "try %s; catch; end; quit;"' % matlab_cmd)
    # Get training and test image IDs
    with open('trainImgIds.txt', 'rb') as trainIdsFile:
        trainIds = np.loadtxt(trainIdsFile, dtype='string')
    with open('valImgIds.txt', 'rb') as testIdsFile:
        testIds = np.loadtxt(testIdsFile, dtype='string')
    # Delete the ID files
    os.remove('trainImgIds.txt')
    os.remove('valImgIds.txt')

    if not os.path.exists(gv.g_corresp_folder):
        os.mkdir(gv.g_corresp_folder)
    info_file_train = open(os.path.join(gv.g_corresp_folder, 'pascal_corresp_lmdb_info_train.csv'), 'w')
    info_file_train.write(INFO_FILE_HEADER)
    info_file_test = open(os.path.join(gv.g_corresp_folder, 'pascal_corresp_lmdb_info_test.csv'), 'w')
    info_file_test.write(INFO_FILE_HEADER)

    for synset, class_name in synset_name_pairs:
        # object_class = SYNSET_CLASSIDX_MAP[synset]
        object_class = SYNSET_OLDCLASSIDX_MAP[synset]
        for dataset_source in DATASET_SOURCES:
            class_source_id = '%s_%s' % (class_name, dataset_source)
            for anno_file in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, class_source_id))):
                anno_file_id = os.path.splitext(os.path.basename(anno_file))[0]
                if anno_file_id in trainIds:
                    anno_file_set = 'train'
                elif anno_file_id in testIds:
                    anno_file_set = 'test'
                else:
                    continue

                anno = loadmat(os.path.join(ANNOTATIONS_ROOT, class_source_id, anno_file))['record']
                full_image_path = os.path.join(IMAGES_ROOT, class_source_id, anno['filename'])
                full_image = imread(full_image_path)
                # Convert grayscale images to "color"
                if full_image.ndim == 2:
                    full_image = np.dstack((full_image, full_image, full_image))

                # Make objs an array regardless of how many objects there are
                objs = np.array([anno['objects']]) if isinstance(anno['objects'], dict) else anno['objects']
                for obj_i, obj in enumerate(objs):
                    # Only deal with objects in current class
                    if obj['class'] == class_name:
                        # Get crop using bounding box from annotation
                        # Note: Annotations are in MATLAB coordinates (1-indexed), inclusive
                        # Convert to 0-indexed numpy array
                        bbox = np.array(obj['bbox']) - 1

                        # Get visible and in-frame keypoints
                        keypoints = obj['anchors']
                        for keypoint_name in KEYPOINT_TYPES[class_name]:
                            # Get 0-indexed keypoint location
                            keypoint_loc_full = keypoints[keypoint_name]['location'] - 1
                            if keypoint_loc_full.size > 0 and insideBox(keypoint_loc_full, bbox):
                                # Keypoint is valid, so save data associated with it

                                # Get viewpoint label
                                viewpoint = obj['viewpoint']
                                azimuth = np.mod(np.round(viewpoint['azimuth']), 360)
                                elevation = np.mod(np.round(viewpoint['elevation']), 360)
                                tilt = np.mod(np.round(viewpoint['theta']), 360)
                                final_label = np.array([
                                    object_class,
                                    view2label(azimuth, object_class),
                                    view2label(elevation, object_class),
                                    view2label(tilt, object_class)]
                                )
                                # Add info for current keypoint
                                keypoint_class = KEYPOINTCLASS_INDEX_MAP[class_name + '_' + keypoint_name]
                                keypoint_str = keypointInfo2Str(full_image_path, bbox, keypoint_loc_full, keypoint_class, final_label)
                                if anno_file_set == 'train':
                                    info_file_train.write(keypoint_str)
                                else:
                                    info_file_test.write(keypoint_str)

    info_file_train.close()
    info_file_test.close()

def random_number_string(length=8):
    value = random.uniform(0, 10**length)
    format = '%0' + str(length) + 'd'
    return format % value

'''
@args
    start (float): The start time in seconds since the epoch
'''
def print_elapsed_time(start):
    now = time.time()
    elapsed_sec = now - start
    print('Time elapsed: %ds/%.1fm/%.1fh' % (elapsed_sec, elapsed_sec / 60.0, elapsed_sec / 3600.0))

def create_image_lmdb(image_data_root, image_lmdb_root, keys):
    start = time.time()

    # Get image paths and make sure they exist
    paths = [os.path.join(image_data_root, key + '.png') for key in keys]
    for path in paths:
        if not os.path.exists(path):
            print('Could not find path "%s", quitting' % path)
            return

    # Open image LMDB
    if not os.path.exists(image_lmdb_root):
        os.makedirs(image_lmdb_root)
    image_lmdb = lmdb.open(image_lmdb_root, map_size=DEFAULT_LMDB_SIZE)
    image_lmdb_name = os.path.basename(image_lmdb_root)

    # Save images in batch transactions
    num_batches = int(np.ceil(len(paths) / float(TXN_BATCH_SIZE)))
    for i in range(num_batches):
        start_idx = i * TXN_BATCH_SIZE
        end_idx = min((i+1) * TXN_BATCH_SIZE, len(paths))
        with image_lmdb.begin(write=True) as txn:
            for path in paths[start_idx:end_idx]:
                full_path = os.path.join(image_data_root, path)
                image = imread(full_path)
                datum = image_to_datum(image)
                key, _ = os.path.splitext(os.path.basename(path))
                txn.put(key.encode('ascii'), datum.SerializeToString())
        # Print progress every 10 batches
        if i % 10 == 0:
            print('%s: Committed batch %d/%d' % (image_lmdb_name, i+1, num_batches))
            print_elapsed_time(start)

    # Print completion info
    print('Finished creating %s' % image_lmdb_name)
    print_elapsed_time(start)

def create_vector_lmdb(vector_data_root, vector_lmdb_root, keys):
    start = time.time()

    # Get vector paths and make sure they exist
    paths = [os.path.join(vector_data_root, key + '.npy') for key in keys]
    for path in paths:
        if not os.path.exists(path):
            print('Could not find path "%s", quitting' % path)
            return

    # Open vector LMDB
    if not os.path.exists(vector_lmdb_root):
        os.makedirs(vector_lmdb_root)
    vector_lmdb = lmdb.open(vector_lmdb_root, map_size=DEFAULT_LMDB_SIZE)
    vector_lmdb_name = os.path.basename(vector_lmdb_root)

    # Save vectors in batch transactions
    num_batches = int(np.ceil(len(paths) / float(TXN_BATCH_SIZE)))
    for i in range(num_batches):
        start_idx = i * TXN_BATCH_SIZE
        end_idx = min((i+1) * TXN_BATCH_SIZE, len(paths))
        with vector_lmdb.begin(write=True) as txn:
            for path in paths[start_idx:end_idx]:
                full_path = os.path.join(vector_data_root, path)
                vector = np.load(full_path)
                if vector.dtype == np.float32:
                    vector = vector.astype(np.float)
                datum = vector_to_datum(vector)
                key, _ = os.path.splitext(os.path.basename(path))
                txn.put(key.encode('ascii'), datum.SerializeToString())
        # Print progress every 10 batches
        if i % 10 == 0:
            print('%s: Committed batch %d/%d' % (vector_lmdb_name, i+1, num_batches))

    # Print completion info
    print('Finished creating %s' % vector_lmdb_name)
    print_elapsed_time(start)

def create_tensor_lmdb(tensor_data_root, tensor_lmdb_root, keys):
    start = time.time()

    # Get tensor paths and make sure they exist
    paths = [os.path.join(tensor_data_root, key + '.npy') for key in keys]
    for path in paths:
        if not os.path.exists(path):
            print('Could not find path "%s", quitting' % path)
            return

    # Open tensor LMDB
    if not os.path.exists(tensor_lmdb_root):
        os.makedirs(tensor_lmdb_root)
    tensor_lmdb = lmdb.open(tensor_lmdb_root, map_size=DEFAULT_LMDB_SIZE)
    tensor_lmdb_name = os.path.basename(tensor_lmdb_root)

    # Save tensors in batch transactions
    num_batches = int(np.ceil(len(paths) / float(TXN_BATCH_SIZE)))
    for i in range(num_batches):
        start_idx = i * TXN_BATCH_SIZE
        end_idx = min((i+1) * TXN_BATCH_SIZE, len(paths))
        with tensor_lmdb.begin(write=True) as txn:
            for path in paths[start_idx:end_idx]:
                full_path = os.path.join(tensor_data_root, path)
                tensor = np.load(full_path)
                if tensor.dtype == np.float32:
                    tensor = tensor.astype(np.float)
                datum = tensor_to_datum(tensor)
                key, _ = os.path.splitext(os.path.basename(path))
                txn.put(key.encode('ascii'), datum.SerializeToString())
        # Print progress every 10 batches
        if i % 10 == 0:
            print('%s: Committed batch %d/%d' % (tensor_lmdb_name, i+1, num_batches))

    # Print completion info
    print('Finished creating %s' % tensor_lmdb_name)
    print_elapsed_time(start)

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