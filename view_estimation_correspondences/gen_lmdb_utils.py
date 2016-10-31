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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import global_variables as gv
# sys.path.append(os.path.join(BASE_DIR, '..', 'view_estimation'))

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

def image_to_datum(image):
    # Convert image to weird Caffe format. Fork depending on if image is color or not.
    if image.ndim == 2:
        # If grayscale, transpose to match Caffe dim ordering
        image_caffe = image.reshape((1, image.shape[0], image.shape[1]))
    else:
        # If color, switch from RGB to BGR
        image_caffe = image[:, :, ::-1]
        # Go from H*W*C to C*H*W
        image_caffe = image_caffe.transpose(2, 0, 1)
    # Populate datum
    datum = caffe.proto.caffe_pb2.Datum()
    (datum.channels, datum.height, datum.width) = image_caffe.shape
    datum.data = image_caffe.tobytes()
    return datum

def write_image_to_lmdb(lmdb_obj, key, image):
    datum = image_to_datum(image)
    # Put datum into LMDB
    with lmdb_obj.begin(write=True) as txn:
        txn.put(key.encode('ascii'), datum.SerializeToString())

def vector_to_datum(vec):
    # Reshape vector to be in Caffe format
    vec_caffe = vec.reshape([len(vec), 1, 1])
    # Put datum into LMDB
    datum = caffe.io.array_to_datum(vec_caffe)
    return datum

def write_vec_to_lmdb(lmdb_obj, key, vec):
    datum = vector_to_datum(vec)
    with lmdb_obj.begin(write=True) as txn:
        txn.put(key.encode('ascii'), datum.SerializeToString())

def write_datum_to_lmdb(lmdb_obj, key, datum):
    # Put datum into LMDB
    with lmdb_obj.begin(write=True) as txn:
        txn.put(key.encode('ascii'), datum.SerializeToString())

'''
@input: (full_image_path, bbox, reverse)
    - full_image_path: The path to the full image
    - bbox: A NumPy array of the form [tl_x, tl_y, br_x, br_y] indicating the corners of the crop, inclusive
    - reverse: Boolean flag indicating whether to reverse the image
'''
def full_image_path_to_datum(args):
    full_image_path, bbox, reverse = args

    # Get the cropped image, scale it, and store it
    full_image = imread(full_image_path)
    # Convert grayscale images to "color"
    if full_image.ndim == 2:
        full_image = np.dstack((full_image, full_image, full_image))
    cropped_image = full_image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
    scaled_image = imresize(cropped_image, (gv.g_images_resize_dim, gv.g_images_resize_dim))
    if reverse:
        scaled_image = np.fliplr(scaled_image)
    datum = image_to_datum(scaled_image)
    return datum

######### Loading from LMDB (for debugging) ##################################

def lmdbStrToImage(lmdbStr):
    # Get datum from LMDB
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(lmdbStr)
    # Retrieve image in weird Caffe format
    img = caffe.io.datum_to_array(datum)
    # Change C*H*W to H*W*C
    img = img.transpose((1, 2, 0))
    # Squeeze extra dimension if grayscale
    img = img.squeeze()
    # Change BGR to RGB if image is in color
    if img.ndim == 3:
        img = img[:, :, ::-1]
    return img

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

def appendRandomPrefix(key):
    prefix = random.uniform(0, 1e8)
    return '%08d_%s' % (prefix, key)

'''
@input: (line, reverse)
    - line: The line from the LMDB info CSV
    - reverse: Boolean flag for whether the data should be reversed
@output: The datum representing the scaled image
'''
def csvLineToImageDatum(arg_tuple):
    line, reverse = arg_tuple

    # Extract info from the line
    m = re.match(LINE_FORMAT, line)
    full_image_path = m.group(1)
    bbox = np.array([int(x) for x in m.group(2,3,4,5)])

    # Get the cropped image and scale it
    full_image = imread(full_image_path)
    # Convert grayscale images to "color"
    if full_image.ndim == 2:
        full_image = np.dstack((full_image, full_image, full_image))
    cropped_image = full_image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
    scaled_image = imresize(cropped_image, (gv.g_images_resize_dim, gv.g_images_resize_dim))
    # Flip the image if needed
    if reverse:
        scaled_image = np.fliplr(scaled_image)

    # Return the datum
    datum = image_to_datum(scaled_image)
    return datum

'''
@input: (line, reverse)
    - line: The line from the LMDB info CSV
    - reverse: Boolean flag for whether the data should be reversed
@output: The datum representing the keypoint image
'''
def csvLineToKeypointImageDatum(arg_tuple):
    line, reverse = arg_tuple

    # Extract info from the line
    m = re.match(LINE_FORMAT, line)
    full_image_path = m.group(1)
    bbox = np.array([int(x) for x in m.group(2,3,4,5)])
    keypoint_loc_full = np.array([float(x) for x in m.group(6,7)])

    # Get bounding box dimensions
    bbox_size = np.array([bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1])
    # Get keypoint location inside the bounding box
    keypoint_loc_bb = keypoint_loc_full - bbox[:2]
    keypoint_loc_scaled = np.floor(gv.g_images_resize_dim * keypoint_loc_bb / bbox_size).astype(np.uint8)
    # Push keypoint inside image (sometimes it ends up on edge due to float arithmetic)
    keypoint_loc_scaled[0] = min(keypoint_loc_scaled[0], gv.g_images_resize_dim-1)
    keypoint_loc_scaled[1] = min(keypoint_loc_scaled[1], gv.g_images_resize_dim-1)
    # Create keypoint location image
    keypoint_image = np.zeros((gv.g_images_resize_dim, gv.g_images_resize_dim), dtype=np.uint8)
    keypoint_image[keypoint_loc_scaled[1], keypoint_loc_scaled[0]] = 1
    # Flip the image if needed
    if reverse:
        keypoint_image = np.fliplr(keypoint_image)

    # Return the datum
    datum = image_to_datum(keypoint_image)
    return datum

'''
@input: (line, reverse)
    - line: The line from the LMDB info CSV
    - reverse: Boolean flag for whether the data should be reversed
@output: The datum representing the one-hot keypoint class vector
'''
def csvLineToKeypointClassDatum(arg_tuple):
    line, reverse = arg_tuple

    # Extract info from the line
    m = re.match(LINE_FORMAT, line)
    keypoint_class = int(m.group(8))

    # Get mirror keypoint class if needed
    if reverse:
        keypoint_name = KEYPOINT_CLASSES[keypoint_class]
        keypoint_name_r = keypoint_name
        if 'left' in keypoint_name:
            keypoint_name_r = keypoint_name.replace('left', 'right')
        elif 'right' in keypoint_name:
            keypoint_name_r = keypoint_name.replace('right', 'left')
        keypoint_class = KEYPOINTCLASS_INDEX_MAP[keypoint_name_r]

    # Get one-hot vector encoding of keypoint class
    keypoint_class_vec = np.zeros(len(KEYPOINT_CLASSES), dtype=np.uint8)
    keypoint_class_vec[keypoint_class] = 1

    datum = vector_to_datum(keypoint_class_vec)
    return datum

'''
@input: (line, reverse)
    - line: The line from the LMDB info CSV
    - reverse: Boolean flag for whether the data should be reversed
@output: The datum representing the viewpoint label
'''
def csvLineToViewpointLabelDatum(arg_tuple):
    line, reverse = arg_tuple

    # Extract info from the line
    m = re.match(LINE_FORMAT, line)
    viewpoint_label = np.array([int(x) for x in m.group(9,10,11,12)])

    # Save label for regular image
    viewpoint_label_vec = viewpoint_label
    # Get viewpoint label of flipped image if needed
    if reverse:
        # Extract normal azimuth and tilt
        object_class = viewpoint_label_vec[0]
        azimuth = viewpoint_label_vec[1]
        tilt = viewpoint_label_vec[3]
        # Get reversed azimuth and tilt
        azimuth_r = np.mod(360-azimuth, 360)
        tilt_r = np.mod(-1*tilt, 360)
        # Update viewpoint label
        viewpoint_label_vec[1] = view2label(azimuth_r, object_class)
        viewpoint_label_vec[3] = view2label(tilt_r, object_class)

    datum = vector_to_datum(viewpoint_label_vec)
    return datum

def createCorrespLmdbsMinor(args):
    proc_num, minor_jobs, lmdbs_minor_root = args
    start = time.time()

    if not os.path.exists(lmdbs_minor_root):
        os.mkdir(lmdbs_minor_root)

    # Define LMDBs
    image_lmdb = lmdb.open(os.path.join(lmdbs_minor_root, 'image_lmdb'), map_size=DEFAULT_LMDB_SIZE)
    keypoint_loc_lmdb = lmdb.open(os.path.join(lmdbs_minor_root, 'keypoint_loc_lmdb'), map_size=DEFAULT_LMDB_SIZE)
    keypoint_class_lmdb = lmdb.open(os.path.join(lmdbs_minor_root, 'keypoint_class_lmdb'), map_size=DEFAULT_LMDB_SIZE)
    viewpoint_label_lmdb = lmdb.open(os.path.join(lmdbs_minor_root, 'viewpoint_label_lmdb'), map_size=DEFAULT_LMDB_SIZE)

    for i, minor_job in enumerate(minor_jobs):
        line, reversed = minor_job
        image_datum = csvLineToImageDatum(minor_job)
        keypoint_image_datum = csvLineToKeypointImageDatum(minor_job)
        keypoint_class_datum = csvLineToKeypointClassDatum(minor_job)
        viewpoint_label_datum = csvLineToViewpointLabelDatum(minor_job)

        # Extract info from the line
        m = re.match(LINE_FORMAT, line)
        full_image_path = m.group(1)
        # Get keys for regular and reversed instance
        image_name = os.path.basename(full_image_path)
        if reversed:
            key = appendRandomPrefix(image_name + '_r')
        else:
            key = appendRandomPrefix(image_name)

        # Write datums
        write_datum_to_lmdb(image_lmdb, key, image_datum)
        write_datum_to_lmdb(keypoint_loc_lmdb, key, keypoint_image_datum)
        write_datum_to_lmdb(keypoint_class_lmdb, key, keypoint_class_datum)
        write_datum_to_lmdb(viewpoint_label_lmdb, key, viewpoint_label_datum)

        if i % 1000 == 0:
            now = time.time()
            elapsed_sec = now - start
            print('Finished writing keypoint %d/%d (proc %d)' % (i, len(minor_jobs), proc_num))
            print('Time elapsed: %ds/%.1fm/%.1fh' % (elapsed_sec, elapsed_sec / 60.0, elapsed_sec / 3600.0))

'''
TODO: Randomness cannot be seeded easily since the keys are generated in each minor call. If time, move key generation
to major call and push the keys to the minor function
'''
def createCorrespLmdbsMajor(info_file_path, lmdbs_major_root, num_procs, reverse):
    start = time.time()
    print('Generating LMDBs from CSV')
    print('Info file path: %s' % info_file_path)
    print('LMDBs root: %s' % lmdbs_major_root)
    print('Start date: %s' % time.asctime(time.localtime(start)))

    if not os.path.exists(lmdbs_major_root):
        os.mkdir(lmdbs_major_root)

    # Read the data info from the file
    with open(info_file_path) as info_file:
        lines = info_file.readlines()
    # Remove header row
    lines = lines[1:]

    # Define functions to go from line to argument tuples
    line_to_arg_tuple = lambda l: (l, False)
    line_to_arg_tuple_r = lambda l: (l, True)
    # Generate all jobs
    if reverse:
        l2at_fs = (line_to_arg_tuple, line_to_arg_tuple_r)
        all_minor_jobs = [f(line) for line in lines for f in l2at_fs]
    else:
        all_minor_jobs = [line_to_arg_tuple(line) for line in lines]
    # Randomize the jobs across processors
    random.shuffle(all_minor_jobs)

    # Divide the minor jobs among the processors
    num_jobs_per_proc = len(all_minor_jobs) / num_procs
    remainder = len(all_minor_jobs) % num_procs
    divided_minor_jobs = []
    for i in range(num_procs):
        if i < remainder:
            minor_jobs = all_minor_jobs[i*num_jobs_per_proc+i:(i+1)*num_jobs_per_proc+i+1]
        else:
            minor_jobs = all_minor_jobs[i*num_jobs_per_proc+remainder:(i+1)*num_jobs_per_proc+remainder]
        divided_minor_jobs.append((i, minor_jobs, os.path.join(lmdbs_major_root, str(i))))

    p = multiprocessing.Pool(num_procs)
    p.map(createCorrespLmdbsMinor, divided_minor_jobs)

    now = time.time()
    elapsed_sec = now - start
    print('End date: %s' % time.asctime(time.localtime(now)))
    print('Time elapsed: %ds/%.1fm/%.1fh' % (elapsed_sec, elapsed_sec / 60.0, elapsed_sec / 3600.0))

def createCorrespLmdbs(info_file_path, lmdbs_root, reverse):
    start = time.time()
    print('Generating LMDBs from CSV')
    print('Info file path: %s' % info_file_path)
    print('LMDBs root: %s' % lmdbs_root)
    print('Start date: %s' % time.asctime(time.localtime(start)))

    if not os.path.exists(lmdbs_root):
        os.mkdir(lmdbs_root)

    # Read the data info from the file
    with open(info_file_path) as info_file:
        lines = info_file.readlines()
    # Remove header row
    lines = lines[1:]

    # Define functions to go from line to argument tuples
    line_to_arg_tuple = lambda l: (l, False)
    line_to_arg_tuple_r = lambda l: (l, True)
    # Generate all jobs
    if reverse:
        l2at_fs = (line_to_arg_tuple, line_to_arg_tuple_r)
        all_jobs = [f(line) for line in lines for f in l2at_fs]
    else:
        all_jobs = [line_to_arg_tuple(line) for line in lines]
    # Randomize the jobs
    random.shuffle(all_jobs)

    # Define LMDBs
    image_lmdb = lmdb.open(os.path.join(lmdbs_root, 'image_lmdb'), map_size=DEFAULT_LMDB_SIZE)
    keypoint_loc_lmdb = lmdb.open(os.path.join(lmdbs_root, 'keypoint_loc_lmdb'), map_size=DEFAULT_LMDB_SIZE)
    keypoint_class_lmdb = lmdb.open(os.path.join(lmdbs_root, 'keypoint_class_lmdb'), map_size=DEFAULT_LMDB_SIZE)
    viewpoint_label_lmdb = lmdb.open(os.path.join(lmdbs_root, 'viewpoint_label_lmdb'), map_size=DEFAULT_LMDB_SIZE)

    for i, job in enumerate(all_jobs):
        line, reversed = job
        image_datum = csvLineToImageDatum(job)
        keypoint_image_datum = csvLineToKeypointImageDatum(job)
        keypoint_class_datum = csvLineToKeypointClassDatum(job)
        viewpoint_label_datum = csvLineToViewpointLabelDatum(job)

        # Extract info from the line
        m = re.match(LINE_FORMAT, line)
        full_image_path = m.group(1)
        # Get keys for regular and reversed instance
        image_name = os.path.basename(full_image_path)
        if reversed:
            key = appendRandomPrefix(image_name + '_r')
        else:
            key = appendRandomPrefix(image_name)

        # Write datums
        write_datum_to_lmdb(image_lmdb, key, image_datum)
        write_datum_to_lmdb(keypoint_loc_lmdb, key, keypoint_image_datum)
        write_datum_to_lmdb(keypoint_class_lmdb, key, keypoint_class_datum)
        write_datum_to_lmdb(viewpoint_label_lmdb, key, viewpoint_label_datum)

        if i % 1000 == 0:
            now = time.time()
            elapsed_sec = now - start
            print('Finished writing keypoint %d/%d' % (i, len(all_jobs)))
            print('Time elapsed: %ds/%.1fm/%.1fh' % (elapsed_sec, elapsed_sec / 60.0, elapsed_sec / 3600.0))

    now = time.time()
    elapsed_sec = now - start
    print('End date: %s' % time.asctime(time.localtime(now)))
    print('Time elapsed: %ds/%.1fm/%.1fh' % (elapsed_sec, elapsed_sec / 60.0, elapsed_sec / 3600.0))

def getCorrespLmdbData(lmdbs_root, N):
    # Define LMDBs
    image_lmdb = lmdb.open(os.path.join(lmdbs_root, 'image_lmdb'), readonly=True)
    keypoint_loc_lmdb = lmdb.open(os.path.join(lmdbs_root, 'keypoint_loc_lmdb'), readonly=True)
    keypoint_class_lmdb = lmdb.open(os.path.join(lmdbs_root, 'keypoint_class_lmdb'), readonly=True)
    viewpoint_label_lmdb = lmdb.open(os.path.join(lmdbs_root, 'viewpoint_label_lmdb'), readonly=True)

    images_dict = getFirstNLmdbImgs(image_lmdb, N)
    keypoint_loc_dict = getFirstNLmdbImgs(keypoint_loc_lmdb, N)
    keypoint_class_dict = getFirstNLmdbVecs(keypoint_class_lmdb, N)
    viewpoint_label_dict = getFirstNLmdbVecs(viewpoint_label_lmdb, N)

    return images_dict.keys(), images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict
