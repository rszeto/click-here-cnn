import numpy as np
import scipy.io as spio
import os
import sys
import re
from glob import glob
from scipy.ndimage import imread

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import global_variables as gv
sys.path.append(os.path.join(BASE_DIR, '..', 'view_estimation'))

RENDER4CNN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PASCAL3D_ROOT = os.path.join(RENDER4CNN_ROOT, 'datasets', 'pascal3d')
ANNOTATIONS_ROOT = os.path.join(PASCAL3D_ROOT, 'Annotations')
IMAGES_ROOT = os.path.join(PASCAL3D_ROOT, 'Images')

# synset_name_pairs = gv.g_shape_synset_name_pairs
synset_name_pairs = [('02924116', 'bus'), ('02958343', 'car'), ('03790512', 'motorbike')]

DATASET_SOURCES = ['pascal', 'imagenet']

KEYPOINT_TYPES = {
    'aeroplane': ['left_elevator', 'left_wing', 'noselanding', 'right_elevator', 'right_wing', 'rudder_lower', 'rudder_upper', 'tail'],
    'bicycle': ['head_center', 'left_back_wheel', 'left_front_wheel', 'left_handle', 'left_pedal_center', 'right_back_wheel', 'right_front_wheel', 'right_handle', 'right_pedal_center', 'seat_back', 'seat_front'],
    'boat': ['head', 'head_down', 'head_left', 'head_right', 'tail_left', 'tail_right', 'tail'],
    'bottle': ['mouth', 'body', 'body_left', 'body_right', 'bottom', 'bottom_left', 'bottom_right'],
    'bus': ['body_back_left_lower', 'body_back_left_upper', 'body_back_right_lower', 'body_back_right_upper', 'body_front_left_upper', 'body_front_right_upper', 'body_front_left_lower', 'body_front_right_lower', 'left_back_wheel', 'left_front_wheel', 'right_back_wheel', 'right_front_wheel'],
    'car': ['left_front_wheel', 'left_back_wheel', 'right_front_wheel', 'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield', 'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light', 'right_front_light', 'left_back_trunk', 'right_back_trunk'],
    'chair': ['back_upper_left', 'back_upper_right', 'seat_upper_left', 'seat_upper_right', 'seat_lower_left', 'seat_lower_right', 'leg_upper_left', 'leg_upper_right', 'leg_lower_left', 'leg_lower_right'],
    'diningtable': ['leg_upper_left', 'leg_upper_right', 'leg_lower_left', 'leg_lower_right', 'top_upper_left', 'top_upper_right', 'top_lower_left', 'top_lower_right', 'top_up', 'top_down', 'top_left', 'top_right'],
    'motorbike': ['back_seat', 'front_seat', 'head_center', 'headlight_center', 'left_back_wheel', 'left_front_wheel', 'left_handle_center', 'right_back_wheel', 'right_front_wheel', 'right_handle_center'],
    'sofa': ['front_bottom_left', 'front_bottom_right', 'seat_bottom_left', 'seat_bottom_right', 'left_bottom_back', 'right_bottom_back', 'top_left_corner', 'top_right_corner', 'seat_top_left', 'seat_top_right'],
    'train': ['head_left_bottom', 'head_left_top', 'head_right_bottom', 'head_right_top', 'head_top', 'mid1_left_bottom', 'mid1_left_top', 'mid1_right_bottom', 'mid1_right_top', 'mid2_left_bottom', 'mid2_left_top', 'mid2_right_bottom', 'mid2_right_top', 'tail_left_bottom', 'tail_left_top', 'tail_right_bottom', 'tail_right_top'],
    'tvmonitor': ['front_bottom_left', 'front_bottom_right', 'front_top_left', 'front_top_right', 'back_bottom_left', 'back_bottom_right', 'back_top_left', 'back_top_right']
}

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
    infoFile = open(os.path.join(BASE_DIR, 'syn_corresp_lmdb_info.csv'), 'w')
    infoFile.write(INFO_FILE_HEADER)
    imgNamePattern = re.compile('(.*)_(.*)_a(.*)_e(.*)_t(.*)_d(.*).jpg')
    for synset in os.listdir(gv.g_syn_images_bkg_overlaid_folder):
        for md5 in os.listdir(os.path.join(gv.g_syn_images_bkg_overlaid_folder, synset)):
            for image_path in glob(os.path.join(gv.g_syn_images_bkg_overlaid_folder, synset, md5, '*.jpg')):
                # Get bounding box for object (in this case, whole image)
                image = imread(image_path)
                bbox = np.array([0, 0, image.shape[1]-1, image.shape[0]-1])

                # Get class index from synset
                object_class = SYNSET_CLASSIDX_MAP[synset]
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
    matlab_cmd = 'getPascalTrainValImgs'
    os.system('matlab -nodisplay -r "try %s ; catch; end; quit;"' % matlab_cmd)
    # Get training and test image IDs
    with open('trainImgIds.txt', 'rb') as trainIdsFile:
        trainIds = np.loadtxt(trainIdsFile, dtype='string')
    with open('valImgIds.txt', 'rb') as testIdsFile:
        testIds = np.loadtxt(testIdsFile, dtype='string')
    # Delete the ID files
    os.remove('trainImgIds.txt')
    os.remove('valImgIds.txt')

    infoFileTrain = open(os.path.join(BASE_DIR, 'pascal_corresp_lmdb_info_train.csv'), 'w')
    infoFileTrain.write(INFO_FILE_HEADER)
    infoFileTest = open(os.path.join(BASE_DIR, 'pascal_corresp_lmdb_info_test.csv'), 'w')
    infoFileTest.write(INFO_FILE_HEADER)

    for synset, class_name in synset_name_pairs:
        object_class = SYNSET_CLASSIDX_MAP[synset]
        for dataset_source in DATASET_SOURCES:
            classSourceId = '%s_%s' % (class_name, dataset_source)
            for annoFile in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, classSourceId))):
                annoFileId = os.path.splitext(os.path.basename(annoFile))[0]
                if annoFileId in trainIds:
                    annoFileSet = 'train'
                elif annoFileId in testIds:
                    annoFileSet = 'test'
                else:
                    continue

                anno = loadmat(os.path.join(ANNOTATIONS_ROOT, classSourceId, annoFile))['record']
                fullImagePath = os.path.join(IMAGES_ROOT, classSourceId, anno['filename'])
                fullImage = imread(fullImagePath)
                # Convert grayscale images to "color"
                if fullImage.ndim == 2:
                    fullImage = np.dstack((fullImage, fullImage, fullImage))

                # Make objs an array regardless of how many objects there are
                objs = np.array([anno['objects']]) if isinstance(anno['objects'], dict) else anno['objects']
                for objI, obj in enumerate(objs):
                    # Only deal with objects in current class
                    if obj['class'] == class_name:
                        # Get crop using bounding box from annotation
                        # Note: Annotations are in MATLAB coordinates (1-indexed), inclusive
                        # Convert to 0-indexed numpy array
                        bbox = np.array(obj['bbox']) - 1

                        # Get visible and in-frame keypoints
                        keypts = obj['anchors']
                        for keypoint_name in KEYPOINT_TYPES[class_name]:
                            # Get 0-indexed keypoint location
                            keyptLocFull = keypts[keypoint_name]['location'] - 1
                            if keyptLocFull.size > 0 and insideBox(keyptLocFull, bbox):
                                # Keypoint is valid, so save data associated with it

                                # Get viewpoint label
                                viewpoint = obj['viewpoint']
                                azimuth = np.mod(np.round(viewpoint['azimuth']), 360)
                                elevation = np.mod(np.round(viewpoint['elevation']), 360)
                                tilt = np.mod(np.round(viewpoint['theta']), 360)
                                finalLabel = np.array([
                                    object_class,
                                    view2label(azimuth, object_class),
                                    view2label(elevation, object_class),
                                    view2label(tilt, object_class)]
                                )
                                # Add info for current keypoint
                                keypoint_class = KEYPOINTCLASS_INDEX_MAP[class_name + '_' + keypoint_name]
                                keyptStr = keypointInfo2Str(fullImagePath, bbox, keyptLocFull, keypoint_class, finalLabel)
                                if annoFileSet == 'train':
                                    infoFileTrain.write(keyptStr)
                                else:
                                    infoFileTest.write(keyptStr)

    infoFileTrain.close()
    infoFileTest.close()

if __name__ == '__main__':
    createSynKeypointCsv()
    createPascalKeypointCsv()