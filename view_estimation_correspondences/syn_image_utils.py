import numpy as np
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

# synset_name_pairs = gv.g_shape_synset_name_pairs
synset_name_pairs = [('02924116', 'bus'), ('02958343', 'car'), ('03790512', 'motorbike')]

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


'''
@brief:
    convert 360 view degree to view estimation label
    e.g. for bicycle with class_idx 1, label will be 360~719
'''
def view2label(degree, class_index):
  return int(degree)%360 + class_index*360

def createSynKeypointCsv():
    infoFile = open(os.path.join(BASE_DIR, 'syn_corresp_lmdb_info.csv'), 'w')
    infoFile.write('imgPath,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
    imgNamePattern = re.compile('(.*)_(.*)_a(.*)_e(.*)_t(.*)_d(.*).jpg')
    for synset in os.listdir(gv.g_syn_images_bkg_overlaid_folder):
        print(synset)
        for md5 in os.listdir(os.path.join(gv.g_syn_images_bkg_overlaid_folder, synset)):
            print(md5)
            for image_path in glob(os.path.join(gv.g_syn_images_bkg_overlaid_folder, synset, md5, '*.jpg')):

                # Get class index from synset
                object_class = SYNSET_CLASSIDX_MAP[synset]
                class_name = SYNSET_CLASSNAME_MAP[synset]

                # Extract azimuth, elevation, rotation from file name
                image_name = os.path.basename(image_path)
                m = re.match(imgNamePattern, image_name)
                azimuth, elevation, tilt = m.group(3, 4, 5)

                # Get CSV that lists keypoint locations in the image
                csvPath = image_path.replace('.jpg', '_keypoint2d.csv')
                keypointData = np.genfromtxt(csvPath, delimiter=',', dtype=None)
                for row in keypointData:
                    keypoint_name, x, y = row
                    keypoint_class = KEYPOINTCLASS_INDEX_MAP[class_name + '_' + keypoint_name]
                    infoFile.write('%s,%d,%d,%d,%d,%d,%d,%d\n' % (image_path, x, y, keypoint_class, object_class, view2label(azimuth, object_class), view2label(elevation, object_class), view2label(tilt, object_class)))
    infoFile.close()

if __name__ == '__main__':
    createSynKeypointCsv()