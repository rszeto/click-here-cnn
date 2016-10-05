import numpy as np
import sys
import os
import skimage
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import syn_image_utils as utils
import global_variables as gv

def viewCorrespLmdbs(lmdbs_root, N):
    keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = utils.getCorrespLmdbData(lmdbs_root, N)

    f, ax = plt.subplots(1, 3)
    plot_title = plt.suptitle('')
    for key in sorted(keys):
        image = images_dict[key]
        keypoint_loc_image = keypoint_loc_dict[key]
        keypoint_class_vec = keypoint_class_dict[key]
        viewpoint_label_vec = viewpoint_label_dict[key]

        # Create keypoint visualization
        keypoint_loc_i, keypoint_loc_j = np.where(keypoint_loc_image == 1)
        keypoint_loc_i = keypoint_loc_i[0]
        keypoint_loc_j = keypoint_loc_j[0]
        keypoint_vis = np.zeros((gv.g_images_resize_dim, gv.g_images_resize_dim, 3), dtype=np.uint8)
        rr,cc = skimage.draw.circle(keypoint_loc_i, keypoint_loc_j, 5, (gv.g_images_resize_dim, gv.g_images_resize_dim))
        keypoint_vis[rr,cc, :] = [255, 0, 0]

        # Get keypoint class
        keypoint_class = np.where(keypoint_class_vec == 1)[0][0]
        keypoint_name = utils.KEYPOINT_CLASSES[keypoint_class]

        # Get object class and angles
        object_class = viewpoint_label_vec[0]
        azimuth = viewpoint_label_vec[1] - 360*object_class
        elevation = viewpoint_label_vec[2] - 360*object_class
        tilt = viewpoint_label_vec[3] - 360*object_class

        ax[0].imshow(image)
        ax[1].imshow(keypoint_vis)
        ax[2].imshow(image)
        ax[2].imshow(keypoint_vis, alpha=0.5)

        # Set title
        title = ''
        title += 'Key: %s\n' % key
        title += '(Az, el, ti): (%d, %d, %d)\n' % (azimuth, elevation, tilt)
        title += 'Keypoint location (i,j): (%d, %d)\n' % (keypoint_loc_i, keypoint_loc_j)
        title += 'Keypoint class: %s (%d)' % (keypoint_name, keypoint_class)
        plot_title.set_text(title)

        # Hide axes
        plt.setp([a.get_xticklabels() for a in ax], visible=False)
        plt.setp([a.get_yticklabels() for a in ax], visible=False)
        plt.draw()
        plt.waitforbuttonpress()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python view_lmdbs.py <lmdb root name> <# examples>')
        exit()

    lmdbs_root_name = sys.argv[1]
    num_examples = int(sys.argv[2])
    viewCorrespLmdbs(os.path.join(gv.g_corresp_folder, lmdbs_root_name), num_examples)