import numpy as np
import sys
import os
import skimage
import matplotlib.pyplot as plt
import lmdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import gen_lmdb_utils as utils
import global_variables as gv

def getCorrespLmdbData(lmdbs_root, N):
    # Define LMDBs
    image_lmdb = lmdb.open(os.path.join(lmdbs_root, 'image_lmdb'), readonly=True)
    keypoint_loc_lmdb = lmdb.open(os.path.join(lmdbs_root, 'keypoint_loc_lmdb'), readonly=True)
    keypoint_class_lmdb = lmdb.open(os.path.join(lmdbs_root, 'keypoint_class_lmdb'), readonly=True)
    viewpoint_label_lmdb = lmdb.open(os.path.join(lmdbs_root, 'viewpoint_label_lmdb'), readonly=True)

    images_dict = utils.getFirstNLmdbImgs(image_lmdb, N)
    keypoint_loc_dict = utils.getFirstNLmdbImgs(keypoint_loc_lmdb, N)
    keypoint_class_dict = utils.getFirstNLmdbVecs(keypoint_class_lmdb, N)
    viewpoint_label_dict = utils.getFirstNLmdbVecs(viewpoint_label_lmdb, N)

    return images_dict.keys(), images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict

def viewCorrespLmdbs(lmdbs_root, N):
    keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = getCorrespLmdbData(lmdbs_root, N)

    f, ax = plt.subplots(1, 3)
    plot_title = plt.suptitle('')
    for key in sorted(keys):
        image = images_dict[key]
        keypoint_loc_image = keypoint_loc_dict[key]
        keypoint_class_vec = keypoint_class_dict[key]
        viewpoint_label_vec = viewpoint_label_dict[key]

        # Create keypoint visualization
        keypoint_loc_i, keypoint_loc_j = np.where(keypoint_loc_image != 0)
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
        title += '(Obj_cls, az, el, ti): (%d, %d, %d, %d)\n' % (object_class, azimuth, elevation, tilt)
        title += 'Keypoint location (i,j): (%d, %d)\n' % (keypoint_loc_i, keypoint_loc_j)
        title += 'Keypoint class: %s (%d)' % (keypoint_name, keypoint_class)
        plot_title.set_text(title)

        # Hide axes
        plt.setp([a.get_xticklabels() for a in ax], visible=False)
        plt.setp([a.get_yticklabels() for a in ax], visible=False)
        plt.draw()
        plt.waitforbuttonpress()

def viewImageLmdbs(image_lmdb_root, N):
    image_lmdb = lmdb.open(image_lmdb_root, readonly=True)
    images_dict = utils.getFirstNLmdbImgs(image_lmdb, N)
    keys = images_dict.keys()

    f = plt.figure()
    plot_title = plt.suptitle('')
    for key in sorted(keys):
        image = images_dict[key]
        plt.imshow(image, cmap='Greys_r')

        # Set title
        title = ''
        title += 'Key: %s\n' % key
        plot_title.set_text(title)

        # Hide axes
        plt.axis('off')
        plt.draw()
        plt.waitforbuttonpress()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python view_lmdbs.py <root path of lmdb(s)> <# examples> <mode>')
        print('Mode can be either "correspondences" or "images"')
        exit()

    if sys.argv[3] == 'correspondences':
        lmdbs_root_path = sys.argv[1]
        num_examples = int(sys.argv[2])
        viewCorrespLmdbs(lmdbs_root_path, num_examples)
    elif sys.argv[3] == 'images':
        image_lmdb_root = sys.argv[1]
        num_examples = int(sys.argv[2])
        viewImageLmdbs(image_lmdb_root, num_examples)
    else:
        print('Unrecognized mode ' + sys.argv[3])
