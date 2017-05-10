import numpy as np
import sys
import os
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lmdb
import pickle
import argparse
from matplotlib import gridspec
from matplotlib import rcParams
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import gen_lmdb_utils as utils
import global_variables as gv

import evaluateAcc
from eval_utils import softmax

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

def visualize_predictions(lmdbs_root, N, activation_cache_files, labels, output_dir=gv.g_qual_comp_folder, reorder_by_perf=False):
    keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = getCorrespLmdbData(lmdbs_root, N)
    activation_dicts = [pickle.load(open(f, 'rb')) for f in activation_cache_files]

    # Prepare plotting
    fig = plt.figure(figsize=(4, 3.5))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Set font size
    rcParams.update({'font.size': 15})
    # Set up array for storing errors
    azimuth_errs = np.zeros((len(keys), len(activation_dicts)))

    keys.sort()
    for k, key in enumerate(keys):
        # Extract data from LMDBs
        image = images_dict[key]
        keypoint_loc_image = keypoint_loc_dict[key]
        viewpoint_label_vec = viewpoint_label_dict[key]

        # Get data from true viewpoint label
        azimuth_label = viewpoint_label_vec[1] % 360

        for d, activation_dict in enumerate(activation_dicts):
            azimuth_activations = activation_dict['azimuth'][key]
            # Turn the activations into predictions and get azimuth prediction
            azimuth_preds = softmax(azimuth_activations)
            final_azimuth_pred = np.argmax(azimuth_preds)
            # Store the azimuth error
            azimuth_err = np.abs(final_azimuth_pred - azimuth_label)
            if azimuth_err > 180:
                azimuth_err = 180 - azimuth_err % 180
            azimuth_errs[k, d] = azimuth_err

            # Create heatmap for azimuth predictions
            azimuth_preds_vis = np.zeros((20, azimuth_preds.size, 3))
            for i in range(azimuth_preds.size):
                azimuth_preds_vis[:, i, :] = (azimuth_preds[i] - np.min(azimuth_preds))/(np.max(azimuth_preds) - np.min(azimuth_preds))/1.2
            # Mark predicted azimuth
            i_min = max(final_azimuth_pred-2, 0)
            i_max = min(final_azimuth_pred+2, 360)
            azimuth_preds_vis[:, i_min:i_max, :] = [1, 0, 1]

            # Plot viewpoint heatmap
            b = fig.add_axes([.3, .09*d+.07, .65, .07])
            b.imshow(azimuth_preds_vis, aspect='auto')
            # Hide tick marks and y labels
            b.tick_params(bottom='off', top='off', left='off', right='off', labelleft='off')
            # Only show angle labels for the bottom heatmap
            if d == 0:
                plt.setp(b, xticks=range(0, 361, 180))
            else:
                plt.setp(b.get_xticklabels(), visible=False)
            # Label the heatmap
            plt.text(-.35, .2, labels[d], transform=b.transAxes)

        # Plot ground truth marker
        c = fig.add_axes([.3, .09*(len(activation_dicts))+.03, .65, .07])
        c.axis('off')
        plt.scatter(azimuth_label, 0, color=(0, 1, 0), marker='v', s=50)
        plt.axis([0, 360, -1, 1])

        # Plot image and keypoint visualization
        a = fig.add_axes([0, .09*len(activation_dicts)+.05, 1, .97-.07*len(activation_dicts)-.05])
        a.imshow(image, aspect='auto')
        keypoint_loc_i, keypoint_loc_j = np.unravel_index(np.argmax(keypoint_loc_image), keypoint_loc_image.shape)
        a.imshow(np.ones((227, 227, 3)), alpha=0.3)
        a.axis('off')
        a.set_aspect('auto')
        plt.scatter(keypoint_loc_j, keypoint_loc_i, color='k', marker='*', s=900)
        plt.scatter(keypoint_loc_j, keypoint_loc_i, color=(1, 165/255., 0), marker='*', s=300)
        plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(a.get_yticklabels(), visible=False)

        # Save plot
        fig.savefig(os.path.join(output_dir, key + '.png'), dpi=100)
        plt.clf()

    if reorder_by_perf:
        assert(len(activation_dicts) == 2)
        azimuth_err_diffs = azimuth_errs[:, 0] - azimuth_errs[:, 1]
        azimuth_err_diffs_sorted_indexes = np.argsort(azimuth_err_diffs)
        for i, sorted_index in enumerate(azimuth_err_diffs_sorted_indexes):
            key = keys[sorted_index]
            new_file_name = '%06d_%s.png' % (i, key)
            os.rename(os.path.join(output_dir, key + '.png'), os.path.join(output_dir, new_file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('lmdbs_root', type=str, help='The root path of the correspondence LMDBs')
    parser.add_argument('num_examples', type=int, help='The number of examples to visualize')
    parser.add_argument('activation_cache_files_labels', type=str, nargs='*', help='Pairs of cached activation files and labels.')
    parser.add_argument('--reorder_by_perf', action='store_true', help='Whether to reorder the saved images by relative performance.')

    args = parser.parse_args()
    # Separate the cache files and labels
    activation_cache_files = args.activation_cache_files_labels[::2]
    labels = args.activation_cache_files_labels[1::2]
    assert(len(activation_cache_files) > 0)
    assert(len(labels) > 0)

    visualize_predictions(args.lmdbs_root, args.num_examples, activation_cache_files, labels, reorder_by_perf=args.reorder_by_perf)
