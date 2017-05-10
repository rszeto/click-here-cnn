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
from evaluateAcc import softmax

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

def visualize_predictions(lmdbs_root, N, activation_cache_files, output_dir, labels=None, reorder_by_perf=False):
    keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = getCorrespLmdbData(lmdbs_root, N)
    activation_dicts = [pickle.load(open(f, 'rb')) for f in activation_cache_files]

    # Prepare plotting
    fig = plt.figure(figsize=(4, 3.5))
    if output_dir and not os.path.exists(output_dir):
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
        keypoint_class_vec = keypoint_class_dict[key]
        viewpoint_label_vec = viewpoint_label_dict[key]

        # Get data from true viewpoint label
        obj_class = viewpoint_label_vec[0]
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
            # # Mark ground truth label
            # i_min = max(azimuth_label-3, 0)
            # i_max = min(azimuth_label+3, 360)
            # azimuth_preds_vis[:, i_min:i_max, :] = [1, 0, 1]
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
            if labels:
                text = plt.text(-.35, .2, labels[d], transform=b.transAxes)

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

        if output_dir:
            # Save plot
            fig.savefig(os.path.join(output_dir, key + '.png'), dpi=100)
        else:
            # Draw plot
            plt.draw()
            plt.waitforbuttonpress()
        plt.clf()

    if output_dir and reorder_by_perf:
        assert(len(activation_dicts) == 2)
        azimuth_err_diffs = azimuth_errs[:, 0] - azimuth_errs[:, 1]
        azimuth_err_diffs_sorted_indexes = np.argsort(azimuth_err_diffs)
        for i, sorted_index in enumerate(azimuth_err_diffs_sorted_indexes):
            key = keys[sorted_index]
            new_file_name = '%06d_%s.png' % (i, key)
            os.rename(os.path.join(output_dir, key + '.png'), os.path.join(output_dir, new_file_name))

def plot_error_distributions(lmdbs_root, N, activation_cache_files, output_dir, labels=None, zoom=False):
    keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = getCorrespLmdbData(lmdbs_root, N)
    activation_dicts = [pickle.load(open(f, 'rb')) for f in activation_cache_files]

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert labels to numpy array for comparing against activations (which are returned as a matrix)
    viewpoint_labels_as_mat = np.zeros((len(keys), 4))
    for i, key in enumerate(keys):
        viewpoint_labels_as_mat[i, :] = viewpoint_label_dict[key]
    # Extract the angle activations for the correct object class
    obj_classes = viewpoint_labels_as_mat[:, 0]

    # Prepare plot
    # plt.figure(figsize=(5, 4))
    # plt.axis([0, 2.5, 0, 2000])

    angle_names = ['azimuth', 'elevation', 'tilt']
    for i, activation_dict in enumerate(activation_dicts):
        full_activations = []
        for angle_name in angle_names:
            arr = []
            for key in keys:
                arr.append(activation_dict[angle_name][key])
            full_activations.append(arr)

        preds = evaluateAcc.activations_to_preds(full_activations, obj_classes)
        angle_dists = evaluateAcc.compute_angle_dists(preds, viewpoint_labels_as_mat)

        fig = plt.figure(figsize=(4, 4))
        ax = plt.subplot(111)
        plt.hist(angle_dists, bins=30)
        fig_title = labels[i] if labels else 'figure_%d' % i
        plt.title(fig_title)
        plt.xlabel('Error (rad)')
        plt.tight_layout()

        if zoom:
            plt.axis([1.5, 2.25, 0, 250])
        else:
            plt.axis([0, 2.25, 0, 3000])
            p = patches.Rectangle((1.5, 0), 2.25, 250, fill=False, linestyle='dashed')
            ax.add_patch(p)

        if output_dir:
            # Save plot
            filename = fig_title + '-z.eps' if zoom else fig_title + '.eps'
            fig.savefig(os.path.join(output_dir, filename), dpi=300)

    if not output_dir:
        plt.show()


def visualize_predictions_confidence(lmdbs_root, N, activation_cache_files, output_dir, key, labels=None, reorder_by_perf=False):
    '''
    Generate histogram and heatmap of azimuth confidences for the given instance (Figure 1 in ICCV submission)
    '''
    keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = getCorrespLmdbData(lmdbs_root, N)
    activation_dicts = [pickle.load(open(f, 'rb')) for f in activation_cache_files]

    # Prepare plotting
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Set font size
    rcParams.update({'font.size': 15})
    # Set up array for storing errors
    azimuth_errs = np.zeros((len(keys), len(activation_dicts)))

    # Extract data from LMDBs
    image = images_dict[key]
    keypoint_loc_image = keypoint_loc_dict[key]
    keypoint_class_vec = keypoint_class_dict[key]
    viewpoint_label_vec = viewpoint_label_dict[key]

    # Get data from true viewpoint label
    obj_class = viewpoint_label_vec[0]
    azimuth_label = viewpoint_label_vec[1] % 360

    # Show image with keypoint
    f = plt.figure('image_kp', facecolor='white', figsize=(4, 4))
    ax = f.add_axes([.05, .05, .9, .9])
    ax.imshow(image)
    keypoint_loc_i, keypoint_loc_j = np.unravel_index(np.argmax(keypoint_loc_image), keypoint_loc_image.shape)
    ax.imshow(np.ones((227, 227, 3)), alpha=0.3)
    plt.axis('off')
    ax.set_aspect('auto')
    ax.scatter(keypoint_loc_j, keypoint_loc_i, color='k', marker='*', s=900)
    ax.scatter(keypoint_loc_j, keypoint_loc_i, color=(1, 165/255., 0), marker='*', s=300)

    # Show image
    f = plt.figure('image', facecolor='white', figsize=(4, 4))
    ax = f.add_axes([.05, .05, .9, .9])
    ax.imshow(image)
    plt.axis('off')
    ax.set_aspect('auto')

    for d, activation_dict in enumerate(activation_dicts):

        azimuth_activations = activation_dict['azimuth'][key]
        # Filter out azimuth activations for other object classes
        class_activations = azimuth_activations[360*obj_class:360*(obj_class+1)]
        # Turn the activations into predictions and get azimuth prediction
        azimuth_preds = softmax(class_activations)
        final_azimuth_pred = np.argmax(azimuth_preds)

        # Create heatmap for azimuth predictions
        azimuth_preds_vis = np.zeros((20, azimuth_preds.size, 3))
        for i in range(azimuth_preds.size):
            azimuth_preds_vis[:, i, :] = (azimuth_preds[i] - np.min(azimuth_preds))/(np.max(azimuth_preds) - np.min(azimuth_preds))/1.2
        # # Mark ground truth label
        # i_min = max(azimuth_label-3, 0)
        # i_max = min(azimuth_label+3, 360)
        # azimuth_preds_vis[:, i_min:i_max, :] = [1, 0, 1]
        # Mark predicted azimuth
        i_min = max(final_azimuth_pred-2, 0)
        i_max = min(final_azimuth_pred+2, 360)
        azimuth_preds_vis[:, i_min:i_max, :] = [1, 0, 1]

        # Plot viewpoint heatmap
        fig = plt.figure('confidence_%d' % d, figsize=(4, 4), facecolor='white')
        b = fig.add_axes([.075, .07, .85, .07])
        b.imshow(azimuth_preds_vis, aspect='auto')
        # Hide tick marks and y labels
        b.tick_params(bottom='off', top='off', left='off', right='off', labelleft='off', labelbottom='off')

        a = fig.add_axes([.075, .09*2+.05, .85, .97-.07*3-.05])
        a.bar(range(360), azimuth_preds, width=1, linewidth=0)
        a.axis([0, 360, 0, 0.07])
        a.tick_params(bottom='off', top='off', left='off', right='off', labelleft='off')
        # Show angle labels for confidence bar chart
        plt.setp(a, xticks=range(0, 361, 180))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('lmdbs_root', type=str, help='The root path of the correspondence LMDBs')
    parser.add_argument('num_examples', type=int, help='The number of examples to visualize')
    parser.add_argument('mode', type=str, help='Mode ("visualize" for heatmap visualization and "error" for error histogram").')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to store the plots. By default, no plots will be saved.')
    parser.add_argument('--reorder_by_perf', action='store_true', help='Whether to reorder the saved images. This is only applicable when output_dir is specified and two cache files are given.')
    parser.add_argument('activation_cache_files_labels', type=str, nargs='*', help='Pairs of cached activation files and labels')

    args = parser.parse_args()
    # Separate the cache files and labels
    activation_cache_files = args.activation_cache_files_labels[::2]
    labels = args.activation_cache_files_labels[1::2]
    assert(len(activation_cache_files) > 0)
    assert(len(labels) > 0)

    if args.mode == 'visualize':
        visualize_predictions(args.lmdbs_root, args.num_examples, activation_cache_files, args.output_dir, labels, args.reorder_by_perf)
    elif args.mode == 'error':
        plot_error_distributions(args.lmdbs_root, args.num_examples, activation_cache_files, args.output_dir, labels, zoom=True)
    elif args.mode == 'visualize_confidence':
        visualize_predictions_confidence(args.lmdbs_root, args.num_examples, activation_cache_files, args.output_dir, key, labels, args.reorder_by_perf)
