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

from eval_utils import getCorrespLmdbData
from eval_utils import compute_angle_dists
from eval_utils import deg2rad

def plot_error_distributions(score_cache_file, model_name, lmdbs_root=gv.g_corresp_pascal_test_lmdb_folder, output_dir=gv.g_error_dist_vis_folder, zoom=False):
    lmdb_keys, images_dict, keypoint_loc_dict, keypoint_class_dict, viewpoint_label_dict = getCorrespLmdbData(lmdbs_root, 1e8)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert labels to numpy array for comparing against activations (which are returned as a matrix)
    viewpoint_labels_as_mat = np.array([viewpoint_label_dict[key] for key in lmdb_keys])

    print('Importing scores from cache file')
    score_dict = pickle.load(open(score_cache_file, 'rb'))
    scores = np.zeros((len(lmdb_keys), len(gv.g_angle_names), 360))
    for i, angle_name in enumerate(gv.g_angle_names):
        for j, key in enumerate(lmdb_keys):
            scores[j, i, :] = score_dict[angle_name][key]

    # Get predictions by taking the highest-scoring angle per rotation axis
    preds = np.argmax(scores, axis=2)
    # Compare predictions to ground truth labels
    angle_dists = compute_angle_dists(preds, viewpoint_labels_as_mat)

    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    plt.hist(angle_dists, bins=35)
    plt.title(model_name)
    plt.xlabel('Error (rad)')
    plt.tight_layout()

    if zoom:
        plt.axis([1.5, 2.25, 0, 400])
    else:
        plt.axis([0, 2.25, 0, 3000])
        p = patches.Rectangle((1.5, 0), 2.25, 400, fill=False, linestyle='dashed')
        ax.add_patch(p)

    # Save plot
    filename = model_name + '-z.eps' if zoom else model_name + '.eps'
    fig.savefig(os.path.join(output_dir, filename), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache_file', type=str, help='Path to the cached predictions')
    parser.add_argument('model_name', type=str, help='The name of the model (appears as the plot title)')
    parser.add_argument('--zoom', action='store_true', help='Whether to zoom in on the tail')
    parser.add_argument('--lmdbs_root', type=str, help='Root of the keypoint LMDBs')

    args = parser.parse_args()
    if args.lmdbs_root:
        plot_error_distributions(args.cache_file, args.model_name, lmdbs_root=args.lmdbs_root, zoom=args.zoom)
    else:
        plot_error_distributions(args.cache_file, args.model_name, zoom=args.zoom)