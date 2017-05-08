import os
import sys
import time
from multiprocessing import Process
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import global_variables as gv
import gen_lmdb_utils as utils

def generate_lmdb_from_data(lmdb_data_root, lmdb_root, keys, is_pascal_test=False):
    # Print start time info
    start = time.time()
    print('Creating LMDBs from disk data')
    print('Data root: %s' % lmdb_data_root)
    print('LMDB root: %s' % lmdb_root)
    print('Start date: %s' % time.asctime(time.localtime(start)))

    # Define paths to data and final LMDBs
    image_data_root = os.path.join(lmdb_data_root, 'image')
    image_lmdb_root = os.path.join(lmdb_root, 'image_lmdb')

    gaussian_keypoint_map_data_root = os.path.join(lmdb_data_root, 'gaussian_keypoint_map')
    gaussian_keypoint_map_lmdb_root = os.path.join(lmdb_root, 'gaussian_keypoint_map_lmdb')

    keypoint_class_data_root = os.path.join(lmdb_data_root, 'keypoint_class')
    keypoint_class_lmdb_root = os.path.join(lmdb_root, 'keypoint_class_lmdb')

    viewpoint_label_data_root = os.path.join(lmdb_data_root, 'viewpoint_label')
    viewpoint_label_lmdb_root = os.path.join(lmdb_root, 'viewpoint_label_lmdb')

    euclidean_dt_map_data_root = os.path.join(lmdb_data_root, 'euclidean_dt_map')
    euclidean_dt_map_lmdb_root = os.path.join(lmdb_root, 'euclidean_dt_map_lmdb')

    manhattan_dt_map_data_root = os.path.join(lmdb_data_root, 'manhattan_dt_map')
    manhattan_dt_map_lmdb_root = os.path.join(lmdb_root, 'manhattan_dt_map_lmdb')

    chessboard_dt_map_data_root = os.path.join(lmdb_data_root, 'chessboard_dt_map')
    chessboard_dt_map_lmdb_root = os.path.join(lmdb_root, 'chessboard_dt_map_lmdb')

    zero_keypoint_map_data_root = os.path.join(lmdb_data_root, 'zero_keypoint_map')
    zero_keypoint_map_lmdb_root = os.path.join(lmdb_root, 'zero_keypoint_map_lmdb')

    zero_keypoint_class_data_root = os.path.join(lmdb_data_root, 'zero_keypoint_class')
    zero_keypoint_class_lmdb_root = os.path.join(lmdb_root, 'zero_keypoint_class_lmdb')

    gaussian_attn_map_data_root = os.path.join(lmdb_data_root, 'gaussian_attn_map')
    gaussian_attn_map_lmdb_root = os.path.join(lmdb_root, 'gaussian_attn_map_lmdb')

    perturbation_sigmas = range(5, 50, 5)
    keypoint_map_perturbed_data_roots = [os.path.join(lmdb_data_root, 'perturbed_%d_chessboard_dt_map' % perturb_sigma) for perturb_sigma in perturbation_sigmas]
    keypoint_map_perturbed_lmdb_roots = [os.path.join(lmdb_root, 'perturbed_%d_chessboard_dt_map_lmdb' % perturb_sigma) for perturb_sigma in perturbation_sigmas]

    # Create and run all the LMDB creation processes in parallel
    processes = []
    # Image
    processes.append(Process(target=utils.create_image_lmdb, args=(image_data_root, image_lmdb_root, keys)))
    # Keypoint maps
    processes.append(Process(target=utils.create_image_lmdb, args=(gaussian_keypoint_map_data_root, gaussian_keypoint_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(euclidean_dt_map_data_root, euclidean_dt_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(manhattan_dt_map_data_root, manhattan_dt_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(chessboard_dt_map_data_root, chessboard_dt_map_lmdb_root, keys)))
    # Keypoint class
    processes.append(Process(target=utils.create_vector_lmdb, args=(keypoint_class_data_root, keypoint_class_lmdb_root, keys)))
    # Viewpoint labels
    processes.append(Process(target=utils.create_vector_lmdb, args=(viewpoint_label_data_root, viewpoint_label_lmdb_root, keys)))
    # Zeroed-out keypoint map and class (only evaluated at PASCAL test time)
    if is_pascal_test:
        processes.append(Process(target=utils.create_tensor_lmdb, args=(zero_keypoint_map_data_root, zero_keypoint_map_lmdb_root, keys)))
        processes.append(Process(target=utils.create_vector_lmdb, args=(zero_keypoint_class_data_root, zero_keypoint_class_lmdb_root, keys)))
    # Gaussian skip-connection
    processes.append(Process(target=utils.create_tensor_lmdb, args=(gaussian_attn_map_data_root, gaussian_attn_map_lmdb_root, keys)))
    # Perturbed keypoint maps (only evaluated at PASCAL test time)
    if is_pascal_test:
        for i, sigma in enumerate(perturbation_sigmas):
            processes.append(Process(target=utils.create_image_lmdb, args=(keypoint_map_perturbed_data_roots[i], keypoint_map_perturbed_lmdb_roots[i], keys)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print('Finished creating LMDBs at root %s' % lmdb_root)
    utils.print_elapsed_time(start)
    print


def generate_lmdb(data_root_path, lmdb_root_path, is_pascal_test=False):
    keys_path = os.path.join(data_root_path, 'keys.txt')
    with open(keys_path, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    generate_lmdb_from_data(data_root_path, lmdb_root_path, keys, is_pascal_test=is_pascal_test)


if __name__ == '__main__':

    # Synthetic training data
    generate_lmdb(gv.g_corresp_syn_train_lmdb_data_folder, gv.g_corresp_syn_train_lmdb_folder)
    # Synthetic validation data
    generate_lmdb(gv.g_corresp_syn_test_lmdb_data_folder, gv.g_corresp_syn_test_lmdb_folder)
    # PASCAL training data
    generate_lmdb(gv.g_corresp_pascal_train_lmdb_data_folder, gv.g_corresp_pascal_train_lmdb_folder)
    # PASCAL test data
    generate_lmdb(gv.g_corresp_pascal_test_lmdb_data_folder, gv.g_corresp_pascal_test_lmdb_folder, is_pascal_test=True)

