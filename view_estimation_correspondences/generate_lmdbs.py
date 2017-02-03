import os
import sys
import time
from multiprocessing import Process

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import global_variables as gv
import gen_lmdb_utils as utils

def generate_lmdb_from_data(lmdb_data_root, lmdb_root, keys):
    # Print start time info
    start = time.time()
    print('Creating LMDBs from disk data')
    print('Data root: %s' % lmdb_data_root)
    print('LMDB root: %s' % lmdb_root)
    print('Start date: %s' % time.asctime(time.localtime(start)))

    # Define paths to data and final LMDBs
    image_data_root = os.path.join(lmdb_data_root, 'image')
    image_lmdb_root = os.path.join(lmdb_root, 'image_lmdb')

    binary_keypoint_map_data_root = os.path.join(lmdb_data_root, 'keypoint_loc')
    binary_keypoint_map_lmdb_root = os.path.join(lmdb_root, 'keypoint_loc_lmdb')

    gaussian_keypoint_map_data_root = os.path.join(lmdb_data_root, 'gaussian_keypoint_map')
    gaussian_keypoint_map_lmdb_root = os.path.join(lmdb_root, 'gaussian_keypoint_map_lmdb')

    keypoint_class_data_root = os.path.join(lmdb_data_root, 'keypoint_class')
    keypoint_class_lmdb_root = os.path.join(lmdb_root, 'keypoint_class_lmdb')

    viewpoint_label_data_root = os.path.join(lmdb_data_root, 'viewpoint_label')
    viewpoint_label_lmdb_root = os.path.join(lmdb_root, 'viewpoint_label_lmdb')

    pool5_weight_map_data_root = os.path.join(lmdb_data_root, 'weight_maps', 'pool5')
    pool5_weight_map_lmdb_root = os.path.join(lmdb_root, 'pool5_weight_maps_lmdb')

    euclidean_dt_map_data_root = os.path.join(lmdb_data_root, 'euclidean_dt_map')
    euclidean_dt_map_lmdb_root = os.path.join(lmdb_root, 'euclidean_dt_map_lmdb')

    manhattan_dt_map_data_root = os.path.join(lmdb_data_root, 'manhattan_dt_map')
    manhattan_dt_map_lmdb_root = os.path.join(lmdb_root, 'manhattan_dt_map_lmdb')

    chessboard_dt_map_data_root = os.path.join(lmdb_data_root, 'chessboard_dt_map')
    chessboard_dt_map_lmdb_root = os.path.join(lmdb_root, 'chessboard_dt_map_lmdb')

    zero_keypoint_map_data_root = os.path.join(lmdb_data_root, 'zero_keypoint_map')
    zero_keypoint_map_lmdb_root = os.path.join(lmdb_root, 'zero_keypoint_map_lmdb')

    # Create and run all the LMDB creation processes in parallel
    processes = []
    processes.append(Process(target=utils.create_image_lmdb, args=(image_data_root, image_lmdb_root, keys)))
    processes.append(Process(target=utils.create_image_lmdb, args=(binary_keypoint_map_data_root, binary_keypoint_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_image_lmdb, args=(gaussian_keypoint_map_data_root, gaussian_keypoint_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_vector_lmdb, args=(keypoint_class_data_root, keypoint_class_lmdb_root, keys)))
    processes.append(Process(target=utils.create_vector_lmdb, args=(viewpoint_label_data_root, viewpoint_label_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(pool5_weight_map_data_root, pool5_weight_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(euclidean_dt_map_data_root, euclidean_dt_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(manhattan_dt_map_data_root, manhattan_dt_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(chessboard_dt_map_data_root, chessboard_dt_map_lmdb_root, keys)))
    processes.append(Process(target=utils.create_tensor_lmdb, args=(zero_keypoint_map_data_root, zero_keypoint_map_lmdb_root, keys)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print('Finished creating LMDBs at root %s' % lmdb_root)
    utils.print_elapsed_time(start)
    print

if __name__ == '__main__':
    for mode in sys.argv[1:]:
        if mode == 'syn/train':
            # Generate syn LMDB
            with open(os.path.join(gv.g_corresp_syn_lmdb_data_folder, 'keys_train.txt'), 'r') as f:
                keys = [line.strip() for line in f.readlines()]
            generate_lmdb_from_data(gv.g_corresp_syn_lmdb_data_folder, gv.g_z_corresp_syn_train_lmdb_folder, keys)
        elif mode == 'syn/val':
            with open(os.path.join(gv.g_corresp_syn_lmdb_data_folder, 'keys_val.txt'), 'r') as f:
                keys = [line.strip() for line in f.readlines()]
            generate_lmdb_from_data(gv.g_corresp_syn_lmdb_data_folder, gv.g_z_corresp_syn_val_lmdb_folder, keys)
        elif mode == 'real/train':
            # Generate PASCAL full train LMDB
            with open(os.path.join(gv.g_corresp_real_train_lmdb_data_folder, 'keys.txt'), 'r') as f:
                keys = [line.strip() for line in f.readlines()]
            generate_lmdb_from_data(gv.g_corresp_real_train_lmdb_data_folder, gv.g_z_corresp_real_train_lmdb_folder, keys)
        elif mode == 'real/train_train':
            # Generate PASCAL train split LMDB
            with open(os.path.join(gv.g_corresp_real_train_train_lmdb_data_folder, 'keys.txt'), 'r') as f:
                keys = [line.strip() for line in f.readlines()]
            generate_lmdb_from_data(gv.g_corresp_real_train_train_lmdb_data_folder, gv.g_z_corresp_real_train_train_lmdb_folder, keys)
        elif mode == 'real/train_val':
            # Generate PASCAL validation split LMDB
            with open(os.path.join(gv.g_corresp_real_train_val_lmdb_data_folder, 'keys.txt'), 'r') as f:
                keys = [line.strip() for line in f.readlines()]
            generate_lmdb_from_data(gv.g_corresp_real_train_val_lmdb_data_folder, gv.g_z_corresp_real_train_val_lmdb_folder, keys)
        elif mode == 'real/test':
            # Generate PASCAL test LMDB
            with open(os.path.join(gv.g_corresp_real_test_lmdb_data_folder, 'keys.txt'), 'r') as f:
                keys = [line.strip() for line in f.readlines()]
            generate_lmdb_from_data(gv.g_corresp_real_test_lmdb_data_folder, gv.g_z_corresp_real_test_lmdb_folder, keys)
