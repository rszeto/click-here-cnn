import os
import sys
import random
import lmdb
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import gen_lmdb_utils as utils
import global_variables as gv

if __name__ == '__main__':
    '''
    # Create synthetic image info file
    print('Creating syn LMDB CSV')
    start = time.time()
    print('Start time: ' + time.asctime(time.localtime(start)))
    utils.createSynKeypointCsv()
    end = time.time()
    print('End time: ' + time.asctime(time.localtime(end)))
    '''
    
    # Create synthetic image LMDB
    print('Creating syn LMDB')
    utils.createCorrespLmdbsMajor(os.path.join(gv.g_corresp_folder, 'syn_corresp_lmdb_info.csv'), gv.g_corresp_syn_images_lmdb_folder, gv.g_syn_lmdb_thread_count, True)

    '''
    # Create real image info file
    print('Creating real LMDB CSV')
    start = time.time()
    print('Start time: ' + time.asctime(time.localtime(start)))
    utils.createPascalKeypointCsv()
    end = time.time()
    print('End time: ' + time.asctime(time.localtime(end)))
    

    # Make PASCAL LMDB folder
    if not os.path.exists(gv.g_corresp_real_images_lmdb_folder):
        os.mkdir(gv.g_corresp_real_images_lmdb_folder)
    
    # PASCAL training LMDB
    if not os.path.exists(gv.g_corresp_real_images_train_lmdb_folder):
        os.mkdir(gv.g_corresp_real_images_train_lmdb_folder)
    utils.createCorrespLmdbsMajor(os.path.join(gv.g_corresp_folder, 'pascal_corresp_lmdb_info_train.csv'), gv.g_corresp_real_images_train_lmdb_folder, gv.g_real_lmdb_thread_count, True)

    # PASCAL test LMDB
    if not os.path.exists(gv.g_corresp_real_images_test_lmdb_folder):
        os.mkdir(gv.g_corresp_real_images_test_lmdb_folder)
    utils.createCorrespLmdbsMajor(os.path.join(gv.g_corresp_folder, 'pascal_corresp_lmdb_info_test.csv'), gv.g_corresp_real_images_test_lmdb_folder, gv.g_real_lmdb_thread_count, False)
    '''