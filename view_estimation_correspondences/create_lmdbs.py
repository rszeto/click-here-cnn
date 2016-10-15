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
    # ### CREATE CSV FILES ###
    # print('Creating syn LMDB CSV')
    # start = time.time()
    # print('Start time: ' + time.asctime(time.localtime(start)))
    # utils.createSynKeypointCsv()
    # end = time.time()
    # print('End time: ' + time.asctime(time.localtime(end)))
    # # print('Creating real LMDB CSV')
    # # utils.createPascalKeypointCsv()

    ### CREATE LMDBs ###
    # Regular synthetic LMDB
    if not os.path.exists(gv.g_corresp_syn_images_lmdb_folder):
        os.mkdir(gv.g_corresp_syn_images_lmdb_folder)
    print('Creating syn LMDB')
    random.seed(123)
    utils.createCorrespLmdbs(os.path.join(gv.g_corresp_folder, 'syn_corresp_lmdb_info_small.csv'), gv.g_corresp_syn_images_lmdb_folder)

    print('Printing LMDB sizes for syn_lmdbs')
    for lmdb_name in utils.LMDB_NAMES:
        lmdb_obj = lmdb.open(os.path.join(gv.g_corresp_syn_images_lmdb_folder, lmdb_name))
        with lmdb_obj.begin() as txn:
            print('%d (%s)' % (txn.stat()['entries'], lmdb_name))
    print

    '''

    ### CREATE PASCAL LMDBs ###
    # Make PASCAL LMDB folder
    if not os.path.exists(gv.g_corresp_real_images_lmdb_folder):
        os.mkdir(gv.g_corresp_real_images_lmdb_folder)
    
    # PASCAL training LMDB
    if not os.path.exists(gv.g_corresp_real_images_train_lmdb_folder):
        os.mkdir(gv.g_corresp_real_images_train_lmdb_folder)
    utils.createCorrespLmdbs(os.path.join(gv.g_corresp_folder, 'pascal_corresp_lmdb_info_train.csv'), gv.g_corresp_real_images_train_lmdb_folder)

    print('Printing LMDB sizes for real_lmdbs/train')
    for lmdb_name in utils.LMDB_NAMES:
        lmdb_obj = lmdb.open(os.path.join(gv.g_corresp_real_images_train_lmdb_folder, lmdb_name))
        with lmdb_obj.begin() as txn:
            print('%d (%s)' % (txn.stat()['entries'], lmdb_name))
    print

    # PASCAL test LMDB
    if not os.path.exists(gv.g_corresp_real_images_test_lmdb_folder):
        os.mkdir(gv.g_corresp_real_images_test_lmdb_folder)
    utils.createCorrespLmdbs(os.path.join(gv.g_corresp_folder, 'pascal_corresp_lmdb_info_test.csv'), gv.g_corresp_real_images_test_lmdb_folder)

    print('Printing LMDB sizes for real_lmdbs/test')
    for lmdb_name in utils.LMDB_NAMES:
        lmdb_obj = lmdb.open(os.path.join(gv.g_corresp_real_images_test_lmdb_folder, lmdb_name))
        with lmdb_obj.begin() as txn:
            print('%d (%s)' % (txn.stat()['entries'], lmdb_name))
    print

    '''
