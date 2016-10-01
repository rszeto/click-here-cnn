import os
import sys
import random
import lmdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import syn_image_utils as utils
import global_variables as gv

if __name__ == '__main__':
    ### CREATE CSV FILES ###
    utils.createSynKeypointCsv()
    utils.createPascalKeypointCsv()

    ### CREATE LMDBs ###
    random.seed(123)
    # Regular synthetic LMDB
    if not os.path.exists(gv.g_corresp_syn_images_lmdb_folder):
        os.mkdir(gv.g_corresp_syn_images_lmdb_folder)
    utils.createCorrespLmdbs(os.path.join(gv.g_corresp_folder, 'syn_corresp_lmdb_info.csv'), gv.g_corresp_syn_images_lmdb_folder)

    print('Printing LMDB sizes for syn_lmdbs')
    for lmdb_name in utils.LMDB_NAMES:
        lmdb_obj = lmdb.open(os.path.join(gv.g_corresp_syn_images_lmdb_folder, lmdb_name))
        with lmdb_obj.begin() as txn:
            print('%d (%s)' % (txn.stat()['entries'], lmdb_name))
    print