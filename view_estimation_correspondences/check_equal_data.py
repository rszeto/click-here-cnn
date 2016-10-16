import sys
import os
import lmdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import gen_lmdb_utils as utils
import global_variables as gv

def checkLmdbsEqual(lmdb_root_a, lmdb_root_b):
    lmdb_obj_a = lmdb.open(lmdb_root_a, readonly=True)
    lmdb_obj_b = lmdb.open(lmdb_root_b, readonly=True)

    with lmdb_obj_a.begin() as txn:
        N_a = txn.stat()['entries']
    with lmdb_obj_b.begin() as txn:
        N_b = txn.stat()['entries']
    assert(N_a == N_b)

    with lmdb_obj_a.begin() as txn_a:
        with lmdb_obj_b.begin() as txn_b:
            cursor_a = txn_a.cursor()
            cursor_b = txn_b.cursor()
            for i in range(N_a):
                cursor_a.next()
                cursor_b.next()
                assert(cursor_a.key() == cursor_b.key())
                assert(cursor_a.value() == cursor_b.value())

    print('LMDBs are equal')

if __name__ == '__main__':
    corresp_lmdb_root_a = '/home/szetor/Documents/DENSO_VAC/RenderForCNN/data/correspondences/syn_lmdbs'
    corresp_lmdb_root_b = '/home/szetor/Documents/DENSO_VAC/RenderForCNN/data/correspondences/syn_lmdbs_old'
    lmdb_names = ['image_lmdb', 'keypoint_class_lmdb', 'keypoint_loc_lmdb', 'viewpoint_label_lmdb']

    for lmdb_name in lmdb_names:
        lmdb_root_a = os.path.join(corresp_lmdb_root_a, lmdb_name)
        lmdb_root_b = os.path.join(corresp_lmdb_root_b, lmdb_name)
        checkLmdbsEqual(lmdb_root_a, lmdb_root_b)
