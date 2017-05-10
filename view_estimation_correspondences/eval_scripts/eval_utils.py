import numpy as np
import lmdb
import os
import sys
import scipy

# Import custom LMDB utilities
eval_scripts_path = os.path.dirname(os.path.abspath(__file__))
view_est_corresp_path = os.path.dirname(eval_scripts_path)
sys.path.append(view_est_corresp_path)
import gen_lmdb_utils as utils
# Import global variables
render4cnn_path = os.path.dirname(view_est_corresp_path)
sys.path.append(render4cnn_path)
import global_variables as gv


def softmax(x):
    numers = np.exp(x)
    return numers/np.sum(numers)


# Angle conversions
def deg2rad(deg_angle):
    return deg_angle * np.pi / 180.0


def rad2deg(rad_angle):
    return rad_angle * 180.0 / np.pi

# Produce the rotation matrix from the axis rotations
def angle2dcm(xRot, yRot, zRot, deg_type='deg'):
    if deg_type == 'deg':
        xRot = deg2rad(xRot)
        yRot = deg2rad(yRot)
        zRot = deg2rad(zRot)

    xMat = np.array([
        [np.cos(xRot), np.sin(xRot), 0],
        [-np.sin(xRot), np.cos(xRot), 0],
        [0, 0, 1]
    ])

    yMat = np.array([
        [np.cos(yRot), 0, -np.sin(yRot)],
        [0, 1, 0],
        [np.sin(yRot), 0, np.cos(yRot)]
    ])

    zMat = np.array([
        [1, 0, 0],
        [0, np.cos(zRot), np.sin(zRot)],
        [0, -np.sin(zRot), np.cos(zRot)]
    ])

    return np.dot(zMat, np.dot(yMat, xMat))


def compute_angle_dists(preds, viewpoint_labels_as_mat):
    angle_dists = np.zeros(viewpoint_labels_as_mat.shape[0])
    for i in range(viewpoint_labels_as_mat.shape[0]):
        # Get rotation matrices from prediction and ground truth angles
        predR = angle2dcm(preds[i, 0], preds[i, 1], preds[i, 2])
        gtR = angle2dcm(viewpoint_labels_as_mat[i, 1], viewpoint_labels_as_mat[i, 2], viewpoint_labels_as_mat[i, 3])
        # Get geodesic distance
        angleDist = scipy.linalg.norm(scipy.linalg.logm(np.dot(predR.T, gtR)), 2) / np.sqrt(2)
        angle_dists[i] = angleDist

    return angle_dists


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