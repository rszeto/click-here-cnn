import numpy as np
import os
import sys
import re
from scipy.ndimage import imread
import lmdb
import random
from scipy.misc import imresize
import time
import multiprocessing
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imsave
from functools import partial
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
import gen_lmdb_utils as utils

# Import Caffe
sys.path.append(g_pycaffe_path)
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

# How wide the Gaussian kernel should be
SIGMA = 23
# The number of workers
NUM_WORKERS = 30

ACTIVATION_WEIGHT_MAP_CACHE = {}

def generate_lmdb_data(info_file_path, lmdb_data_root, reverse, num_workers=NUM_WORKERS):
    # Seed RNG to replicate LMDB generation for debugging
    random.seed(123)
    
    # Print initial info
    start = time.time()
    print('Generating LMDB data from CSV')
    print('Info file path: %s' % info_file_path)
    print('LMDB data root: %s' % lmdb_data_root)
    print('Start date: %s\n' % time.asctime(time.localtime(start)))

    lmdb_data_paths = dict(
        data=os.path.join(lmdb_data_root, 'image'),
        binary_kp_map=os.path.join(lmdb_data_root, 'keypoint_loc'),
        gaussian_kp_map=os.path.join(lmdb_data_root, 'gaussian_keypoint_map'),
        keypoint_class_vector=os.path.join(lmdb_data_root, 'keypoint_class'),
        viewpoint_label=os.path.join(lmdb_data_root, 'viewpoint_label'),
        euclidean_dt_map=os.path.join(lmdb_data_root, 'euclidean_dt_map'),
        manhattan_dt_map=os.path.join(lmdb_data_root, 'manhattan_dt_map'),
        chessboard_dt_map=os.path.join(lmdb_data_root, 'chessboard_dt_map'),
        zero_keypoint_map=os.path.join(lmdb_data_root, 'zero_keypoint_map'),
        zero_keypoint_class=os.path.join(lmdb_data_root, 'zero_keypoint_class'),
        gaussian_attn_map=os.path.join(lmdb_data_root, 'gaussian_attn_map'),
        perturbed_5_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_5_chessboard_dt_map'),
        perturbed_10_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_10_chessboard_dt_map'),
        perturbed_15_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_15_chessboard_dt_map'),
        perturbed_20_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_20_chessboard_dt_map'),
        perturbed_25_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_25_chessboard_dt_map'),
        perturbed_30_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_30_chessboard_dt_map'),
        perturbed_35_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_35_chessboard_dt_map'),
        perturbed_40_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_40_chessboard_dt_map'),
        perturbed_45_chessboard_dt_map=os.path.join(lmdb_data_root, 'perturbed_45_chessboard_dt_map')
    )

    # Create folders under the data root
    for path in lmdb_data_paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    # Read the data info from the CSV file
    print('Reading data info file')
    with open(info_file_path) as info_file:
        lines = info_file.readlines()
    # Remove header row
    lines = lines[1:]

    # Generate all jobs
    print('Generating jobs from the info file')
    all_jobs = []
    for i, line in enumerate(lines):
        if reverse:
            all_jobs.append((utils.random_number_string(), line, False))
            all_jobs.append((utils.random_number_string(), line, True))
        else:
            all_jobs.append((utils.random_number_string(), line, False))

    # Start multithreading pool
    pool = multiprocessing.Pool(num_workers)
    # Save data to disk
    save_data_for_job_p = partial(save_data_for_job, lmdb_data_paths=lmdb_data_paths)
    pool_results = pool.imap(save_data_for_job_p, all_jobs)
    # Print progress
    for i, _ in enumerate(pool_results):
        if i % 1000 == 0:
            print('Saved data for job %d/%d' % (i, len(all_jobs)))
            utils.print_elapsed_time(start)

    # Write the keys to file
    print('Writing keys to file')
    all_job_keys = pool.map(get_job_key, all_jobs)
    key_file_path = os.path.join(lmdb_data_root, 'keys.txt')
    with open(key_file_path, 'w') as f:
        for key in sorted(all_job_keys):
            f.write(key + '\n')

    end = time.time()
    print('\nEnd date: %s' % time.asctime(time.localtime(end)))
    utils.print_elapsed_time(start)

def save_data_for_job(job, lmdb_data_paths):
    job_key = get_job_key(job)

    image = job_to_image(job)
    path = os.path.join(lmdb_data_paths['data'], job_key + '.png')
    imsave(path, image)

    binary_map = job_to_binary_keypoint_map(job)
    path = os.path.join(lmdb_data_paths['binary_kp_map'], job_key + '.png')
    imsave(path, binary_map)

    gaussian_map = job_to_gaussian_keypoint_map(job)
    path = os.path.join(lmdb_data_paths['gaussian_kp_map'], job_key + '.png')
    imsave(path, gaussian_map)

    keypoint_class_vector = job_to_keypoint_class_vector(job)
    path = os.path.join(lmdb_data_paths['keypoint_class_vector'], job_key + '.npy')
    np.save(path, keypoint_class_vector)

    viewpoint_label = job_to_viewpoint_label(job)
    path = os.path.join(lmdb_data_paths['viewpoint_label'], job_key + '.npy')
    np.save(path, viewpoint_label)

    euclidean_dt_map = job_to_euclidean_dt_map(job)
    path = os.path.join(lmdb_data_paths['euclidean_dt_map'], job_key + '.npy')
    np.save(path, euclidean_dt_map)
    
    manhattan_dt_map = job_to_manhattan_dt_map(job)
    path = os.path.join(lmdb_data_paths['manhattan_dt_map'], job_key + '.npy')
    np.save(path, manhattan_dt_map)

    chessboard_dt_map = job_to_chessboard_dt_map(job)
    path = os.path.join(lmdb_data_paths['chessboard_dt_map'], job_key + '.npy')
    np.save(path, chessboard_dt_map)

    zero_keypoint_map = job_to_zero_keypoint_map(job)
    path = os.path.join(lmdb_data_paths['zero_keypoint_map'], job_key + '.npy')
    np.save(path, zero_keypoint_map)

    zero_keypoint_class_vector = job_to_zero_keypoint_class_vector(job)
    path = os.path.join(lmdb_data_paths['zero_keypoint_class'], job_key + '.npy')
    np.save(path, zero_keypoint_class_vector)

    gaussian_attn_map = job_to_gaussian_attn_map(job)
    path = os.path.join(lmdb_data_paths['gaussian_attn_map'], job_key + '.npy')
    np.save(path, gaussian_attn_map)

    for perturb_sigma in range(5, 50, 5):
        perturbed_chessboard_dt_map = job_to_perturbed_chessboard_dt_map(job, perturb_sigma)
        path_key = 'perturbed_%d_chessboard_dt_map' % perturb_sigma
        path = os.path.join(lmdb_data_paths[path_key], job_key + '.png')
        imsave(path, perturbed_chessboard_dt_map)


def get_job_key(job):
    key_prefix, line, reverse = job

    # Extract image path from the line
    m = re.match(utils.LINE_FORMAT, line)
    full_image_path = m.group(1)
    keypoint_class = m.group(8)
    obj_class_id = m.group(9)
    # obj_id = m.group(13)
    obj_id = '0'
    # Get the file name without the extension
    full_image_name, _ = os.path.splitext(os.path.basename(full_image_path))
    if reverse:
        return key_prefix + '_' + full_image_name + '_obj' + obj_id + '_objc' + obj_class_id + '_kp' + keypoint_class + '_r'
    else:
        return key_prefix + '_' + full_image_name + '_obj' + obj_id + '_objc' + obj_class_id + '_kp' + keypoint_class

'''
@args
    job ((str, str, bool)): A job tuple consisting of the key prefix, the line describing the instance, and
        whether this example should be flipped
'''
def job_to_image(job):
    key_prefix, line, reverse = job

    # Extract info from the line
    m = re.match(utils.LINE_FORMAT, line)
    full_image_path = m.group(1)
    bbox = np.array([int(x) for x in m.group(2,3,4,5)])

    # Get the cropped image and scale it
    full_image = imread(full_image_path)
    # Convert grayscale images to "color"
    if full_image.ndim == 2:
        full_image = np.dstack((full_image, full_image, full_image))
    cropped_image = full_image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
    scaled_image = imresize(cropped_image, (g_images_resize_dim, g_images_resize_dim))
    # Flip the image if needed
    if reverse:
        scaled_image = np.fliplr(scaled_image)

    return scaled_image

'''
@args
    job ((str, str, bool)): A job tuple consisting of the key prefix, the line describing the instance, and
        whether this example should be flipped
'''
def job_to_binary_keypoint_map(job):
    key_prefix, line, reverse = job

    # Extract info from the line
    m = re.match(utils.LINE_FORMAT, line)
    bbox = np.array([int(x) for x in m.group(2,3,4,5)])
    keypoint_loc_full = np.array([float(x) for x in m.group(6,7)])

    if keypoint_loc_full[0] == -1 and keypoint_loc_full[1] == -1:
        return np.zeros((g_images_resize_dim, g_images_resize_dim), dtype=np.uint8)

    # Get bounding box dimensions
    bbox_size = np.array([bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1])
    # Get keypoint location inside the bounding box
    keypoint_loc_bb = keypoint_loc_full - bbox[:2]
    keypoint_loc_scaled = np.floor(g_images_resize_dim * keypoint_loc_bb / bbox_size).astype(np.uint8)
    # Push keypoint inside image (sometimes it ends up on edge due to float arithmetic)
    keypoint_loc_scaled[0] = min(keypoint_loc_scaled[0], g_images_resize_dim-1)
    keypoint_loc_scaled[1] = min(keypoint_loc_scaled[1], g_images_resize_dim-1)
    # Create keypoint location image
    keypoint_image = np.zeros((g_images_resize_dim, g_images_resize_dim), dtype=np.uint8)
    keypoint_image[keypoint_loc_scaled[1], keypoint_loc_scaled[0]] = 1
    # Flip the image if needed
    if reverse:
        keypoint_image = np.fliplr(keypoint_image)

    return keypoint_image * 255

'''
@args
    job ((str, str, bool)): A job tuple consisting of the key prefix, the line describing the instance, and
        whether this example should be flipped
'''
def job_to_gaussian_keypoint_map(job):
    # Scale binary map values
    keypoint_image = job_to_binary_keypoint_map(job)
    keypoint_image = keypoint_image / np.max(keypoint_image) * 1e6
    # Generate Gaussian map by convolving the binary map
    gaussian_keypoint_image = gaussian_filter(keypoint_image, SIGMA, mode='constant')
    # Normalize so the maximum value is 1
    gaussian_keypoint_image = gaussian_keypoint_image / np.max(gaussian_keypoint_image)

    return gaussian_keypoint_image

'''
@args
    job ((str, str, bool)): A job tuple consisting of the key prefix, the line describing the instance, and
        whether this example should be flipped
'''
def job_to_keypoint_class_vector(job):
    key_prefix, line, reverse = job

    # Extract info from the line
    m = re.match(utils.LINE_FORMAT, line)
    keypoint_class = int(m.group(8))

    # Get mirror keypoint class if needed
    if reverse:
        keypoint_name = utils.KEYPOINT_CLASSES[keypoint_class]
        keypoint_name_r = keypoint_name
        if 'left' in keypoint_name:
            keypoint_name_r = keypoint_name.replace('left', 'right')
        elif 'right' in keypoint_name:
            keypoint_name_r = keypoint_name.replace('right', 'left')
        keypoint_class = utils.KEYPOINTCLASS_INDEX_MAP[keypoint_name_r]

    # Get one-hot vector encoding of keypoint class
    keypoint_class_vec = np.zeros(len(utils.KEYPOINT_CLASSES), dtype=np.uint8)
    if keypoint_class > 0:
        keypoint_class_vec[keypoint_class] = 1

    return keypoint_class_vec

'''
@args
    job ((str, str, bool)): A job tuple consisting of the key prefix, the line describing the instance, and
        whether this example should be flipped
'''
def job_to_zero_keypoint_class_vector(job):
    return np.zeros(len(utils.KEYPOINT_CLASSES), dtype=np.uint8)

'''
@args
    job ((str, str, bool)): A job tuple consisting of the key prefix, the line describing the instance, and
        whether this example should be flipped
'''
def job_to_viewpoint_label(job):
    key_prefix, line, reverse = job

    # Extract info from the line
    m = re.match(utils.LINE_FORMAT, line)
    viewpoint_label = np.array([int(x) for x in m.group(9,10,11,12)])

    # Save label for regular image
    viewpoint_label_vec = viewpoint_label
    # Get viewpoint label of flipped image if needed
    if reverse:
        # Extract normal azimuth and tilt
        object_class = viewpoint_label_vec[0]
        azimuth = viewpoint_label_vec[1]
        tilt = viewpoint_label_vec[3]
        # Get reversed azimuth and tilt
        azimuth_r = np.mod(360-azimuth, 360)
        tilt_r = np.mod(-1*tilt, 360)
        # Update viewpoint label
        viewpoint_label_vec[1] = utils.view2label(azimuth_r, object_class)
        viewpoint_label_vec[3] = utils.view2label(tilt_r, object_class)

    return viewpoint_label_vec

def job_to_zero_keypoint_map(job):
    return np.zeros((g_images_resize_dim, g_images_resize_dim), dtype=np.uint8)

def job_to_gaussian_attn_map(job):
    ret = np.zeros((13, 13))
    ret[6, 6] = 1
    ret = gaussian_filter(ret, 3)
    ret /= np.sum(ret)
    return ret

def job_to_euclidean_dt_map(job):
    keypoint_image = job_to_binary_keypoint_map(job)
    # Invert keypoint image
    keypoint_image_inv = 1 - keypoint_image/np.max(keypoint_image)
    # Apply distance transform
    return scipy.ndimage.distance_transform_edt(keypoint_image_inv)

def job_to_manhattan_dt_map(job):
    keypoint_image = job_to_binary_keypoint_map(job)
    # Invert keypoint image
    keypoint_image_inv = 1 - keypoint_image/np.max(keypoint_image)
    # Apply distance transform
    return scipy.ndimage.distance_transform_cdt(keypoint_image_inv, metric='taxicab')

def job_to_chessboard_dt_map(job):
    key_prefix, line, reverse = job

    # Extract info from the line
    m = re.match(utils.LINE_FORMAT, line)
    keypoint_loc_full = np.array([float(x) for x in m.group(6,7)])

    if keypoint_loc_full[0] == -1 and keypoint_loc_full[1] == -1:
        return np.zeros((g_images_resize_dim, g_images_resize_dim), dtype=np.uint8)

    keypoint_image = job_to_binary_keypoint_map(job)
    # Invert keypoint image
    keypoint_image_inv = 1 - keypoint_image/np.max(keypoint_image)
    # Apply distance transform
    return scipy.ndimage.distance_transform_cdt(keypoint_image_inv, metric='chessboard')

def job_to_perturbed_chessboard_dt_map(job, sigma):
    keypoint_image = job_to_binary_keypoint_map(job)
    # Find keypoint coordinate
    keypoint_loc = np.unravel_index(np.argmax(keypoint_image), keypoint_image.shape)
    # Sample a perturbed coordinate
    perturbed_keypoint_loc = np.array([-1, -1])
    while not utils.insideBox(perturbed_keypoint_loc, [0, 0, g_images_resize_dim-1, g_images_resize_dim-1]):
        perturbed_keypoint_loc = np.random.multivariate_normal(keypoint_loc, sigma ** 2 * np.eye(2)).astype(np.int)
    # Create input to distance transform
    perturbed_keypoint_image = np.full((g_images_resize_dim, g_images_resize_dim), 1)
    perturbed_keypoint_image[perturbed_keypoint_loc[0], perturbed_keypoint_loc[1]] = 0
    # Apply distance transform
    return scipy.ndimage.distance_transform_cdt(perturbed_keypoint_image, metric='chessboard')

def save_vector_data(args, vector_data_root):
    vector, key = args
    path = os.path.join(vector_data_root, key + '.npy')
    np.save(path, vector)

def path_to_key(path):
    return os.path.splitext(os.path.basename(path))[0]

def weight_averaged_activations(args):
    activations, weights = args
    return np.sum(np.sum(activations * weights, axis=2), axis=1)

def path_to_transformed_image(path, transformer, data_layer_name):
    image = imread(path)
    image_t = transformer.preprocess(data_layer_name, image)
    return image_t

def join_paths(base, dir):
    return os.path.join(dir, base)


if __name__ == '__main__':
    # Create CSVs
    utils.create_syn_image_keypoint_csvs()
    utils.create_pascal_image_keypoint_csvs()

    # Generate synthetic train data with data augmentation (horizontal flip)
    generate_lmdb_data(g_syn_train_image_keypoint_info_file, g_corresp_syn_train_lmdb_data_folder, True)
    # Generate synthetic test data with no data augmentation
    generate_lmdb_data(g_syn_test_image_keypoint_info_file, g_corresp_syn_test_lmdb_data_folder, False)
    # Create PASCAL training data with data augmentation (horizontal flip)
    generate_lmdb_data(g_pascal_train_image_keypoint_info_file, g_corresp_pascal_train_lmdb_data_folder, True)
    # Create PASCAL test data without data augmentation
    generate_lmdb_data(g_pascal_test_image_keypoint_info_file, g_corresp_pascal_test_lmdb_data_folder, False)
