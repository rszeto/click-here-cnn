#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import socket

g_render4cnn_root_folder = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
g_blender_executable_path = '/z/home/szetor/sw/blender-2.71/blender' #!! MODIFY if necessary
g_matlab_executable_path = 'matlab' # !! MODIFY if necessary
g_pycaffe_path = '/z/home/szetor/sw/caffe-r4cnn/python'
g_data_folder = os.path.abspath(os.path.join(g_render4cnn_root_folder, 'data'))
g_datasets_folder = os.path.abspath(os.path.join(g_render4cnn_root_folder, 'datasets'))
g_shapenet_root_folder = os.path.join(g_datasets_folder, 'shapenet-correspondences')
g_pascal3d_root_folder = os.path.join(g_datasets_folder, 'pascal3d')
g_sun2012pascalformat_root_folder = os.path.join(g_datasets_folder, 'sun2012pascalformat')

# ------------------------------------------------------------
# RENDER FOR CNN PIPELINE
# ------------------------------------------------------------
g_shape_synset_name_pairs = [('02691156', 'aeroplane'),
                             ('02834778', 'bicycle'),
                             ('02858304', 'boat'),
                             ('02876657', 'bottle'),
                             ('02924116', 'bus'),
                             ('02958343', 'car'),
                             ('03001627', 'chair'),
                             ('04379243', 'diningtable'),
                             ('03790512', 'motorbike'),
                             ('04256520', 'sofa'),
                             ('04468005', 'train'),
                             ('03211117', 'tvmonitor')]
g_shape_synsets = [x[0] for x in g_shape_synset_name_pairs]
g_shape_names = [x[1] for x in g_shape_synset_name_pairs]
g_syn_images_folder = os.path.join(g_data_folder, 'syn_images')
g_syn_images_cropped_folder = os.path.join(g_data_folder, 'syn_images_cropped')
g_syn_images_bkg_overlaid_folder = os.path.join(g_data_folder, 'syn_images_cropped_bkg_overlaid')
g_syn_bkg_filelist = os.path.join(g_sun2012pascalformat_root_folder, 'filelist.txt')
g_syn_bkg_folder = os.path.join(g_sun2012pascalformat_root_folder, 'JPEGImages')
g_syn_cluttered_bkg_ratio = 0.8
g_blank_blend_file_path = os.path.join(g_render4cnn_root_folder, 'render_pipeline/blank.blend') 
# g_syn_images_num_per_category = 200000
g_syn_images_num_per_category = 100
g_syn_rendering_thread_num = 4

# Rendering is computational demanding. you may want to consider using multiple servers.
#g_hostname_synset_idx_map = {'<server1-hostname>': [0,1],
#                             '<server2-hostname>': [2,3,4],
#                             '<server3-hostname>': [5,6,7],
#                             '<server4-hostname>':[8,9], 
#                             '<server5-hostname>':[10,11]}
# Only process buses, cars, and motorbikes for CH-CNN
g_hostname_synset_idx_map = {socket.gethostname(): [4,5,8]}

# Crop and overlay is IO-heavy, running on local FS is much faster
# Only process buses, cars, and motorbikes for CH-CNN
g_crop_hostname_synset_idx_map = {socket.gethostname(): [4,5,8]}
g_overlay_hostname_synset_idx_map = {socket.gethostname(): [4,5,8]}

# view and truncation distribution estimation
g_matlab_kde_folder = os.path.join(g_render4cnn_root_folder, 'render_pipeline/kde/matlab_kde_package') 
g_view_statistics_folder = os.path.join(g_data_folder, 'view_statistics')
g_view_distribution_folder = os.path.join(g_data_folder, 'view_distribution')
g_view_distribution_files = dict(zip(g_shape_synsets, [os.path.join(g_view_distribution_folder, name+'.txt') for name in g_shape_names]))
g_truncation_statistics_folder = os.path.join(g_data_folder, 'truncation_statistics')
g_truncation_distribution_folder = os.path.join(g_data_folder, 'truncation_distribution')
g_truncation_distribution_files = dict(zip(g_shape_synsets, [os.path.join(g_truncation_distribution_folder, name+'.txt') for name in g_shape_names]))

# render_model_views
g_syn_light_num_lowbound = 0
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 20
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = -90
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 2
g_syn_light_energy_std = 2
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1

# ------------------------------------------------------------
# VIEW_ESTIMATION
# ------------------------------------------------------------
g_syn_images_lmdb_folder = os.path.join(g_data_folder, 'syn_lmdbs')
g_syn_images_lmdb_pathname_prefix = os.path.join(g_syn_images_lmdb_folder, 'syn_lmdbs')
g_syn_images_resize_dim = 227
g_images_resize_dim = 227

g_real_images_folder = os.path.join(g_data_folder, 'real_images')
g_real_images_voc12val_det_bbox_folder = os.path.join(g_real_images_folder, 'voc12val_det_bbox')
g_real_images_voc12val_easy_gt_bbox_folder = os.path.join(g_real_images_folder, 'voc12val_easy_gt_bbox')
g_real_images_voc12train_all_gt_bbox_folder = os.path.join(g_real_images_folder, 'voc12train_all_gt_bbox')
g_real_images_voc12train_flip = 0
g_real_images_voc12train_aug_n = 1
g_real_images_voc12train_jitter_IoU = 1
g_real_images_lmdb_folder = os.path.join(g_data_folder, 'real_lmdbs')
g_real_images_voc12train_all_gt_bbox_lmdb_prefix = os.path.join(g_real_images_lmdb_folder, 'voc12train_all_gt_bbox_lmdb')

g_detection_results_folder = os.path.join(g_data_folder, 'detection_results')
g_rcnn_detection_bbox_mat_filelist = os.path.join(g_detection_results_folder, 'bbox_mat_filelist.txt')

# testing
g_caffe_param_file = os.path.join(g_render4cnn_root_folder,'caffe_models', 'render4cnn_3dview.caffemodel') 
g_caffe_deploy_file = os.path.join(g_render4cnn_root_folder, 'caffe_models', 'deploy.prototxt') 
g_image_mean_file = os.path.join(g_render4cnn_root_folder, 'caffe_models', 'imagenet_mean.npy')
g_image_mean_binaryproto_file = os.path.join(g_render4cnn_root_folder, 'train', 'imagenet_mean_227x227.binaryproto')
g_caffe_prob_keys = ['fc-azimuth','fc-elevation','fc-tilt']
g_test_batch_size = 256

# Image-keypoint info
g_image_keypoint_info_folder = os.path.join(g_data_folder, 'image_keypoint_info')
g_syn_image_keypoint_info_file = os.path.join(g_image_keypoint_info_folder, 'syn_image_keypoint_info.csv')
g_pascal_train_image_keypoint_info_file = os.path.join(g_image_keypoint_info_folder, 'pascal_train_image_keypoint_info.csv')
g_pascal_test_image_keypoint_info_file = os.path.join(g_image_keypoint_info_folder, 'pascal_test_image_keypoint_info.csv')

# LMDB data folders
g_corresp_lmdb_data_folder = os.path.join(g_data_folder, 'lmdb_data')
g_corresp_syn_lmdb_data_folder = os.path.join(g_corresp_lmdb_data_folder, 'syn')
g_corresp_pascal_train_lmdb_data_folder = os.path.join(g_corresp_lmdb_data_folder, 'pascal', 'train')
g_corresp_pascal_test_lmdb_data_folder = os.path.join(g_corresp_lmdb_data_folder, 'pascal', 'test')

# LMDB folders on /z
g_corresp_lmdb_folder = os.path.join(g_data_folder, 'lmdb')
g_corresp_syn_train_lmdb_folder = os.path.join(g_corresp_lmdb_folder, 'syn')
g_corresp_pascal_train_lmdb_folder = os.path.join(g_corresp_lmdb_folder, 'pascal', 'train')
g_corresp_pascal_test_lmdb_folder = os.path.join(g_corresp_lmdb_folder, 'pascal', 'test')

# LMDB folders on /scratch
g_scratch_corresp_lmdb_folder = '/scratch/home/szetor/Documents/DENSO_VCA/RenderForCNN/data/lmdb-new'
g_scratch_corresp_syn_train_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'syn', 'train')
g_scratch_corresp_syn_val_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'syn', 'val')
g_scratch_corresp_real_train_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'real', 'train')
g_scratch_corresp_real_train_train_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'real', 'train_train')
g_scratch_corresp_real_train_val_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'real', 'train_val')
g_scratch_corresp_real_test_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'real', 'test')
g_scratch_corresp_real_test_perturbed_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'real', 'test_perturbed')
g_scratch_corresp_real_test_det_lmdb_folder = os.path.join(g_scratch_corresp_lmdb_folder, 'real', 'test_det')

# correspondence model location
g_corresp_model_root_folder = os.path.join(g_render4cnn_root_folder, 'train')
g_render4cnn_weights_path = os.path.join(g_corresp_model_root_folder, 'render4cnn_3dview.caffemodel')
g_alexnet_weights_path = os.path.join(g_corresp_model_root_folder, 'bvlc_alexnet.caffemodel')

# Experiments folder
g_experiments_root_folder = os.path.join(g_render4cnn_root_folder, 'experiments')
g_experiments_snapshot_root_folder = os.path.join(g_experiments_root_folder, 'snapshots')
