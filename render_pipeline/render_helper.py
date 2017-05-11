#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import random
import tempfile
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *


'''
@input: 
    shape_synset e.g. '03001627' (each category has a synset)
@output: 
    a list of (synset, md5, obj_filename, view_num) for each shape of that synset category
    where synset is the input synset, md5 is md5 of one shape in the synset category,
    obj_filename is the obj file of the shape, view_num is the number of images to render for that shape
'''
def load_one_category_shape_list(shape_synset):
    shape_md5_list = os.listdir(os.path.join(g_shapenet_root_folder,shape_synset))
    # Only consider folders with keypoints. Remove next two lines all models have keypoints
    has_keypoints_fn = lambda x: os.path.exists(os.path.join(g_shapenet_root_folder, shape_synset, x, 'keypoints.json'))
    shape_md5_list = filter(has_keypoints_fn, shape_md5_list)

    shape_list = []
    for i in range(len(shape_md5_list)):
        shape_md5 = shape_md5_list[i]
        # Compute how many renders of the current model should be produced
        # First part divides num. renders across all models, second part adds one to some models to account for remainder
        view_num = g_syn_images_num_per_category/len(shape_md5_list) + int(i % len(shape_md5_list) < g_syn_images_num_per_category % len(shape_md5_list))
        shape_list.append(((shape_md5, os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'model.obj'), view_num, os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'keypoints.json'))))
    return shape_list

'''
@input: 
    shape synset
@output:
    samples of viewpoints (plus distances) from pre-generated file, each element of view_params is
    a list of azimuth,elevation,tilt angles and distance
'''
def load_one_category_shape_views(synset):
    # return shape_synset_view_params
    if not os.path.exists(g_view_distribution_files[synset]):
        print('Failed to read view distribution files from %s for synset %s' % 
              (g_view_distribution_files[synset], synset))
        exit()
    view_params = open(g_view_distribution_files[synset]).readlines()
    view_params = [[float(x) for x in line.strip().split(' ')] for line in view_params] 
    return view_params

'''
@input:
    shape_list and view_params as output of load_one_category_shape_list/views
@output:
    list of commands to save rendered images to g_syn_images_folder/<synset>/<md5>/xxx.png
''' 
def render_one_category_model_views_commands(shape_synset, shape_list, view_params):
    tmp_dirname = tempfile.mkdtemp(dir=g_data_folder, prefix='tmp_view_')
    if not os.path.exists(tmp_dirname):
        os.mkdir(tmp_dirname)

    print('Generating rendering commands...')
    commands = []
    for shape_md5, shape_file, view_num, keypoints_file in shape_list:
        # write tmp view file if any views should be computed
        if view_num > 0 and os.path.exists(keypoints_file):
            tmp = tempfile.NamedTemporaryFile(dir=tmp_dirname, delete=False)
            for i in range(view_num):
                paramId = random.randint(0, len(view_params)-1)
                tmp_string = '%f %f %f %f\n' % (view_params[paramId][0], view_params[paramId][1], view_params[paramId][2], max(0.01,view_params[paramId][3]))
                tmp.write(tmp_string)
            tmp.close()

            command = '%s %s --background --python %s -- %s %s %s %s %s %s > /dev/null 2>&1' % (
                    g_blender_executable_path,
                    g_blank_blend_file_path,
                    os.path.join(BASE_DIR, 'render_model_views.py'),
                    shape_file,
                    shape_synset,
                    shape_md5,
                    tmp.name,
                    os.path.join(g_syn_images_folder, shape_synset, shape_md5),
                    keypoints_file)
            commands.append(command)
    print('done (%d commands)!'%(len(commands)))
    return (commands, tmp_dirname)