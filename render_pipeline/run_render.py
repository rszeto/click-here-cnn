#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
RENDER_ALL_SHAPES
@brief:
    render all shapes of PASCAL3D 12 rigid object classes
'''

import os
import sys
import socket
import random
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from render_helper import *

if __name__ == '__main__':
    # Get parallelization parameters from environment if available
    hostname = socket.gethostname()
    rank = int(os.environ['PBS_VNODENUM']) if 'PBS_VNODENUM' in os.environ.keys() else 0
    num_procs = int(os.environ['PBS_NUM_NODES']) if 'PBS_NUM_NODES' in os.environ.keys() else 1
    print('Hostname: %s\nRank: %d\nNum procs: %d' % (hostname, rank, num_procs))

    if not os.path.exists(g_syn_images_folder):
        os.mkdir(g_syn_images_folder) 

    all_commands = []
    tmp_dirnames = []
    for idx in g_hostname_synset_idx_map[socket.gethostname()]:
        synset = g_shape_synsets[idx]
        if not os.path.exists(os.path.join(g_syn_images_folder, synset)):
            os.mkdir(os.path.join(g_syn_images_folder, synset))
        print('%d: %s, %s\n' % (idx, synset, g_shape_names[idx]))
        shape_list = load_one_category_shape_list(synset)
        view_params = load_one_category_shape_views(synset)
        synset_commands, tmp_dirname = render_one_category_model_views_commands(synset, shape_list, view_params)
        all_commands = all_commands + synset_commands
        tmp_dirnames.append(tmp_dirname)

    print('%d commands found' % len(all_commands))

    # Get the commands that this processor should handle
    num_commands_per_proc = len(all_commands)/num_procs
    proc_commands = all_commands[rank*num_commands_per_proc:(rank+1)*num_commands_per_proc]

    # Run the commands
    print('Rendering, it takes long time...')
    report_step = 100
    pool = Pool(g_syn_rendering_thread_num)
    for idx, return_code in enumerate(pool.imap(partial(call, shell=True), proc_commands)):
        if idx % report_step == 0:
            print('[%s] Rendering command %d of %d' % (datetime.datetime.now().time(), idx, len(proc_commands)))
        if return_code != 0:
            print('Rendering command %d of %d (\"%s\") failed' % (idx, len(proc_commands), proc_commands[idx]))

    # Remove temporary directories
    for tmp_dirname in tmp_dirnames:
        shutil.rmtree(tmp_dirname)