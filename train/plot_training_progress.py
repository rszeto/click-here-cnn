# Set Matplotlib to not require X server (for cron use)
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import re
from collections import OrderedDict

# Import global variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from global_variables import *

def get_sorted_progress_files(files):
    '''
    Sort files by progress log number. Assume files include basenames progress.*, progress_1.*, progress_2.*, ...,
    with no gaps, but may be out of order
    :param files: A list of progress files.
    :return:
    '''
    files_arr = np.array(files)
    order = []
    for file in files:
        # Check file name starts with "progress"
        basename = os.path.basename(file)
        m = re.match(r'progress(_\d+)*', basename)
        if m and not m.group(1):
            # match is 'progress'
            order.append(0)
        elif m and m.group(1):
            # match is 'progress_*', first group is _*
            order.append(int(m.group(1)[1:]))
    files_arr = files_arr[np.argsort(order)]
    return files_arr.tolist()

def lines_to_acc_loss(lines):
    '''
    Extract iteration numbers as well as accuracy and loss for three rotation angles
    :param lines: The lines containing the above information
    :return:
    '''
    iters = []
    az_acc = []
    el_acc = []
    ti_acc = []
    az_loss = []
    el_loss = []
    ti_loss = []
    for line in lines:
        # Get iterations
        m = re.search('Iteration ([0-9]+)', line)
        if m:
            iters.append(int(m.group(1)))
        # Get azimuths
        m = re.search('accuracy_azimuth = (([0-9]|\.)+)', line)
        if m:
            az_acc.append(float(m.group(1)))
        m = re.search('loss_azimuth = (([0-9]|\.)+)', line)
        if m:
            az_loss.append(float(m.group(1)))
        # Get elevations
        m = re.search('accuracy_elevation = (([0-9]|\.)+)', line)
        if m:
            el_acc.append(float(m.group(1)))
        m = re.search('loss_elevation = (([0-9]|\.)+)', line)
        if m:
            el_loss.append(float(m.group(1)))
        # Get tilts
        m = re.search('accuracy_tilt = (([0-9]|\.)+)', line)
        if m:
            ti_acc.append(float(m.group(1)))
        m = re.search('loss_tilt = (([0-9]|\.)+)', line)
        if m:
            ti_loss.append(float(m.group(1)))

    # Compute sum of accuracies and losses
    acc_sum = [np.sum(t) for t in zip(az_acc, el_acc, ti_acc)]
    loss_sum = [np.sum(t) for t in zip(az_loss, el_loss, ti_loss)]

    return zip(iters, az_acc, el_acc, ti_acc, az_loss, el_loss, ti_loss, acc_sum, loss_sum)

def train_info_to_tsv(save_path, tuples):
    '''
    Save the given list of training info tuples to a file in TSV format
    :param tuples: A list of tuples, where each tuple includes iteration number, accuracies, and losses
    :return:
    '''
    with open(save_path, 'w') as f:
        f.write('#Iters\tAcc_Azimuth\tAcc_Elevation\tAcc_Tilt\tLoss_Azimuth\tLoss_Elevation\tLoss_Tilt\tAcc_Sum\tLoss_Sum\n')
        for t in tuples:
            f.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % t)

def plot_loss(ax, train_info):
    '''
    Generates a loss plot for the given data
    :param ax: The pyplot axes to draw on
    :param train_info: A list of tuples, where each tuple includes iteration number, accuracies, and losses
    :return:
    '''
    iters = [t[0] for t in train_info]
    az_loss = [t[4] for t in train_info]
    el_loss = [t[5] for t in train_info]
    ti_loss = [t[6] for t in train_info]
    loss_sum = [t[8] for t in train_info]

    ax.plot(iters, az_loss, color='g', label='Azimuth')
    ax.plot(iters, el_loss, color='r', label='Elevation')
    ax.plot(iters, ti_loss, color='c', label='Tilt')
    ax.plot(iters, loss_sum, color='b', label='Total')

    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title('Training loss vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0, 120])

def plot_acc(ax, train_info):
    '''
    Generates an accuracy plot for the given data
    :param ax: The pyplot axes to draw on
    :param train_info: A list of tuples, where each tuple includes iteration number, accuracies, and losses
    :return:
    '''
    iters = [t[0] for t in train_info]
    az_acc = [t[1] for t in train_info]
    el_acc = [t[2] for t in train_info]
    ti_acc = [t[3] for t in train_info]
    acc_sum = [t[7] for t in train_info]

    ax.plot(iters, az_acc, color='g', label='Azimuth')
    ax.plot(iters, el_acc, color='r', label='Elevation')
    ax.plot(iters, ti_acc, color='c', label='Tilt')
    ax.plot(iters, acc_sum, color='b', label='Total')

    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title('Testing accuracy vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Sum of accuracies')
    ax.set_ylim([0, 3])

def main(exp_num):
    print('Creating training plots for experiment %d...' % exp_num)
    # Get experiment path
    exp_dir = glob.glob(os.path.join(g_experiments_root_folder, '%06d*' % exp_num))
    if len(exp_dir) == 0:
        print('Experiment %d does not exist, quitting' % exp_num)
        return
    exp_dir = exp_dir[0]

    # Get progress log paths
    progress_dir = os.path.join(exp_dir, 'progress')
    progress_log_names = get_sorted_progress_files(os.listdir(progress_dir))

    # Keep track of training info across progress logs, keeping latest versions of values
    all_test_info = OrderedDict()
    all_train_info = OrderedDict()
    for progress_log_name in progress_log_names:
        # Read Caffe log file
        progress_log_contents = open(os.path.join(progress_dir, progress_log_name), 'r').read()
        # Extract info from test lines
        test_lines = re.findall('.*Test.*', progress_log_contents, re.MULTILINE)
        test_info = lines_to_acc_loss(test_lines)
        # Store/overwrite info
        for tuple in test_info:
            all_test_info[tuple[0]] = tuple

        # Extract info from training lines
        train_lines = re.findall('.*Train.*', progress_log_contents, re.MULTILINE)
        train_lines += re.findall('Iteration [0-9]+, loss = ', progress_log_contents)
        train_info = lines_to_acc_loss(train_lines)
        for tuple in train_info:
            all_train_info[tuple[0]] = tuple

    # Convert info back to lists
    all_test_info = all_test_info.values()
    all_train_info = all_train_info.values()

    # Save info to TSV files
    train_info_to_tsv(os.path.join(progress_dir, 'progress_test.log'), all_test_info)
    train_info_to_tsv(os.path.join(progress_dir, 'progress_train.log'), all_train_info)

    # Draw progress plots
    plt.figure(figsize=(21, 6))
    ax = plt.subplot(131)
    plot_loss(ax, all_train_info)
    ax = plt.subplot(132)
    plot_loss(ax, all_test_info)
    ax = plt.subplot(133)
    plot_acc(ax, all_test_info)
    # Make global plot adjustments
    plt.suptitle('Experiment %d' % exp_num)
    plt.tight_layout(pad=3, w_pad=18)
    # Save plots to file
    plt.savefig(os.path.join(progress_dir, 'plots.png'))

    print('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_num', type=int, help='The experiment number')
    args = parser.parse_args()

    main(args.exp_num)
