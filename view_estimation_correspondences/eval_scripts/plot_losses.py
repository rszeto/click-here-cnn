# Set Matplotlib to not require X server (for cron use)
import matplotlib
matplotlib.use('Agg')
# Other imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re
from pprint import pprint

# Sort files by progress log number. Assume files include basenames progress.*, progress_1.*, progress_2.*, ...,
# with no gaps, but may be out of order
def get_sorted_progress_files(files):
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
    files_arr = files_arr[order]
    return files_arr.tolist()

# Input: List of *.train files to stitch together
# Output: Table of values
def stitch_train_files(train_files):
    train_files = get_sorted_progress_files(train_files)
    final_data = {}
    for train_file in train_files:
        f = open(train_file, 'r')
        for i, line in enumerate(f):
            # Skip header line
            if i == 0:
                continue
            # Map from iteration number to values
            line_data_strs = line.rstrip().split()
            final_data[int(line_data_strs[0])] = [float(x) for x in line_data_strs[1:]]
        f.close()
    tab = np.zeros((len(final_data), 7))
    for i, key in enumerate(sorted(final_data)):
        tab[i, 0] = key
        tab[i, 1:6] = final_data[key]
        # Set last column as total loss
        tab[i, 6] = np.sum(final_data[key][2:])
    return tab

# Input: List of *.test files to stitch together
# Output: Table of values
def stitch_test_files(test_files):
    test_files = get_sorted_progress_files(test_files)
    final_data = {}
    for test_file in test_files:
        f = open(test_file, 'r')
        for i, line in enumerate(f):
            # Skip header line
            if i == 0:
                continue
            # Map from iteration number to values
            line_data_strs = line.rstrip().split()
            final_data[int(line_data_strs[0])] = [float(x) for x in line_data_strs[1:]]
        f.close()
    tab = np.zeros((len(final_data), 10))
    for i, key in enumerate(sorted(final_data)):
        tab[i, 0] = key
        tab[i, 1:8] = final_data[key]
        # Set second-last column as total loss
        tab[i, 8] = np.sum(final_data[key][4:7])
        # Set last column as summed accuracy
        tab[i, 9] = np.sum(final_data[key][1:4])
    return tab

def plotLosses(train_files, test_files, savePath=None, showPlots=False):
    trainTable = stitch_train_files(train_files)
    testTable = stitch_test_files(test_files)

    # Plot training loss curves
    plt.figure(figsize=(21, 6))
    ax = plt.subplot(131)
    ax.plot(trainTable[:, 0], trainTable[:, 3], color='g', label='Azimuth')
    ax.plot(trainTable[:, 0], trainTable[:, 4], color='r', label='Elevation')
    ax.plot(trainTable[:, 0], trainTable[:, 5], color='c', label='Tilt')
    ax.plot(trainTable[:, 0], trainTable[:, 6], color='b', label='Total')
    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title('Training loss vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_xlim([0, 90000])
    ax.set_ylim([0, 120])
    # ax.set_ylim([50, 60])

    # Plot testing loss curves
    ax = plt.subplot(132)
    ax.plot(testTable[:, 0], testTable[:, 5], color='g', label='Azimuth')
    ax.plot(testTable[:, 0], testTable[:, 6], color='r', label='Elevation')
    ax.plot(testTable[:, 0], testTable[:, 7], color='c', label='Tilt')
    ax.plot(testTable[:, 0], testTable[:, 8], color='b', label='Total')
    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title('Testing loss vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_xlim([0, 90000])
    ax.set_ylim([0, 120])
    # ax.set_ylim([70, 80])

    # Plot testing acc curves
    ax = plt.subplot(133)
    ax.plot(testTable[:, 0], testTable[:, 2], color='g', label='Azimuth')
    ax.plot(testTable[:, 0], testTable[:, 3], color='r', label='Elevation')
    ax.plot(testTable[:, 0], testTable[:, 4], color='c', label='Tilt')
    ax.plot(testTable[:, 0], testTable[:, 9], color='b', label='Total')
    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title('Testing accuracy vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Sum of accuracies')
    # ax.set_xlim([0, 90000])
    ax.set_ylim([0, 3])
    # ax.set_ylim([2, 2.35])

    # Make global plot adjustments
    plt.suptitle(savePath if savePath else '')
    plt.tight_layout(pad=3, w_pad=18)

    # Save figure if specified
    if savePath:
        plt.savefig(os.path.join(savePath, 'plots.png'))
    # Show plots if specified
    if showPlots:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str, help='Path to the experiment to draw plots for')
    parser.add_argument('--savePath', type=str, help='Directory where plots should be saved. Nothing is saved if not specified.')
    parser.add_argument('--showPlots', action='store_true', help='Flag to open plots in a pyplot window.')
    args = parser.parse_args()

    exp_progress_root = os.path.join(args.exp_root, 'progress')
    train_files = glob.glob(os.path.join(exp_progress_root, 'progress*.log.train'))
    test_files = glob.glob(os.path.join(exp_progress_root, 'progress*.log.test'))
    plotLosses(train_files, test_files, savePath=args.savePath, showPlots=args.showPlots)