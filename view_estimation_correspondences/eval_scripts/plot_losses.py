import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plotLosses(args):
    trainLog = args.prefix + '.train'
    testLog = args.prefix + '.test'
    savePath = args.savePath
    
    # Extract information from training table
    # Note: Last row may be incomplete. Blank values are nan
    with open(trainLog) as f:
        splitLines = [line.split() for line in f.readlines()]
        header = splitLines[0]
        dataAsStr = splitLines[1:]
        trainTable = np.full((len(dataAsStr), len(header)), np.nan)
        for i,line in enumerate(dataAsStr):
            trainTable[i, :len(line)] = line
        # Append total loss as last column
        totalLoss = np.sum(trainTable[:, 3:], axis=1).reshape((trainTable.shape[0]), 1)
        trainTable = np.hstack((trainTable, totalLoss))
        # Add total loss to header
        header.append('Loss_Total')

    # Plot training loss curves
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(trainTable[:, 0], trainTable[:, 3], color='g', label='Azimuth')
    ax.plot(trainTable[:, 0], trainTable[:, 4], color='r', label='Elevation')
    ax.plot(trainTable[:, 0], trainTable[:, 5], color='c', label='Tilt')
    ax.plot(trainTable[:, 0], trainTable[:, 6], color='b', label='Total')
    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Add labels
    ax.set_title(savePath + '\nTraining loss vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0, 120])
    # Save figure if specified
    if savePath:
        plt.savefig(os.path.join(savePath, 'trainLoss.png'))

    # Extract information from testing table
    # Note: Last row may be incomplete. Blank values are nan
    with open(testLog) as f:
        splitLines = [line.split() for line in f.readlines()]
        header = splitLines[0]
        dataAsStr = splitLines[1:]
        testTable = np.full((len(dataAsStr), len(header)), np.nan)
        for i,line in enumerate(dataAsStr):
            testTable[i, :len(line)] = line
        # Append total loss
        totalLoss = np.sum(testTable[:, 5:8], axis=1).reshape(testTable.shape[0], 1)
        testTable = np.hstack((testTable, totalLoss))
        # Add total loss to header
        header.append('Loss_Total')
        # Append total acc
        totalAcc = np.sum(testTable[:, 2:5], axis=1).reshape(testTable.shape[0], 1)
        testTable = np.hstack((testTable, totalAcc))
        header.append('Acc_Total')
    
    # Plot testing loss curves
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(testTable[:, 0], testTable[:, 5], color='g', label='Azimuth')
    ax.plot(testTable[:, 0], testTable[:, 6], color='r', label='Elevation')
    ax.plot(testTable[:, 0], testTable[:, 7], color='c', label='Tilt')
    ax.plot(testTable[:, 0], testTable[:, 8], color='b', label='Total')
    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Add labels
    ax.set_title(savePath + '\nTesting loss vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0, 120])
    # Save figure if specified
    if savePath:
        plt.savefig(os.path.join(savePath, 'testLoss.png'))

    # Plot testing acc curves
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(testTable[:, 0], testTable[:, 2], color='g', label='Azimuth')
    ax.plot(testTable[:, 0], testTable[:, 3], color='r', label='Elevation')
    ax.plot(testTable[:, 0], testTable[:, 4], color='c', label='Tilt')
    ax.plot(testTable[:, 0], testTable[:, 9], color='b', label='Total')
    # Adjust graph so legend appears to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Add labels
    ax.set_title(savePath + '\nTesting accuracy vs. iterations')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Sum of accuracies')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim([0, 2.5])
    # Save figure if specified
    if savePath:
        plt.savefig(os.path.join(savePath, 'testAcc.png'))

    # Show plots if specified
    if args.showPlots:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help='Prefix of the .train and .test files. E.g. ###.log will draw plots from ###.log.train and ###.log.test.')
    parser.add_argument('--savePath', type=str, help='Directory where plots should be saved. Nothing is saved if not specified.')
    parser.add_argument('--showPlots', action='store_true', help='Flag to open plots in a pyplot window.')
    args = parser.parse_args()
    plotLosses(args)
