import os
from pprint import pprint
from scipy.ndimage import imread
from scipy.misc import imresize
from createHeatmapUtils import *
import numpy as np
from warnings import warn
import lmdb
import csv
import glob

'''
@brief:
    convert 360 view degree to view estimation label
    e.g. for bicycle with class_idx 1, label will be 360~719
'''
def view2label(degree, class_index):
    return int(degree)%360 + class_index*360

def keypointInfo2Arr(fullImagePath, bbox, keyptLoc, keyptClass, viewptLabel):
    return np.array([fullImagePath, bbox[0], bbox[1], bbox[2], bbox[3], keyptLoc[0], keyptLoc[1], keyptClass, viewptLabel[0], viewptLabel[1], viewptLabel[2], viewptLabel[3]])

# Make LMDB directories if needed
lmdbDirsToMake = ['train_full', 'train_split_train', 'train_split_val', 'test']
for lmdbDir in lmdbDirsToMake:
    if not os.path.isdir(lmdbDir):
        os.mkdir(lmdbDir)

# Open scaled image LMDB (30 GB = 32212254720 bytes)
# 154587 B/keypt * 52625 keypts * 2 flip ~ 15.15 GB
scaledLMDBTrainFull = lmdb.open('train_full/scaledLMDB', map_size=32212254720)
scaledLMDBTrainTrain = lmdb.open('train_split_train/scaledLMDB', map_size=32212254720)
scaledLMDBTrainVal = lmdb.open('train_split_val/scaledLMDB', map_size=32212254720)
scaledLMDBTest = lmdb.open('test/scaledLMDB', map_size=32212254720)
# Open heatmap image LMDB (10 GB = 10737418240 bytes)
# 51529 B/keypt * 52625 keypts * 2 flip ~ 4.8 GB
heatmapLMDBTrainFull = lmdb.open('train_full/heatmapLMDB', map_size=10737418240)
heatmapLMDBTrainTrain = lmdb.open('train_split_train/heatmapLMDB', map_size=10737418240)
heatmapLMDBTrainVal = lmdb.open('train_split_val/heatmapLMDB', map_size=10737418240)
heatmapLMDBTest = lmdb.open('test/heatmapLMDB', map_size=10737418240)
# Open one-hot keypoint LMDB (50 MB = 52428800 bytes)
# 124 B/keypt * 52625 keypts * 2 flip ~ 12.44 MB
keyptClassLMDBTrainFull = lmdb.open('train_full/keyptClassLMDB', map_size=52428800)
keyptClassLMDBTrainTrain = lmdb.open('train_split_train/keyptClassLMDB', map_size=52428800)
keyptClassLMDBTrainVal = lmdb.open('train_split_val/keyptClassLMDB', map_size=52428800)
keyptClassLMDBTest = lmdb.open('test/keyptClassLMDB', map_size=52428800)
# Open viewpoint label LMDB (10 MB = 10485760 bytes)
# 32 B/keypt * 52625 keypts * 2 flip ~ 3.2 MB
viewptLabelLMDBTrainFull = lmdb.open('train_full/viewptLabelLMDB', map_size=10485760)
viewptLabelLMDBTrainTrain = lmdb.open('train_split_train/viewptLabelLMDB', map_size=10485760)
viewptLabelLMDBTrainVal = lmdb.open('train_split_val/viewptLabelLMDB', map_size=10485760)
viewptLabelLMDBTest = lmdb.open('test/viewptLabelLMDB', map_size=10485760)

# Count total number of visible keypoints in PASCAL3D+
totalNumKeypts = 0
# Set maximum number of keypoints (reduce for testing)
maxNumKeypts = 1e10

# Initialize arrays to store keypoint info
infoArrTrainFull = np.zeros((0, 12))
infoArrTrainTrain = np.zeros((0, 12))
infoArrTrainVal = np.zeros((0, 12))
infoArrTest = np.zeros((0, 12))

# Generate info for each keypoint in PASCAL3D+ and store in CSV files
numInfoFiles = len(glob.glob('info-*.csv'))
if numInfoFiles < 4:
    # Generate train and test lists and store in file
    matlab_cmd = 'getPascalTrainValImgs'
    os.system('matlab -nodisplay -r "try %s ; catch; end; quit;"' % matlab_cmd)
    # Get training and test image IDs
    with open('trainImgIds.txt', 'rb') as trainIdsFile:
        trainIds = np.loadtxt(trainIdsFile, dtype='string')
        # Shuffle IDs to prevent same-object clumps in case IDs exhibit some sort of ordering
        np.random.shuffle(trainIds)
        # Get IDs for train and validation splits
        trainTrainIds = trainIds[:int(np.floor(.8*trainIds.size))]
        trainValIds = trainIds[int(np.floor(.8*trainIds.size)):]
    with open('valImgIds.txt', 'rb') as testIdsFile:
        testIds = np.loadtxt(testIdsFile, dtype='string')
    # Delete the ID files
    os.remove('trainImgIds.txt')
    os.remove('valImgIds.txt')
    
    for classIdx, rigidClass in enumerate(getRigidClasses()):
        for datasetSource in DATASET_SOURCES:
            classSourceId = '%s_%s' % (rigidClass, datasetSource)
            for annoFile in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, classSourceId))):
                annoFileId = os.path.splitext(os.path.basename(annoFile))[0]
                if annoFileId in trainTrainIds:
                    annoFileSet = 'train_split_train'
                elif annoFileId in trainValIds:
                    annoFileSet = 'train_split_val'
                elif annoFileId in testIds:
                    annoFileSet = 'test'
                else:
                    continue
                anno = loadmat(os.path.join(ANNOTATIONS_ROOT, classSourceId, annoFile))['record']
                fullImagePath = os.path.join(IMAGES_ROOT, classSourceId, anno['filename'])
                fullImage = imread(fullImagePath)
                # Convert grayscale images to "color"
                if fullImage.ndim == 2:
                    fullImage = np.dstack((fullImage, fullImage, fullImage))
    
                # Make objs an array regardless of how many objects there are
                objs = np.array([anno['objects']]) if isinstance(anno['objects'], dict) else anno['objects']
                for objI,obj in enumerate(objs):
                    # Only deal with objects in current class
                    if obj['class'] == rigidClass:
                        # Get crop using bounding box from annotation
                        bbox = np.array([int(x) for x in obj['bbox']])
                        croppedImg = fullImage[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                        # Some gt bounding boxes are outside of the image (?!). If this happens, skip the object
                        if croppedImg.size <= 0:
                            warn('Object %d in %s has a bad bounding box, skipping' % (objI, anno['filename']))
                            continue
                        # Scale the object
                        scaledImg = imresize(croppedImg, (IMAGE_SIZE, IMAGE_SIZE))
    
                        # Get visible and in-frame keypoints
                        keypts = obj['anchors']
                        for keyptType in KEYPOINT_TYPES[rigidClass]:
                            keyptLocFull = keypts[keyptType]['location']
                            if keyptLocFull.size > 0 and insideBox(keyptLocFull, bbox):
                                # Keypoint is valid, so save data associated with it
                            
                                # Calculate keypoint location inside box
                                keyptLoc = keyptLocFull - bbox[:2]
                                # Get viewpoint label
                                viewpoint = obj['viewpoint']
                                azimuth = np.mod(np.round(viewpoint['azimuth']), 360)
                                elevation = np.mod(np.round(viewpoint['elevation']), 360)
                                tilt = np.mod(np.round(viewpoint['theta']), 360)
                                finalLabel = np.array([classIdx, view2label(azimuth, classIdx), view2label(elevation, classIdx), view2label(tilt, classIdx)])
                                # Add info for current keypoint
                                keyptArr = keypointInfo2Arr(fullImagePath, bbox, keyptLocFull, getKeyptTypeId(rigidClass, keyptType), finalLabel)
                                if annoFileSet == 'train_split_train':
                                    infoArrTrainTrain = np.vstack((infoArrTrainTrain, keyptArr))
                                    infoArrTrainFull = np.vstack((infoArrTrainFull, keyptArr))
                                elif annoFileSet == 'train_split_val':
                                    infoArrTrainVal = np.vstack((infoArrTrainVal, keyptArr))
                                    infoArrTrainFull = np.vstack((infoArrTrainFull, keyptArr))
                                else:
                                    infoArrTest = np.vstack((infoArrTest, keyptArr))
                                
                                # Update number of keypoints
                                totalNumKeypts += 1
                                # Print number of keypoints found
                                if totalNumKeypts % 1000 == 0:
                                    pprint('Found %d keypoints' % totalNumKeypts)
                                # If we reached max number of keypoints, leave
                                if totalNumKeypts >= maxNumKeypts: break
                    if totalNumKeypts >= maxNumKeypts: break
                if totalNumKeypts >= maxNumKeypts: break
            if totalNumKeypts >= maxNumKeypts: break
        if totalNumKeypts >= maxNumKeypts: break
    
    # Print size info
    print('==============')
    print('Number of keypoints found: %d' % totalNumKeypts)
    
    # Write to CSV
    # Training - train split
    with open('info-train_split_train.csv', 'wb') as infoFile:
        # Write header
        infoFile.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
        writer = csv.writer(infoFile, delimiter=',')
        writer.writerows(infoArrTrainTrain)
    # Training - validation split
    with open('info-train_split_val.csv', 'wb') as infoFile:
        # Write header
        infoFile.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
        writer = csv.writer(infoFile, delimiter=',')
        writer.writerows(infoArrTrainVal)
    # Training full
    with open('info-train_full.csv', 'wb') as infoFile:
        # Write header
        infoFile.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
        writer = csv.writer(infoFile, delimiter=',')
        writer.writerows(infoArrTrainFull)
    with open('info-test.csv', 'wb') as infoFile:
        # Write header
        infoFile.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
        writer = csv.writer(infoFile, delimiter=',')
        writer.writerows(infoArrTest)

# Read in the CSV files and shuffle the rows
infoArrTrainFull = np.loadtxt('info-train_full.csv', delimiter=',', dtype='string', skiprows=1)
infoArrTrainTrain = np.loadtxt('info-train_split_train.csv', delimiter=',', dtype='string', skiprows=1)
infoArrTrainVal = np.loadtxt('info-train_split_val.csv', delimiter=',', dtype='string', skiprows=1)
infoArrTest = np.loadtxt('info-test.csv', delimiter=',', dtype='string', skiprows=1)
np.random.shuffle(infoArrTrainFull)
np.random.shuffle(infoArrTrainTrain)
np.random.shuffle(infoArrTrainVal)
np.random.shuffle(infoArrTest)

# Generate LMDBs. Currently, flipped images are next to each other, so this may lower performance.
# This should be corrected if results are promising
flip = True
# Save cropped and scaled object
saveScaledImgs(infoArrTrainFull, scaledLMDBTrainFull, flip=True)
saveScaledImgs(infoArrTrainTrain, scaledLMDBTrainTrain, flip=True)
saveScaledImgs(infoArrTrainVal, scaledLMDBTrainVal, flip=False)
saveScaledImgs(infoArrTest, scaledLMDBTest, flip=False)
# Save heatmap LMDB
saveHeatmaps(infoArrTrainFull, heatmapLMDBTrainFull, flip=True)
saveHeatmaps(infoArrTrainTrain, heatmapLMDBTrainTrain, flip=True)
saveHeatmaps(infoArrTrainVal, heatmapLMDBTrainVal, flip=False)
saveHeatmaps(infoArrTest, heatmapLMDBTest, flip=False)
# Save keypoint class LMDB
saveKeyptClasses(infoArrTrainFull, keyptClassLMDBTrainFull, flip=True)
saveKeyptClasses(infoArrTrainTrain, keyptClassLMDBTrainTrain, flip=True)
saveKeyptClasses(infoArrTrainVal, keyptClassLMDBTrainVal, flip=False)
saveKeyptClasses(infoArrTest, keyptClassLMDBTest, flip=False)
# Save viewpoint label LMDB
saveViewptLabels(infoArrTrainFull, viewptLabelLMDBTrainFull, flip=True)
saveViewptLabels(infoArrTrainTrain, viewptLabelLMDBTrainTrain, flip=True)
saveViewptLabels(infoArrTrainVal, viewptLabelLMDBTrainVal, flip=False)
saveViewptLabels(infoArrTest, viewptLabelLMDBTest, flip=False)