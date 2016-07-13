import os
from pprint import pprint
from scipy.ndimage import imread
from scipy.misc import imresize
from createHeatmapUtils import *
import numpy as np
from warnings import warn
import lmdb
import csv

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
if not os.path.isdir('train'):
    os.mkdir('train')
if not os.path.isdir('val'):
    os.mkdir('val')

# Open scaled image LMDB (30 GB = 32212254720 bytes)
# 154587 B/keypt * 182528 keypts ~ 26.28 GB
scaledLMDBTrain = lmdb.open('train/scaledLMDB', map_size=32212254720)
scaledLMDBVal = lmdb.open('val/scaledLMDB', map_size=32212254720)
# Open heatmap image LMDB (10 GB = 10737418240 bytes)
# 51529 B/keypt * 182528 keypts ~ 8.76 GB
heatmapLMDBTrain = lmdb.open('train/heatmapLMDB', map_size=10737418240)
heatmapLMDBVal = lmdb.open('val/heatmapLMDB', map_size=10737418240)
# Open one-hot keypoint LMDB (50 MB = 52428800 bytes)
# 124 B/keypt * 182528 keypts ~ 21.58 MB
keyptClassLMDBTrain = lmdb.open('train/keyptClassLMDB', map_size=52428800)
keyptClassLMDBVal = lmdb.open('val/keyptClassLMDB', map_size=52428800)
# Open viewpoint label LMDB (10 MB = 10485760 bytes)
# 32 B/keypt * 182528 keypts ~ 5.5 MB
viewptLabelLMDBTrain = lmdb.open('train/viewptLabelLMDB', map_size=10485760)
viewptLabelLMDBVal = lmdb.open('val/viewptLabelLMDB', map_size=10485760)
# Count total number of visible keypoints in PASCAL3D+
totalNumKeypts = 0
# Set maximum number of keypoints (reduce for testing)
maxNumKeypts = 1e10

# Initialize arrays to store keypoint info
infoArrTrain = np.zeros((0, 12))
infoArrVal = np.zeros((0, 12))

# Generate info for each keypoint in PASCAL3D+ and store in CSV files
if not (os.path.exists('infoFileTrain.csv') and os.path.exists('infoFileTrain.csv')):
    # Generate train and val lists and store in file
    matlab_cmd = 'getPascalTrainValImgs'
    os.system('matlab -nodisplay -r "try %s ; catch; end; quit;"' % matlab_cmd)
    # Get training and validation IDs
    with open('trainImgIds.txt', 'rb') as trainIdsFile:
        trainIds = np.loadtxt(trainIdsFile, dtype='string')
    with open('valImgIds.txt', 'rb') as valIdsFile:
        valIds = np.loadtxt(valIdsFile, dtype='string')
    # Delete the ID files
    os.remove('trainImgIds.txt')
    os.remove('valImgIds.txt')
    
    for classIdx, rigidClass in enumerate(getRigidClasses()):
        for datasetSource in DATASET_SOURCES:
            classSourceId = '%s_%s' % (rigidClass, datasetSource)
            for annoFile in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, classSourceId))):
                annoFileId = os.path.splitext(os.path.basename(annoFile))[0]
                if annoFileId in trainIds:
                    annoFileSet = 'train'
                elif annoFileId in valIds:
                    annoFileSet = 'val'
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
                                if annoFileSet == 'train':
                                    infoArrTrain = np.vstack((infoArrTrain, keypointInfo2Arr(fullImagePath, bbox, keyptLocFull, getKeyptTypeId(rigidClass, keyptType), finalLabel)))
                                else:
                                    infoArrVal = np.vstack((infoArrVal, keypointInfo2Arr(fullImagePath, bbox, keyptLocFull, getKeyptTypeId(rigidClass, keyptType), finalLabel)))
                                
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
    with open('infoFileTrain.csv', 'wb') as infoFileTrain:
        # Write header
        infoFileTrain.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
        # Write rows    
        infoFileTrainWriter = csv.writer(infoFileTrain, delimiter=',')
        infoFileTrainWriter.writerows(infoArrTrain)
    with open('infoFileVal.csv', 'wb') as infoFileVal:
        # Write header
        infoFileVal.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
        # Write rows    
        infoFileValWriter = csv.writer(infoFileVal, delimiter=',')
        infoFileValWriter.writerows(infoArrVal)

# Read in the CSV files and shuffle the rows
infoArrTrainShuf = np.loadtxt('infoFileTrain.csv', delimiter=',', dtype='string', skiprows=1)
infoArrValShuf = np.loadtxt('infoFileVal.csv', delimiter=',', dtype='string', skiprows=1)
np.random.shuffle(infoArrTrainShuf)
np.random.shuffle(infoArrValShuf)

# Save cropped and scaled object
saveScaledImgs(infoArrTrainShuf, scaledLMDBTrain)
saveScaledImgs(infoArrValShuf, scaledLMDBVal)
# Save heatmap LMDB
saveHeatmaps(infoArrTrainShuf, heatmapLMDBTrain)
saveHeatmaps(infoArrValShuf, heatmapLMDBVal)
# Save keypoint class LMDB
saveKeyptClasses(infoArrTrainShuf, keyptClassLMDBTrain)
saveKeyptClasses(infoArrValShuf, keyptClassLMDBVal)
# Save viewpoint label LMDB
saveViewptLabels(infoArrTrainShuf, viewptLabelLMDBTrain)
saveViewptLabels(infoArrValShuf, viewptLabelLMDBVal)