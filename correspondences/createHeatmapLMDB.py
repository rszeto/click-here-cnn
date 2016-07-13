import os
from os.path import *
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

def keypointInfo2Arr(infoFileWriter, fullImagePath, bbox, keyptLoc, keyptClass, viewptLabel):
    return np.array([fullImagePath, bbox[0], bbox[1], bbox[2], bbox[3], keyptLoc[0], keyptLoc[1], keyptClass, viewptLabel[0], viewptLabel[1], viewptLabel[2], viewptLabel[3]])

# Open scaled image LMDB (30 GB = 32212254720 bytes)
# 154587 B/keypt * 182528 keypts ~ 26.28 GB
scaledLMDB = lmdb.open('scaledLMDB', map_size=32212254720)
# Open heatmap image LMDB (10 GB = 10737418240 bytes)
# 51529 B/keypt * 182528 keypts ~ 8.76 GB
heatmapLMDB = lmdb.open('heatmapLMDB', map_size=10737418240)
# Open one-hot keypoint LMDB (50 MB = 52428800 bytes)
# 124 B/keypt * 182528 keypts ~ 21.58 MB
keyptClassLMDB = lmdb.open('keyptClassLMDB', map_size=52428800)
# Open viewpoint label LMDB (1 MB = 1048576 bytes)
# 4 B/keypt * 182528 keypts ~ 713 kB
viewptLabelLMDB = lmdb.open('viewptLabelLMDB', map_size=1048576)
# Count total number of visible keypoints in PASCAL3D+
totalNumKeypts = 0
# Set maximum number of keypoints (for testing)
maxNumKeypts = 10

# Create file to store keypoint info
infoFileWriter = open('infoFile.csv', 'w')
infoFileWriter.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
# Initialize array to store keypoint info
infoArr = np.zeros((0, 12))

for classIdx, rigidClass in enumerate(getRigidClasses()):
    for datasetSource in DATASET_SOURCES:
        classSourceId = '%s_%s' % (rigidClass, datasetSource)
        for annoFile in sorted(os.listdir(os.path.join(ANNOTATIONS_ROOT, classSourceId))):
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
                            infoArr = np.vstack((infoArr, keypointInfo2Arr(infoFileWriter, fullImagePath, bbox, keyptLocFull, getKeyptTypeId(rigidClass, keyptType), finalLabel)))
                            
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

np.random.shuffle(infoArr)

# Write to CSV
with open('infoFile.csv', 'wb') as infoFile:
    # Write header
    infoFile.write('imgPath,bboxTLX,bboxTLY,bboxBRX,bboxBRY,imgKeyptX,imgKeyptY,keyptClass,objClass,azimuthClass,elevationClass,rotationClass\n')
    # Write rows    
    infoFileWriter = csv.writer(infoFile, delimiter=',')
    infoFileWriter.writerows(infoArr)

# Save cropped and scaled object
with open('infoFile.csv', 'rb') as infoFileReader:
    with scaledLMDB.begin(write=True) as txn:
        csvReader = csv.reader(infoFileReader, delimiter=',')
        # Skip header row
        next(csvReader, None)
        for i, row in enumerate(csvReader):
            if i % 1000 == 0:
                pprint('Wrote %d scaled images to LMDB' % i)
            # Recover full image
            fullImagePath = row[0]
            fullImage = imread(fullImagePath)
            # Convert grayscale images to "color"
            if fullImage.ndim == 2:
                fullImage = np.dstack((fullImage, fullImage, fullImage))
            # Recover object BB
            bbox = np.array([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
            # Crop and scale the object
            croppedImg = fullImage[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            scaledImg = imresize(croppedImg, (IMAGE_SIZE, IMAGE_SIZE))
            # Save
            saveImgToLMDB(txn, '%08d' % i, scaledImg)

# Save heatmap LMDB
with open('infoFile.csv', 'rb') as infoFileReader:
    with heatmapLMDB.begin(write=True) as txn:
        csvReader = csv.reader(infoFileReader, delimiter=',')
        # Skip header row
        next(csvReader, None)
        for i, row in enumerate(csvReader):
            if i % 1000 == 0:
                pprint('Wrote %d heatmap images to LMDB' % i)
            # Recover keypoint location on full image
            keyptLocFull = [float(row[5]), float(row[6])]
            # Recover object BB and size
            bbox = np.array([int(row[1]), int(row[2]), int(row[3]), int(row[4])])
            bboxSize = np.array([bbox[2]-bbox[0]+1, bbox[3]-bbox[1]])
            # Calculate keypoint location inside box
            keyptLoc = keyptLocFull - bbox[:2]
            # Get keypoint location on IMAGE_SIZExIMAGE_SIZE scaled image
            keyptLocScaled = np.floor(IMAGE_SIZE * keyptLoc / bboxSize).astype(np.uint8)
            # Create heatmap
            heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            heatmap[keyptLocScaled[1], keyptLocScaled[0]] = 1
            # Save heatmap image
            saveImgToLMDB(txn, '%08d' % i, heatmap)

# Save keypoint class LMDB
with open('infoFile.csv', 'rb') as infoFileReader:
    with keyptClassLMDB.begin(write=True) as txn:
        csvReader = csv.reader(infoFileReader, delimiter=',')
        # Skip header row
        next(csvReader, None)
        for i, row in enumerate(csvReader):
            if i % 1000 == 0:
                pprint('Wrote %d keypoint class vectors to LMDB' % i)
            # Recover keypoint class
            keyptClass = int(row[7])
            # Create one-hot vector for keypoint class
            keyptClassVec = np.zeros(numKeyptTypes, dtype=np.uint8)
            keyptClassVec[keyptClass] = 1
            # Save vector
            saveVecToLMDB(txn, '%08d' % i, keyptClassVec)

# Save viewpoint label LMDB
with open('infoFile.csv', 'rb') as infoFileReader:
    with viewptLabelLMDB.begin(write=True) as txn:
        csvReader = csv.reader(infoFileReader, delimiter=',')
        # Skip header row
        next(csvReader, None)
        for i, row in enumerate(csvReader):
            if i % 1000 == 0:
                pprint('Wrote %d viewpoint labels to LMDB' % i)
            # Recover viewpoint label
            viewptLabel = np.array(row[8:], dtype=np.float64)
            # Save viewpoint label
            saveVecToLMDB(txn, '%08d' % i, viewptLabel)

# Print size info
print('==============')
print('Number of keypoints found: %d' % totalNumKeypts)
