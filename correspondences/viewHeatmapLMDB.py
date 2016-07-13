import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
from createHeatmapUtils import *
import skimage

# Examine correspondence LMDBs 
scaledImgs = getFirstNLmdbImgs('scaledLMDB', 100)
heatmapImgs = getFirstNLmdbImgs('heatmapLMDB', 100)
keyptClassVecs = getFirstNLmdbVecs('keyptClassLMDB', 100)
viewptLabelVecs = getFirstNLmdbVecs('viewptLabelLMDB', 100)
keys = scaledImgs.keys()

# Initialize figure and subplot handles for correspondence visualization
f, ax = plt.subplots(1, 3)

for key in keys:
    img = scaledImgs[key]
    print('Image size: %s' % (img.shape,))

    # Extract keypoint location
    heatmap = heatmapImgs[key]
    keyptLocI, keyptLocJ = np.nonzero(heatmap)
    keyptLocI = keyptLocI[0]
    keyptLocJ = keyptLocJ[0]
    # Show info about heatmap
    print('Heatmap size: %s' % (heatmap.shape,))
    
    # Show keypoint on image
    keyptOnImg = np.copy(img)
    rr,cc = skimage.draw.circle(keyptLocI, keyptLocJ, 5, (IMAGE_SIZE, IMAGE_SIZE))
    keyptOnImg[rr, cc, :] = [255, 0, 0]
    
    # Print label for given keypoint
    keyptClassVec = keyptClassVecs[key]
    keyptClassLabel = getKeyptTypeLabel(np.nonzero(keyptClassVec)[0][0])
    print('Keypoint class label: %s' % keyptClassLabel)
    
    # Get viewpoint label (for comparison with Render for CNN)
    viewptLabel = viewptLabelVecs[key]
    print('Viewpoint label: %s' % (viewptLabel,))
    
    # Visualize info
    ax[0].imshow(img)
    ax[0].set_title('%s' % (viewptLabel,))
    ax[1].imshow(heatmap, cmap='Greys_r')
    ax[1].set_title('Row: %d, Col: %d' % (keyptLocI, keyptLocJ))
    ax[2].imshow(keyptOnImg)
    ax[2].set_title(keyptClassLabel)
    plt.draw()
    plt.waitforbuttonpress()

# Examine Render for CNN LMDBs
r4cnnImgs = getFirstNLmdbImgs('/home/szetor/Documents/DENSO_VAC/RenderForCNN/data/real_lmdbs/voc12train_all_gt_bbox_lmdb_image', 1)
r4cnnLabels = getFirstNLmdbVecs('/home/szetor/Documents/DENSO_VAC/RenderForCNN/data/real_lmdbs/voc12train_all_gt_bbox_lmdb_label', 1)
keys = r4cnnImgs.keys()

for key in keys:
    r4cnnImg = r4cnnImgs[key]
    r4cnnViewptLabel = r4cnnLabels[key]
    print('Image size: %s' % (r4cnnImg.shape,))
    print('Viewpoint label: %s' % (r4cnnViewptLabel,))
    
    plt.figure()
    plt.imshow(r4cnnImg)
    plt.title('%s' % (r4cnnViewptLabel,))
    plt.show()