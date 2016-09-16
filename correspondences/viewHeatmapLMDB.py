import numpy as np
import matplotlib.pyplot as plt
import skimage.draw
from createHeatmapUtils import *
import skimage


# Examine correspondence LMDBs
N = 10
imgSet = 'test'
scaledImgs = getFirstNLmdbImgs(imgSet + '/scaledLMDB', N)
heatmapImgs = getFirstNLmdbImgs(imgSet + '/heatmapLMDB', N)
keyptClassVecs = getFirstNLmdbVecs(imgSet + '/keyptClassLMDB', N)
viewptLabelVecs = getFirstNLmdbVecs(imgSet + '/viewptLabelLMDB', N)
keys = scaledImgs.keys()
print('Found %d keys for %s LMDBs' % (len(keys), imgSet))

# Initialize figure and subplot handles for correspondence visualization
# f, ax = plt.subplots(1, 3)
f, ax = plt.subplots(1, 2)
pltTitle = plt.suptitle('')

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
    # Get azimuth/elevation/rotation from viewpoint label
    az = viewptLabel[1] - 360*viewptLabel[0]
    el = viewptLabel[2] - 360*viewptLabel[0]
    ro = viewptLabel[3] - 360*viewptLabel[0]

    # Visualize info
    ax[0].imshow(img)
    ax[1].imshow(keyptOnImg)
    # ax[1].imshow(heatmap, cmap='Greys_r')
    # ax[2].imshow(keyptOnImg)
    # Hide axes
    plt.setp([a.get_xticklabels() for a in ax], visible=False)
    plt.setp([a.get_yticklabels() for a in ax], visible=False)
    # Set title
    title = ''
    title += '(Azimuth, Elevation, In-plane rot): (%d, %d, %d)\n' % (az, el, ro)
    title += 'Keypoint location: (%d, %d)\n' % (keyptLocI, keyptLocJ)
    title += 'Keypoint class: %s\n' % keyptClassLabel
    pltTitle.set_text(title)
    # Draw plots
    plt.draw()

    # # Save image
    # plt.imsave('image.png', img)
    # plt.imsave('heatmap.png', heatmap, cmap='Greys_r')
    # plt.imsave('keypoint.png', keyptOnImg)

    plt.waitforbuttonpress()

'''

# Examine Render for CNN LMDBs
r4cnnImgs = getFirstNLmdbImgs('/home/szetor/Documents/DENSO_VAC/RenderForCNN/data/real_lmdbs/voc12val_easy_gt_bbox/scaledLMDB', 1e10)
r4cnnLabels = getFirstNLmdbVecs('/home/szetor/Documents/DENSO_VAC/RenderForCNN/data/real_lmdbs/voc12val_easy_gt_bbox/viewptLabelLMDB', 1e10)
keys = r4cnnImgs.keys()

for key in keys:
   r4cnnImg = r4cnnImgs[key]
   r4cnnViewptLabel = r4cnnLabels[key]

   # plt.figure()
   print(key)
   plt.imshow(r4cnnImg)
   plt.title('%s' % (r4cnnViewptLabel,))
   plt.draw()
   plt.waitforbuttonpress()

plt.close()

'''