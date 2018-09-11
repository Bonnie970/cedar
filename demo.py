import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

#%matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #IMAGE_MAX_DIM = 1080
    # NUM_CLASSES = 2 doesn't work cuz coco is pretrained with 81 classes
    #USE_MINI_MASK = False
    DETECTION_MIN_CONFIDENCE = 0.9

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def mask_expend(masks, n_pixel):
    num_obj = masks.shape[-1]
    newmasks = []
    for obj in np.dsplit(masks, num_obj):
        m1 = obj.reshape(len(masks), len(masks[0]))
        #m1 = np.array([[0,0,0,0,],[0,1,1,0],[0,1,1,0],[0,0,0,0], [0,0,0,0]])
        m2 = m1.T.copy()
        for x in [m1, m2]:
            for row in x:
                if not np.any(row):
                    continue
                for pos in np.where(row[:-1] != row[1:])[0]:
                    if row[pos] == 0:
                        i = 0
                        while i < n_pixel:
                            if pos >= 0:
                                row[pos] = 1
                                pos -= 1
                                i += 1
                            else:
                                break
                    else:
                        i = 0
                        pos += 1
                        while i < n_pixel:
                            if pos < len(row):
                                row[pos] = 1
                                pos += 1
                                i += 1
                            else:
                                break
        m = (m1+m2.T)>=1
        newmasks.append(m)
    return np.dstack(tuple(newmasks))

#def detect_image(image):

image = os.path.join(IMAGE_DIR, '7-1.0.jpg')
image = skimage.io.imread(image)  # random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]

trimaps = [r['masks']]
for n_pixel in [20, 40]:
    trimap = mask_expend(r['masks'].copy(), n_pixel)
    trimaps.append(trimap)
trimaps = np.dstack(tuple(trimaps))

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
visualize.display_instances(image, r['rois'], trimaps, r['class_ids'],class_names, r['scores'])

'''for img in [7]:#[1,2,3,4,5,6,7,8]:
    # load resized pictures
    for scale in [1.0]:#[1.1, 1.25, 1.5, 1.75, 2.0]:#[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        detect_image(os.path.join(IMAGE_DIR, '{}-{}.jpg'.format(img, scale)))


detect_image(os.path.join(IMAGE_DIR, '7-1.0.jpg'))
detect_image(os.path.join(IMAGE_DIR, 'frame1.jpg'))
detect_image(os.path.join(IMAGE_DIR, '1045023827_4ec3e8ba5c_z.jpg'))
'''
