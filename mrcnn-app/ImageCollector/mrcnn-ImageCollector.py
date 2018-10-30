import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

import cv2
import time


class MaskRCNN():

    def __init__(self,mask_rcnn_root):
        # Root directory of the project
        ROOT_DIR = mask_rcnn_root

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
        import coco
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)




        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
        self.ax = []
        self.plt = []


    def devide_images(self,image_path_list):
        image_save_index = 0
        for image_path in image_path_list:
            image = skimage.io.imread(image_path)
            results = self.model.detect([image],verbose=1)
            r = results[0]


            rois = r["rois"]
            N = rois.shape[0]
            class_ids = r["class_ids"]
            masks = r["masks"]
            scores = r["scores"]
            if not os.path.exists("output"):
                os.mkdir("output")
            for i in range(N):
                if not np.any(rois[i]):continue
                y1,x1,y2,x2 = rois[i]
                #print(x1,x2,y1,y2)
                #print(image.shape)
                new_image = image[y1:y2,x1:x2,:]

                #print(new_image.shape)
                mask = masks[:,:,i]


                label = self.class_names[class_ids[i]]

                plt.imshow(new_image)

                try:
                    plt.savefig("output\%s\%s%s.png"%(label,label,image_save_index))
                    image_save_index += 1
                except FileNotFoundError:

                     os.mkdir("output\%s" % label)
                     plt.savefig("output\%s\%s%s.png" % (label, label, image_save_index))
                     image_save_index += 1
                except:
                    continue








def tst():
    # set MaskRCNN root: https://github.com/matterport/Mask_RCNN
    temp = MaskRCNN("G:\wbc\GitHub\Mask_RCNN")
    # images
    temp.devide_images(["C:\\Users\Administrator\OneDrive - shanghaitech.edu.cn\图片\\"+ i for i in
                        [
                        "9247489789_132c0d534a_z.jpg",
                          "3132016470_c27baa00e8_z.jpg",
                          "3651581213_f81963d1dd_z.jpg"

                          ]



                        ]


                       )


if __name__ == '__main__':
    tst()