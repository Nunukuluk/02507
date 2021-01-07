import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from keypointExtraction import KeypointExtraction
from matching import Matching

# Load data
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):

        if filename == 'Components.bmp':
            comp_img = cv.imread(os.path.join(folder,filename))
        else:
            img = cv.imread(os.path.join(folder,filename))
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                images.append(img)
    return comp_img, images

folder = "FeatureMatchingDataset"
algorithm = "ORB"
matcher = "KNN"

comp_img, imgs = load_images_from_folder(folder)

# Prepare data
# Extract keypoints and descriptors # ORB 

keypointExtractor = KeypointExtraction(comp_img, imgs[0])
keypoints = keypointExtractor.keypointsORB()

print('after keypoints')

# Match 
matching = Matching([comp_img, imgs[0]], keypoints, matcher, algorithm)
matches = matching.match_keypoints()


matching.draw_matches(matches)
