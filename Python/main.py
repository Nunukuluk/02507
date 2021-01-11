import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
import pandas as pd
import math
from featureMatcher import FeatureMatcher
from templateMatcher import TemplateMatcher
from generalFunctions import *


warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


'''
- Preprocessing: lower resolution, binarize
- Template Matching: with different rotations, get center pixel
'''

# clear output folder
clear_output_folder()

# folder paths
imgs_folder_path = "../FeatureMatchingDataset"
templates_folder_path = "../Templates/45/0"

# loading images from folders
comp_img, imgs = load_images_from_folder(imgs_folder_path)
_, templates = load_images_from_folder(templates_folder_path)

# Load labels and positions
labels = pd.read_csv('../Labels.txt', delimiter=' ', na_filter= False)
for i in range(0, 30):
    labels[str(i)] = labels[str(i)].map(lambda x: eval(x) if x != '' else x)



# t = template picture no. (max = 5), i = image no. (max = 38)
t = 3 # screws are 0 and 3
i = 11
res = 1

# idxs_images = [0, 4, 6, 7, 8, 11, 36, 31]
idxs_images = [36]

for i, indx in enumerate(idxs_images):
    img = imgs[indx]
    resized = pre_processing(img, res)
    for i_t, t in enumerate(templates):
        print("Image no. " + str(indx) + ". Template no. " + str(i_t))
        resized_template = pre_processing(t, res)
        if i_t == 0:
            templateMatcher = TemplateMatcher(resized_template, resized, cv.TM_SQDIFF_NORMED)
        else:
            templateMatcher.set_template(resized_template)
            
        positions = templateMatcher.template_matching(i_t)
        
    print(labels.iloc[indx+1])


# Preprocessing 
'''
# Template matching
templateMatcher = TemplateMatcher(rotated_templates, resized, cv.TM_SQDIFF_NORMED)
templateMatcher.template_matching()


# Feature matching - ORB and KNN
algorithm = "ORB"
matcher = "KNN"
featureMatcher = FeatureMatcher(comp_img, imgs[0], matcher, algorithm)
featureMatcher.compute()
'''

'''
# To do
- ORB looked into it but left it hanging. Maybe they have some ideas of how it could work?
- rotations for other screw
- compute template matching for all images with single type
- maybe one image with all types (seperated)
- one image with all types (close together)
- for 1 template, show the parts that each rotation finds 

Considerations:
- calculate center pixel of each object
- 
'''