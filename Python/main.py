import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
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
templates_folder_path = "../Templates"

# loading images from folders
comp_img, imgs = load_images_from_folder(imgs_folder_path)
_, templates = load_images_from_folder(templates_folder_path)


# t = template picture no. (max = 5), i = image no. (max = 38)
t = 0
i = 11
angle = 45

output_figure(templates[t], imgs[i])

# Preprocessing
res = 0.5
# thresholded = pre_processing(imgs[i], res, threshold=True)
# thresholded_template = pre_processing(templates[t], res, threshold=True)
resized = pre_processing(imgs[i], res)
resized_template = pre_processing(templates[t], res)
output_figure(resized_template, resized)

#rotate_image(templates[t], angle, t)
_, rotated_templates = load_images_from_folder(templates_folder_path + "/" + str(angle))
for i in range(len(rotated_templates)):
    rotated_templates[i] = pre_processing(rotated_templates[i], res)

# Template matching
templateMatcher = TemplateMatcher(rotated_templates, resized, cv.TM_SQDIFF_NORMED)
templateMatcher.template_matching()

'''
# Feature matching - ORB and KNN
algorithm = "ORB"
matcher = "KNN"
featureMatcher = FeatureMatcher(comp_img, imgs[0], matcher, algorithm)
featureMatcher.compute()
'''