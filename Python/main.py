import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
from featureMatcher import FeatureMatcher
from templateMatcher import TemplateMatcher
from generalFunctions import *

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# clear output folder
clear_output_folder()

# folder paths
imgs_folder_path = "../FeatureMatchingDataset"
templates_folder_path = "../Templates"

# loading images from folders
comp_img, imgs = load_images_from_folder(imgs_folder_path)
_, templates = load_images_from_folder(templates_folder_path)

# Feature matching - ORB and KNN
'''
algorithm = "ORB"
matcher = "KNN"
featureMatcher = FeatureMatcher(comp_img, imgs[0], matcher, algorithm)
featureMatcher.compute()
'''
t = 2 # template picture no. (max = 5)
i = 38 # image no. (max = 38)

save_figure(templates[t], imgs[i])

# Template matching
templateMatcher = TemplateMatcher(templates[t], imgs[i], cv.TM_SQDIFF_NORMED)
templateMatcher.template_matching()
