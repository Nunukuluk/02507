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
from evaluation import Evaluation
import time
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# clear output folder
clear_output_folder()

# folder paths
imgs_folder_path = "../FeatureMatchingDataset"
templates_folder_path = "../Templates/45/"

# loading images from folders
comp_img, imgs = load_images_from_folder(imgs_folder_path)
templates = []
for i in range(0,6):
    _, images = load_images_from_folder(templates_folder_path + str(i))
    templates.append(images)

# Load labels and positions
labels = pd.read_csv('../Labels.txt', delimiter=' ', na_filter= False)
for i in range(0, 30):
    labels[str(i)] = labels[str(i)].map(lambda x: eval(x) if x != '' else x)

# Save for rotation
# _, new_template = load_images_from_folder("../Templates/part_00/")
# rotate_image(new_template[0], 45, 0)

idxs_images = np.array(np.arange(0,39)) #[36] #, 4, 6, 7, 8, 11, 36, 31] 12-2 bolts

template_matching = False
feature_matching = True

# ----------- Template matching ----------- #
if template_matching:
    res = 0.55
    threshold = False

    level_1 = np.arange(0, 12) # Images with only a single part for simple detection. 2-13 : 0-11
    level_2 = np.arange(12, 18) # Images with two similar parts. 14-19 : 12-17
    level_3 = np.arange(18, 22) # Images with 3 different parts. 20-23 : 18-21
    level_4 = np.concatenate((np.arange(22, 27), np.arange(33, 37), [38])) # Images with all 6 parts nicely seperated 24-28,35-38,40 : 22-26
    level_5 = np.concatenate((np.arange(27, 33), [37])) # Images with all 6 parts close together 29-34
    #level_5 = np.array(np.arange(32, 39)) # 1 of each parts (total of 6 parts in each image)

    accuracies, distances, runtimes, misclassified, missing, parts = [], [], [], [], [], []

    idxs_images = [1]#level_1# np.concatenate((level_1, level_2, level_3, level_4, level_5))
    level = 11

    evaluater = Evaluation()

    for i, indx in enumerate(idxs_images):

        predictions = []
        boxes = []
        true = [i for i in labels.iloc[indx+1][1:].tolist() if i] 
        start_time = time.time()
        img = imgs[indx]
        resized = pre_processing([img], res, threshold=threshold)

        for i_t, t in enumerate(templates):
            resized_templates = pre_processing(t, res, threshold=threshold)

            templateMatcher = TemplateMatcher(resized_templates, resized[0], cv.TM_SQDIFF_NORMED, res)
            
            box, pred = templateMatcher.template_matching(i_t)
            predictions.append(pred)
            boxes.append(box)
        

        # Evaluate runtime for each image after all used templates
        runtimes.append(time.time() - start_time)

        # templateMatcher.set_result(img.astype('float32') / 255.0) # Draw on original image
        templateMatcher.plot_template_matching_results(boxes, true, predictions)
        
        # Evaluate accuracy
        acc, dist = evaluater.accuracy(true, predictions, False)
        num_misclassified, num_missing, num_parts = evaluater.compute_confusion_matrix()
        accuracies.append(acc)
        distances.append(dist)
        misclassified.append(num_misclassified)
        missing.append(num_missing)
        parts.append(num_parts)
    
    mean_accuracy = sum(accuracies)/len(idxs_images)
    mean_runtime = sum(runtimes)/len(runtimes)
    mean_distance = sum(distances)/len(distances)
    print("\n\n------- RESULTS -------")
    print("Mean accuracy: %.2f percent" % (mean_accuracy * 100))
    print("Mean runtime: %.2f seconds" % mean_runtime)
    print("Mean distance: {} pixels".format(int(mean_distance)))
    print("Misclassification rate: {}".format(sum(misclassified)), "and %.2f percent" % ((sum(misclassified)/sum(parts))*100))
    print("Missing:  {}".format(sum(missing)), "and %.2f percent" % ((sum(missing)/sum(parts))*100))
    cm = evaluater.get_confusion_matrix()
    print("Confusion matrix: \n" , cm)
    
    df_cm = pd.DataFrame(cm, columns=np.arange(6), index=np.arange(6))
    title = 'Confusion Matrix for Category ' + str(level)
    path = 'plots/cm_' + str(level)
    evaluater.plot_confusion_matrix(cm, title, path)

# ----------- Feature matching ----------- #
if feature_matching:
    img = imgs[36]
    matcher = 'BRUTE'
    algorithm = 'ORB'

    for i_t, t in enumerate(templates):
        featureMatcher = FeatureMatcher(t[0], img, matcher, algorithm)
        featureMatcher.compute()


