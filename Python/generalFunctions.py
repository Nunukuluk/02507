import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv 
from datetime import datetime
import glob

# Load data
def load_images_from_folder(folder):
    images = []
    comp_img = None
    for filename in os.listdir(folder):

        if filename == 'Components.bmp':
            comp_img = cv.imread(os.path.join(folder,filename))
            comp_img = cv.cvtColor(comp_img, cv.COLOR_BGR2GRAY)
            # norm_img = np.zeros((comp_img.shape[0], comp_img.shape[1]))
            # comp_img = cv.normalize(comp_img,  norm_img, 0, 1, cv.NORM_MINMAX)
            comp_img = comp_img.astype('float32')
            comp_img /= 255.0
        else:
            img = cv.imread(os.path.join(folder,filename))
            if img is not None:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                # norm_img = np.zeros((img.shape[0], img.shape[1]))
                # img = cv.normalize(img, norm_img, 0, 1, cv.NORM_MINMAX)
                img = img.astype('float32')
                img /= 255.0
                images.append(img)
    return comp_img, images

def save_figure(*argv):  
    lim = len(argv)
    fig = plt.figure(figsize=(10, 10))
    for i, arg in enumerate(range(lim)):  
        ax = fig.add_subplot(1,lim,arg+1)
        #ax.imshow(argv[arg])
        ax.imshow(argv[arg], cmap='gray')
        dateTimeObj = datetime.now()
    plt.savefig('output/' + str(dateTimeObj) + "_" + str(i) + ".png")

def clear_output_folder():
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)