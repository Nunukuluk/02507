import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv 
from datetime import datetime
import glob
import imutils
from pathlib import Path

# Load data
def load_images_from_folder(folder):
    images = []
    comp_img = None

    for filename in sorted(glob.glob(folder + '/**/*', recursive=True)):

        if filename == folder + '/Components.bmp':
            comp_img = cv.imread(os.path.join(filename), cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(os.path.join(filename), cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return comp_img, images

def pre_processing(images, p_size, threshold=False):
    processed = []
    for img in images:
        out = img.astype('float32') # to gray and float
        out /= 255.0 # normalize (0-1)
        size = (int(img.shape[1] * p_size), int(img.shape[0] * p_size)) # Change resolution of image by percentage
        out = cv.resize(out, size, interpolation = cv.INTER_AREA)
        if threshold:
            # out = cv.medianBlur(out,5)
            # Threshold image
            # out = cv.adaptiveThreshold((out).astype('uint8'), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 10)
            _, out = cv.threshold(out,0.20,1.0,cv.THRESH_BINARY) # 0.20 was good
            #output_figure(out, title='After thresh')
            # Median blur
            out = cv.medianBlur(out, 3) 
            #output_figure(out, title='After median')

            # Opening
            kernel = np.ones((3,3), np.uint8)
            out = cv.morphologyEx(out, cv.MORPH_OPEN, kernel)
            #output_figure(out, title='After opening')
            
            
            # Closing
            morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
            out = cv.morphologyEx(out, cv.MORPH_CLOSE, morph_kernel, iterations=3)
            #output_figure(out, title='After closing')

            '''
                        # Closing
            morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
            out = cv.morphologyEx(out, cv.MORPH_CLOSE, morph_kernel, iterations=3)
            #output_figure(out, title='After closing')

            # Erosion
            kernel = np.ones((5,5),np.uint8)
            out = cv.erode(out, kernel, iterations = 1)

            '''
            '''
            # Otsu's thresholding after Gaussian filtering
            out = cv.GaussianBlur(out,(3,3),0)
            output_figure(out)
            ret3, out = cv.threshold(np.uint8(out), 0.5, 1.0, cv.THRESH_BINARY+cv.THRESH_OTSU)
            output_figure(out)
            
            # Morphological gradient    
            kernel = np.ones((5,5),np.uint8)
            erosion = cv.erode(out, kernel, iterations = 1)
            dilation = cv.dilate(out, kernel, iterations = 1)
            out = dilation - erosion 
            output_figure(out)
            '''

        processed.append(out)
    return processed


def output_figure(*argv, colored_image=False, title=''):  
    lim = len(argv)
    fig = plt.figure(figsize=(10, 10))
    for i, arg in enumerate(range(lim)):  
        ax = fig.add_subplot(1,lim,arg+1)
        if colored_image:
            ax.imshow(argv[arg])
        else:
            ax.imshow(argv[arg], cmap='gray')
        plt.title(title, fontsize=40)
        dateTimeObj = datetime.now()
        plt.axis('off')
    plt.savefig('output/' + str(dateTimeObj) + "_" + str(i) + ".png")

def save_image(images, path):  
    for i, arg in enumerate(images):  
        cv.imwrite(path + str(i) + ".png", arg)

def clear_output_folder():
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

def rotate_image(image, angle, label):
    path = "../Templates/" + str(angle)
    temp_angle = angle
    if not os.path.exists(path):
        os.mkdir(path)  
    
    n_rotations = int(360/angle)
    rotated = []
    for i in range(n_rotations):
        result = imutils.rotate_bound(image, temp_angle)
        rotated.append(result)
        temp_angle += angle

    save_image(rotated, path=path + "/" + str(label) + '0_')

    return result