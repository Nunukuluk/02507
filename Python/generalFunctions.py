import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv 
from datetime import datetime
import glob
import imutils

# Load data
def load_images_from_folder(folder):
    images = []
    comp_img = None
    # for filename in os.listdir(folder):
    for filename in sorted(glob.glob(folder + '/*')):
        if filename == folder + '/Components.bmp':
            comp_img = cv.imread(os.path.join(filename), cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(os.path.join(filename), cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return comp_img, images

def pre_processing(img, p_size, threshold=False):
    out = img.astype('float32') # to gray and float
    out /= 255.0 # normalize (0-1)
    size = (int(img.shape[1] * p_size), int(img.shape[0] * p_size)) # Change resolution of image by percentage
    out = cv.resize(out, size, interpolation = cv.INTER_AREA)
    
    if threshold:
        _, out = cv.threshold(out,0.3,1.0,cv.THRESH_BINARY)
        kernel = np.ones((5,5),np.float32)/25
        out = cv.filter2D(out,-1,kernel)
    
    return out


def output_figure(*argv):  
    lim = len(argv)
    fig = plt.figure(figsize=(10, 10))
    for i, arg in enumerate(range(lim)):  
        ax = fig.add_subplot(1,lim,arg+1)
        ax.imshow(argv[arg], cmap='gray')
        dateTimeObj = datetime.now()
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

    save_image(rotated, path=path + "/" + str(label) + '_')

    return result