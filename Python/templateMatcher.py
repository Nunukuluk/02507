import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from generalFunctions import save_figure

class TemplateMatcher:
    def __init__(self, template, img, method):
        self.template  = template
        self.img = img
        self.method = method

    def template_matching(self):
        temp_img = self.img.copy()
        result = cv.matchTemplate(temp_img, self.template, self.method)
        print(result)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        w, h = self.template.shape[::-1]
        if self.method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        threshold = 0.3
        loc = np.where( result <= threshold)
        print("loc", loc)

        for pt in zip(*loc[::-1]):
            print("pt", pt)
            cv.rectangle(temp_img, pt, (pt[0] + w, pt[1] + h), 1, 2)

        cv.rectangle(temp_img,top_left, bottom_right, 1, 2)
        plt.subplot(121),plt.imshow(result,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(temp_img,cmap = 'gray')
        plt.title('Detected Point(s)'), plt.xticks([]), plt.yticks([])
        plt.suptitle("An image")
        save_figure(temp_img) # result, 
        save_figure(result)
        '''
        # All the 6 methods for comparison in a list
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                    'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        '''

