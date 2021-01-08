import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from generalFunctions import save_image, output_figure
from non_maximum_suppression import non_max_suppression_fast
class TemplateMatcher:
    def __init__(self, templates, img, method):
        self.templates = templates
        self.img = img
        self.method = method

    def template_matching(self):
        result = self.img.copy()
        boxes = []
        for i, t in enumerate(self.templates):
            temp_img = self.img.copy()
            mtResult = cv.matchTemplate(self.img, t, self.method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            w, h = t.shape[::-1]
            if self.method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + w, top_left[1] + h)

            threshold = 0.5
            loc = np.where( mtResult <= threshold)
            for pt in zip(*loc[::-1]):
                # x, y, x+w, y+h
                boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
                # calculate iou against prev
                # if iou passes threshold
                #cv.rectangle(temp_img, pt, (pt[0] + w, pt[1] + h), 1, 2)
                #cv.rectangle(result, pt, (pt[0] + w, pt[1] + h), 1, 2)
            
            # cv.rectangle(temp_img,top_left, bottom_right, 1, 2)

        boxes = np.array(boxes)
        print("box before:", len(boxes))
        boxes = non_max_suppression_fast(boxes, 0.4)
        print("box after:", len(boxes))
        for b in boxes:    
            #cv.rectangle(temp_img, (boxes[0], boxes[1]), (pt[0] + w, pt[1] + h), 1, 2)
            cv.rectangle(result, (b[0], b[1]), (b[2], b[3]), 1, 2)

        output_figure(t, result, temp_img) # result, 
        '''
        # All the 6 methods for comparison in a list
        methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                    'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        '''

