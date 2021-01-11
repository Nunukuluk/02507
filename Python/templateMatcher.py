import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from generalFunctions import save_image, output_figure
from non_maximum_suppression import non_max_suppression_fast
class TemplateMatcher:
    '''
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
                'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    '''
    def __init__(self, template, img, method):
        self.template = template
        self.img = img
        self.method = method
        self.result = img.copy()

    def set_template(self, template):
        self.template = template

    def get_center(self, x1, y1, x2, y2):
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)

    def template_matching(self, label):
        boxes = []
        positions = []
        # make structure with column for label and one for position
        # 2d array with label and (x,y)
        temp_img = self.img.copy()
        mtResult = cv.matchTemplate(self.img, self.template, self.method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(self.result)
        w, h = self.template.shape[::-1]
        if self.method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        threshold = 0.4
        loc = np.where( mtResult <= threshold)
        for pt in zip(*loc[::-1]):
            # x, y, x + w, y + h
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        
        boxes = np.array(boxes)
        boxes = non_max_suppression_fast(boxes, 0.4)
        for b in boxes:
            cv.rectangle(temp_img, (b[0], b[1]), (b[2], b[3]), 1, 5)
            cv.rectangle(self.result, (b[0], b[1]), (b[2], b[3]), 1, 5)
            positions.append((label, self.get_center(b[0], b[1], b[2], b[3])))
            print(self.img.shape, self.get_center(b[0], b[1], b[2], b[3]))
            cv.circle(self.result, self.get_center(b[0], b[1], b[2], b[3]), radius=10, color=1, thickness=-1) 
            cv.circle(self.result, (744, 1368), radius=10, color=1, thickness=-1) # ((744, 1368))
            cv.circle(self.result, (0, 0), radius=20, color=1, thickness=-1) 
            cv.circle(self.result, (int(self.img.shape[1]/2), int(self.img.shape[0]/2)), radius=20, color=1, thickness=-1) 
            #cv.circle(self.result, (self.img.shape[0], self.img.shape[1]), radius=20, color=1, thickness=-1) 
            print(positions)

        output_figure(self.template, self.result, temp_img) 
        return positions

'''
For a template (i.e. a specific label):
Find positions of that label in the image
Return the positions for that label

For an image
    For a template  
        Find position(s) in image for template

'''

