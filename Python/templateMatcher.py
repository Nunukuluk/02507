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
    def __init__(self, templates, img, method, resolution):
        self.templates = templates
        self.img = img
        self.method = method
        self.result = cv.cvtColor(img.copy(),cv.COLOR_GRAY2RGB)
        self.resolution = resolution
        # Set dimensions of bb here and access depending on template

    def set_template(self, templates):
        self.templates = templates

    def set_result(self, result):
        self.result = result

    def get_center(self, x1, y1, x2, y2):
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)

    def template_matching(self, label):
        pred = []
        boxes_out = []
        # The smaller the threshold the stricter
        # Work on 0 3 6
        
        thresholds = [0.35, 0.3, 0.2, 0.25, 0.3, 0.1, 0.35] # - with threshold

        for t in self.templates:
            boxes = []
            temp_img = cv.cvtColor(self.img.copy(),cv.COLOR_GRAY2RGB)
            mtResult = cv.matchTemplate(self.img, t, self.method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(cv.cvtColor(self.result, cv.COLOR_RGB2GRAY))
            w, h = t.shape[::-1]
            if self.method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + w, top_left[1] + h)

            threshold = thresholds[label] 
            loc = np.where(mtResult <= threshold)
            for pt in zip(*loc[::-1]):
                # x, y, x + w, y + h
                boxes.append([int(pt[0] / self.resolution), int(pt[1] / self.resolution), int((pt[0] + w) / self.resolution), int((pt[1] + h) / self.resolution)])
            
            boxes = np.array(boxes)
            boxes = non_max_suppression_fast(boxes, 0.4)

            for b in boxes:
                # Append center position of box with label
                pred.append((label, self.get_center(b[0], b[1], b[2], b[3])))
                boxes_out.append((b[0], b[1], b[2], b[3]))

        return boxes_out, pred

    def plot_template_matching_results(self, boxes, true, pred):

        size = (int(self.result.shape[1] / self.resolution), int(self.result.shape[0] / self.resolution)) # Change resolution of image by percentage
        self.result = cv.resize(self.result, size, interpolation = cv.INTER_AREA)

        for i, b in enumerate(boxes):
            for j, subbox in enumerate(b):
                # Draw bounding boxes
                cv.rectangle(self.result, (subbox[0], subbox[1]), (subbox[2], subbox[3]), 1, 5)
                
                # Draw circle on image showing predicted center position in red
                cv.circle(self.result, pred[i][j][1], radius=10, color=(1,0,0), thickness=-1) 
                cv.putText(self.result, str(pred[i][j][0]), (subbox[0], subbox[3]), cv.FONT_HERSHEY_DUPLEX, 2, (1,0,0), thickness=2)

        for i in range(len(true)):
            # Draw circle on image showing true center position in blue
            w = 300
            h = 300
            center = (true[i][1][0], true[i][1][1])
            cv.rectangle(self.result, (center[0] - int(w/2), center[1] - int(h/2)), (center[0] + int(w/2), center[1] + int(h/2)), color=(1,1,0), thickness=5)
            cv.circle(self.result, true[i][1], radius=10, color=(1,1,0), thickness=-1)
            cv.putText(self.result, str(true[i][0]), (center[0] + int(w/2), center[1] + int(h/2)), cv.FONT_HERSHEY_DUPLEX, 2, (1,1,0), thickness=2)
                
        # Save figure 
        output_figure(self.result, colored_image=True)


