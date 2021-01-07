import cv2 as cv

class KeypointExtraction:
    def __init__(self, comp_img, img):
        self.comp_img = comp_img
        self.img = img
    
    def keypointsORB(self):
        # Initiate ORB detector
        orb = cv.ORB_create()

        # Scaled images
        kp_1 = orb.detect(self.comp_img, None)
        kp_2 = orb.detect(self.img, None)

        # compute the descriptors with ORB
        kp_1, dsc_1 = orb.compute(self.comp_img, kp_1)
        kp_2, dsc_2 = orb.compute(self.img, kp_2)

        return [kp_1, dsc_1, kp_2, dsc_2]