import cv2 as cv
import matplotlib.pyplot as plt 
from generalFunctions import output_figure

class FeatureMatcher:
    def __init__(self, comp_img, img, matcher, algorithm):
        self.comp_img = comp_img
        self.img = img
        self.matcher = matcher
        self.algorithm = algorithm
        self.kpt_1 = None
        self.dsc_1 = None
        self.kpt_2 = None
        self.dsc_2 = None
        self.matches = None
        self.good = []
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
    
    def set_matcher(self, matcher):
        self.matcher = matcher
        
    def extract_keypoints(self):
        if self.algorithm == 'ORB':
            extractor = cv.ORB_create()
        elif self.algorithm == 'BRISK':
            extractor = cv.BRISK_create()
        else:
            print("Not a valid algorithm name")

        # Detect keypoints
        self.kpt_1 = extractor.detect(self.comp_img, None)
        self.kpt_2 = extractor.detect(self.img, None)

        # Compute the descriptors
        self.kpt_1, self.dsc_1 = extractor.compute(self.comp_img, self.kpt_1)
        self.kpt_2, self.dsc_2 = extractor.compute(self.img, self.kpt_2)

    def match_keypoints(self):
        
        if self.matcher == 'KNN': 
            # BFMatcher with default params
            bf = cv.BFMatcher()
            
            self.matches = bf.knnMatch(self.dsc_1, self.dsc_2, k=2)
            
            # filtering the good matches by applying ratio test
            for m,n in self.matches:
                if m.distance < 0.75 * n.distance:
                    self.good.append([m])
            
            # flattening matches for homography
            self.matches  = [item for sublist in self.good for item in sublist]
            
        elif self.matcher == 'BRUTE':
            
            if self.algorithm == 'SIFT':
                bf = cv.BFMatcher()
            elif self.algorithm == 'ORB' or 'BRISK':
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            else:
                print("Not valid algorithm name.")
                return None

            # Match descriptors.
            self.matches = bf.match(self.dsc_1, self.dsc_2)
            # Sort them in the order of their distance.
            self.matches = sorted(self.matches, key = lambda x:x.distance)
        else:
            print("Not valid matcher name. Options: 'KNN or BRUTE")

    def draw_matches(self):
        plt.figure(dpi=300)
        if self.matcher == 'BRUTE':
            # Draw first 10 matches.
            img_matches = cv.drawMatches(self.comp_img, self.kpt_1, self.img, self.kpt_2, self.matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_keypoints1 = cv.drawKeypoints(self.comp_img, self.kpt_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_keypoints2 = cv.drawKeypoints(self.img, self.kpt_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        elif self.matcher == 'KNN':
            # cv2.drawMatchesKnn expects list of lists as matches.
            img_matches = cv.drawMatchesKnn(self.comp_img, self.kpt_1, self.img, self.kpt_2, self.good, None, flags=2) 
            img_keypoints1 = cv.drawKeypoints(self.comp_img, self.kpt_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_keypoints2 = cv.drawKeypoints(self.img, self.kpt_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        else:
            print("Not valid matcher name.")
        
        '''
        plt.imshow(img_matches)
        plt.title("Matches for " + str(self.matcher))
        plt.show()
        '''
        
        output_figure(img_matches, title="Matches for " + str(self.algorithm) + " and " + str(self.matcher))
        
    def compute(self):
        # Extract keypoints
        self.extract_keypoints()
        
        # Match keypoints
        self.match_keypoints()

        # Draw matches
        self.draw_matches()