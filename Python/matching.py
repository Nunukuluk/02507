import cv2 as cv
import matplotlib.pyplot as plt 

class Matching:
    def __init__(self, images, keypoints, matcher, algorithm):
        self.comp_img = images[0]
        self.img = images[1]
        self.kpt_1 = keypoints[0]
        self.dsc_1 = keypoints[1]
        self.kpt_2 = keypoints[2]
        self.dsc_2 = keypoints[3]
        self.matcher = matcher
        self.algorithm = algorithm

    def match_keypoints(self):
        
        if self.matcher == 'KNN': 
            # BFMatcher with default params
            bf = cv.BFMatcher()
            
            matches = bf.knnMatch(self.dsc_1, self.dsc_2, k=2)
            
            # filtering the good matches by applying ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            
            self.good = good
            # flattening matches for homography
            matches  = [item for sublist in good for item in sublist]
            
        elif self.matcher == 'BRUTE':
            
            if self.algorithm == 'SIFT':
                bf = cv.BFMatcher()
            elif self.algorithm == 'ORB' or 'BRISK':
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            else:
                print("Not valid algorithm name.")
                return None

            # Match descriptors.
            matches = bf.match(self.dsc_1, self.dsc_2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
        else:
            print("Not valid matcher name. Options: 'KNN or BRUTE")

        return matches

    def draw_matches(self, matches):
            
        plt.figure(dpi=300)
        
        if self.matcher == 'BRUTE':
            # Draw first 10 matches.
            img_matches = cv.drawMatches(self.comp_img, self.kpt_1, self.img, self.kpt_2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_keypoints1 = cv.drawKeypoints(self.comp_img, self.kpt_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_keypoints2 = cv.drawKeypoints(self.img, self.kpt_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        elif self.matcher == 'KNN':
            # cv2.drawMatchesKnn expects list of lists as matches.
            img_matches = cv.drawMatchesKnn(self.comp_img, self.kpt_1, self.img, self.kpt_2, self.good, None, flags=2)
            print('after drawMatchesKnn')
            img_keypoints1 = cv.drawKeypoints(self.comp_img, self.kpt_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img_keypoints2 = cv.drawKeypoints(self.img, self.kpt_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print('after drawKeypoints')

        else:
            print("Not valid matcher name.")
            return None
        
        print(img_matches)
        plt.imshow(img_matches)
        plt.title("Matches for " + str(self.matcher))
        plt.show()