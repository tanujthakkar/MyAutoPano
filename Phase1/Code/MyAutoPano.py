#!/usr/env/bin python3

# Importing modules
from difflib import Match
from xml.dom import UserDataHandler
import cv2
from cv2 import RANSAC
from matplotlib.pyplot import axis
import numpy as np
import math
import os
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from Helper import *


class MyAutoPano():

    def __init__(self, ImageSetPath, NumFeatures, ResultPath, TestName, ImageSetHeight=None, ImageSetWidth=None):
        self.ImageCount = 0
        self.ImageSetPath = ImageSetPath
        self.ResultPath = ResultPath
        os.makedirs(self.ResultPath, exist_ok = True)
        self.NumFeatures = NumFeatures
        if(not ImageSetHeight and not ImageSetWidth):
            self.ImageSetResize = False
            self.ImageSetHeight = cv2.imread(ImageSetPath[0]).shape[0]
            self.ImageSetWidth = cv2.imread(ImageSetPath[0]).shape[1]
        else:
            self.ImageSetResize = True
            self.ImageSetHeight = ImageSetHeight
            self.ImageSetWidth = ImageSetWidth
        # print(ImageSetHeight, ImageSetWidth)
        self.ImageSet = list()
        self.ImageSetGray = list()
        self.Inliers = np.empty([0, 0, 1])
        self.Homography = np.empty([0, 1, 3, 3])
        self.BlendedImage = None
        self.ImageSetRefId = None
        self.TestName = TestName

        # Toggles
        self.Visualize = False

    def createImageSet(self):
        if(self.ImageSetResize):
            [self.ImageSet.append(cv2.resize(cv2.imread(self.ImageSetPath[img]), None, fx=self.ImageSetHeight, fy=self.ImageSetWidth, interpolation=cv2.INTER_CUBIC)) for img in range(len(self.ImageSetPath))] # Reading images
        else:
            [self.ImageSet.append(cv2.imread(self.ImageSetPath[img])) for img in range(len(self.ImageSetPath))] # Reading images
        [self.ImageSetGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.ImageSet] # Converting images to grayscale
        self.ImageSet = np.array(self.ImageSet)
        self.ImageSetGray = np.float32(np.array(self.ImageSetGray))
        self.ImageSetRefId = int(len(self.ImageSet)/2) # Setting a reference to the anchor image

    def computeHarrisCorners(self, Image, Visualize):
        print("Computing Harris Corners...")

        ImageGray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        CornerScore = cv2.cornerHarris(ImageGray, 2, 3, 0.00001) # Computing corner probability using Harris corners
        # CornerScore = cv2.normalize(CornerScore, None, -1.0, 1.0, cv2.NORM_MINMAX) # Normalizing
        CornerScore[CornerScore<0.001*CornerScore.max()] = 0
        CornerScore = cv2.dilate(CornerScore, None) # Dilating to mark corners
        HarrisCorners = np.copy(Image)
        HarrisCorners[CornerScore>0.001*CornerScore.max()]=[0,0,255] # Marking corners in RGB image
        if(Visualize):
            cv2.imshow("Harris Corners", HarrisCorners)
            cv2.imshow("Corner Score", np.float32(CornerScore))
            cv2.imwrite(self.ResultPath + self.TestName + '_Harris_' + str(self.ImageCount) + '.png', HarrisCorners)
            cv2.waitKey(3)

        return CornerScore, HarrisCorners

    def computeShiTomasiCorners(self, Image, Visualize):
        print("Computing Shi-Tomasi Corners...")

        ImageGray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(ImageGray, self.NumFeatures, 0.01, 3) # Computing corners using the Shi-Tomasi method
        corners = np.int0(corners)
        print("ShiTomasiCorners: %d"%(len(corners)))

        ShiTomasiCorners = np.zeros(Image.shape[0:2])
        ImageSetShiTomasiCorners = np.copy(Image)
        for corner in corners: # Marking corners in RGB image
            x,y = corner.ravel()
            ShiTomasiCorners[y,x] = 255
            cv2.circle(ImageSetShiTomasiCorners,(x,y),2,(0,0,255),-1)

        if(Visualize):
            cv2.imshow("Shi-Tomasi Corners", ImageSetShiTomasiCorners)
            cv2.imshow("Corners", ShiTomasiCorners)
            cv2.imwrite(self.ResultPath + self.TestName + '_Shi-Tomasi_' + str(self.ImageCount) + '.png', ImageSetShiTomasiCorners)
            cv2.waitKey(3)

        return ShiTomasiCorners, ImageSetShiTomasiCorners

    def ANMS(self, Image, ImageCorners, Visualize):
        print("Applying ANMS...")

        ANMSCorners = list()
        local_maximas = peak_local_max(ImageCorners, min_distance=1)
        local_maximas = np.int0(local_maximas)
        print("Local Maximas: %d"%len(local_maximas))

        if(self.NumFeatures > len(local_maximas)):
            self.NumFeatures = len(local_maximas)

        r = [np.Infinity for i in range(len(local_maximas))]
        ED = 0

        for i in tqdm(range(len(local_maximas))):
            for j in range(len(local_maximas)):
                if(ImageCorners[local_maximas[j,0],local_maximas[j,1]] > ImageCorners[local_maximas[i,0],local_maximas[i,1]]):
                    ED = math.sqrt((local_maximas[j,0] - local_maximas[i,0])**2 + (local_maximas[j,1] - local_maximas[i,1])**2)
                if(ED < r[i]):
                    r[i] = ED
            ANMSCorners.append([r[i], local_maximas[i,0], local_maximas[i,1]])

        ANMSCorners = sorted(ANMSCorners, reverse=True)
        ANMSCorners = np.array(ANMSCorners[:self.NumFeatures])
        print("ANMS Corners: %d"%len(ANMSCorners))

        ImageSetLocalMaxima = np.copy(Image)
        ImageSetANMS = np.copy(Image)

        for local_maxima in local_maximas: # Marking corners in RGB image
                y,x = local_maxima.ravel()
                cv2.circle(ImageSetLocalMaxima,(x,y),2,(0,255,0),-1)

        for i in range(self.NumFeatures): # Marking corners in RGB image
            cv2.circle(ImageSetANMS,(int(ANMSCorners[i][2]),int(ANMSCorners[i][1])),2,(0,0,255),-1)
        
        if(Visualize):
            cv2.imshow("Local Max", ImageSetLocalMaxima)
            cv2.imshow("ANMS", ImageSetANMS)
            cv2.imwrite(self.ResultPath + self.TestName + '_ANMS_' + str(self.ImageCount) + '.png', ImageSetANMS)
            cv2.waitKey(3)

        return ANMSCorners, ImageSetLocalMaxima, ImageSetANMS

    def featureDescriptor(self, Image, key_points, Visualize):
        print("Retrieving feature patches...")

        ImageGray = np.float32(cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY))
        patch_size = 40
        features = list()
        for point in range(len(key_points)):
            patch = np.uint8(np.array(neighbors(ImageGray, 20, int(key_points[point][1]), int(key_points[point][2]))))
            patch_gauss = cv2.resize(cv2.GaussianBlur(patch, (5,5), 0), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
            patch_gauss = (patch_gauss - patch_gauss.mean())/(patch_gauss.std()+1e-20)
            features.append(patch_gauss.flatten())
            if(Visualize):
                temp = cv2.circle(np.copy(Image),(int(key_points[point][2]), int(key_points[point][1])),2,(0,0,255),-1)
                cv2.imshow("Feature", temp)
                cv2.imshow("Patch", patch)
                cv2.imshow("Patch gauss", patch_gauss)
                # cv2.waitKey(3)

        features = np.array(features)

        return features

    def featureMatching(self, Image0, Image1, Features0, Features1, ANMSCorners0, ANMSCorners1, Visualize):
        print("Matching features...")
            
        SSDs = list()
        matches = list()
        if(len(Features0) > len(Features1)):
            N = len(Features1)
        else:
            N = len(Features0)
        features = np.arange(N).tolist()

        if(Visualize):
            if(Image0.shape != Image1.shape):
                temp_shape = np.vstack((Image0.shape, Image1.shape)).max(axis=0)
                Image0_ = np.uint8(np.empty(temp_shape))
                Image1_ = np.uint8(np.empty(temp_shape))
                Image0_[0:Image0.shape[0],0:Image0.shape[1]] = Image0
                Image1_[0:Image1.shape[0],0:Image1.shape[1]] = Image1
                temp = np.hstack((Image0_, Image1_))
            else:
                temp = np.hstack((Image0, Image1))
        
        for i in tqdm(range(N)):
            SSDs.clear()
            for j in features:
                SSDs.append([sum((Features0[i] - Features1[j])**2), ANMSCorners1[j][1], ANMSCorners1[j][2]])

            SSDs = sorted(SSDs)

            # if((SSDs[0][0]/SSDs[1][0]) < 0.95):
            #     continue
            
            matches.append([ANMSCorners0[i][1], ANMSCorners0[i][2], SSDs[0][1], SSDs[0][2]])

            temp = cv2.circle(temp,(int(ANMSCorners0[i][2]), int(ANMSCorners0[i][1])),2,(0,0,255),-1)
            temp = cv2.circle(temp,(int(SSDs[0][2])+Image1.shape[1], int(SSDs[0][1])),2,(0,0,255),-1)
            temp = cv2.line(temp, (int(ANMSCorners0[i][2]), int(ANMSCorners0[i][1])), (int(SSDs[0][2])+Image1.shape[1], int(SSDs[0][1])), (0,255,0), 1)
            if(Visualize):
                cv2.imshow("Matches", temp)
                cv2.imwrite(self.ResultPath + self.TestName + '_Matches_' + str(self.ImageCount) + '.png', temp)

        if(Visualize):
            cv2.waitKey(3)

        print("Matches: %d"%len(matches))

        matches = np.array(matches)

        return matches, temp

    def RANSAC(self, Matches, Image0, Image1, iterations, threshold, Visualize):
        print("Performing RANSAC...")

        max_inliers = 0
        best_H = None
        Inliers = list()
        features = np.arange(len(Matches)).tolist()

        for i in tqdm(range(iterations)):

            feature_pairs = np.random.choice(features, 4, replace=False)
            p1 = list()
            p2 = list()
            for j in range(len(feature_pairs)):
                p1.append([Matches[feature_pairs[j]][1], Matches[feature_pairs[j]][0]])
                p2.append([Matches[feature_pairs[j]][3], Matches[feature_pairs[j]][2]])

            H = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p2))
            Hp1 = np.dot(H, np.vstack((Matches[:,1], Matches[:,0], np.ones([1,len(Matches)]))))
            Hp1 = np.array(Hp1/(Hp1[2]+1e-20)).transpose()
            Hp1 = np.delete(Hp1, 2, 1)
            p2_ = list()
            [p2_.append([Matches[x][3], Matches[x][2]]) for x in range(len(Matches))]
            p2_ = np.array(p2_)

            SSD = list()
            [SSD.append(sum((p2_[x] - Hp1[x])**2)) for x in range(len(Matches))]

            SSD = np.array(SSD)
            SSD[SSD <= threshold] = 1
            SSD[SSD > threshold] = 0

            inliers = np.sum(SSD)

            if(inliers > max_inliers):
                max_inliers = inliers
                Inliers = np.where(SSD == 1)
                best_H = H

        p1.clear()
        p2.clear()
        for i in Inliers[0]:
            p1.append([Matches[i][1], Matches[i][0]])
            p2.append([Matches[i][3], Matches[i][2]])

        H, _ = cv2.findHomography(np.float32(p1), np.float32(p2), cv2.RANSAC, 1)

        print("Inliers: %d"%max_inliers)

        if(Image0.shape != Image1.shape):
            temp_shape = np.vstack((Image0.shape, Image1.shape)).max(axis=0)
            Image0_ = np.uint8(np.empty(temp_shape))
            Image1_ = np.uint8(np.empty(temp_shape))
            Image0_[0:Image0.shape[0],0:Image0.shape[1]] = Image0
            Image1_[0:Image1.shape[0],0:Image1.shape[1]] = Image1
            temp = np.hstack((Image0_, Image1_))
        else:
            print('test')
            temp = np.hstack((Image0, Image1))
        for i in Inliers[0]:
            temp = cv2.circle(temp,(int(Matches[i][1]), int(Matches[i][0])),2,(0,0,255),-1)
            temp = cv2.circle(temp,(int(Matches[i][3])+Image1.shape[1], int(Matches[i][2])),2,(0,0,255),-1)
            temp = cv2.line(temp, (int(Matches[i][1]), int(Matches[i][0])), (int(Matches[i][3])+Image1.shape[1], int(Matches[i][2])), (0,255,0), 1)
        if(Visualize):
            cv2.imshow("RANSAC", temp)
            cv2.waitKey(3)

        self.Homography = np.insert(self.Homography, len(self.Homography), np.array([best_H]), axis=0)

        if(H is None):
            H = best_H

        return H, temp

    def mean_blend(self, img1, img2):
        assert(img1.shape == img2.shape)
        locs1 = np.where(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) != 0)
        blended1 = np.copy(img2)
        blended1[locs1[0], locs1[1]] = 0
        locs2 = np.where(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) != 0)
        blended2 = np.copy(img1)
        blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
        blended = cv2.addWeighted(blended1, 0, blended2, 1.0, 0)

        return blended

    def stitchImages(self, Image0, Image1, H, Visualize):
        print("Blending Images...")

        h0, w0 = Image0.shape[:2]
        h1, w1 = Image1.shape[:2]

        c0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2) # Points on Image 1
        c1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2) # Points on Image 2

        # print("Homography Matrix", H)

        c0_ = cv2.perspectiveTransform(c0, H) # Points of Image 1 transformed

        corners = np.concatenate((c0_, c1), axis = 0).reshape(8,2)

        x_min, y_min = np.int0(np.min(corners, axis = 0))
        x_max, y_max = np.int0(np.max(corners, axis = 0))

        H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

        Image0_Warped = cv2.warpPerspective(Image0, np.dot(H_translate, H), (x_max-x_min, y_max-y_min))

        ImageStitched = np.copy(Image0_Warped)
        # ImageStitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = Image1

        idx = np.s_[-y_min:-y_min+h1, -x_min: -x_min+w1]
        ImageStitched[idx] = self.mean_blend(ImageStitched[idx], Image1)

        # idx = np.where(Image1 == [0,0,0])
        # y = idx[0] + -y_min 
        # x = idx[1] + -x_min 
        # ImageStitched[y,x] = Image0_Warped[y,x]

        if(Visualize):
            # cv2.imshow("Image0_", Image0_Warped)
            cv2.imshow("Stiched", ImageStitched)
            # cv2.waitKey(3)

        return ImageStitched
    
    def saveResults(self, ImageCount, Corners_0, Corners_1, ANMS_0, ANMS_1, Matches, RANSAC_, Stich):
        cv2.imwrite(self.ResultPath + self.TestName + '_Corners_0_' + str(ImageCount) + '.png', Corners_0)
        cv2.imwrite(self.ResultPath + self.TestName + '_Corners_1_' + str(ImageCount) + '.png', Corners_1)
        cv2.imwrite(self.ResultPath + self.TestName + '_ANMS_0_' + str(ImageCount) + '.png', ANMS_0)
        cv2.imwrite(self.ResultPath + self.TestName + '_ANMS_1_' + str(ImageCount) + '.png', ANMS_1)
        cv2.imwrite(self.ResultPath + self.TestName + '_Matches_' + str(ImageCount) + '.png', Matches)
        cv2.imwrite(self.ResultPath + self.TestName + '_RANSAC_' + str(ImageCount) + '.png', RANSAC_)
        cv2.imwrite(self.ResultPath + self.TestName + '_Stich_' + str(ImageCount) + '.png', Stich)

    def generatePanorama(self, Visualize):
        print("Generating Panorama...")

        self.createImageSet()
        ImageSet = [x for x in self.ImageSet]
        if(len(ImageSet)%2 != 0):
            ImageSet = ImageSet[:len(ImageSet)//2+1]
        else:
            ImageSet = ImageSet[:len(ImageSet)//2]

        PanoHalves = list()

        half = self.ImageSetRefId//2
        # if(half == 0):
            # half = 1
            
        for i in range(half+1):

            for img in range(len(ImageSet)-1):
                print("Stitching Frames %d & %d"%(img*(i+1), img*(i+1)+1))

                ShiTomasiCorners0, Corners_0 = self.computeShiTomasiCorners(ImageSet[img], True)
                ANMSCorners0, _, ANMS_0 = self.ANMS(ImageSet[img], ShiTomasiCorners0, True)
                Features0 = self.featureDescriptor(ImageSet[img], ANMSCorners0, False)

                ShiTomasiCorners1, Corners_1 = self.computeShiTomasiCorners(ImageSet[img+1], True)
                ANMSCorners1, _, ANMS_1 = self.ANMS(ImageSet[img+1], ShiTomasiCorners1, True)
                Features1 = self.featureDescriptor(ImageSet[img+1], ANMSCorners1, False)

                Matches, Matches_ = self.featureMatching(ImageSet[img], ImageSet[img+1], Features0, Features1, ANMSCorners0, ANMSCorners1, True)
                H, RANSAC_ = self.RANSAC(Matches, ImageSet[img], ImageSet[img+1], 1000, 5, True)
                if(H is not None):
                    I = self.stitchImages(ImageSet[img], ImageSet[img+1], H, True)
                else:
                    print("Not enough overlap, skipping image...")
                    continue

                self.saveResults(self.ImageCount, Corners_0, Corners_1, ANMS_0, ANMS_1, Matches_, RANSAC_, I)
                # cv2.imwrite(self.ResultPath + self.TestName + '_Stich_' + str(self.ImageCount) + '.png', I)
                self.ImageCount += 1
                ImageSet.append(I)
            
            ImageSet = ImageSet[-img-1:]
        
        PanoHalves.append(I)
        ImageSet.clear()
        ImageSet = [x for x in self.ImageSet]
        ImageSet.reverse()
        if(len(ImageSet)%2 != 0):
            ImageSet = ImageSet[:len(ImageSet)//2+1]
        else:
            ImageSet = ImageSet[:len(ImageSet)//2]

        for i in range(half+1):

            for img in range(len(ImageSet)-1):
                print("Stitching Frames %d & %d"%(img*(i+1), img*(i+1)+1))

                ShiTomasiCorners0, Corners_0 = self.computeShiTomasiCorners(ImageSet[img], True)
                ANMSCorners0, _, ANMS_0 = self.ANMS(ImageSet[img], ShiTomasiCorners0, True)
                Features0 = self.featureDescriptor(ImageSet[img], ANMSCorners0, False)

                ShiTomasiCorners1, Corners_1 = self.computeShiTomasiCorners(ImageSet[img+1], True)
                ANMSCorners1, _, ANMS_1 = self.ANMS(ImageSet[img+1], ShiTomasiCorners1, True)
                Features1 = self.featureDescriptor(ImageSet[img+1], ANMSCorners1, False)

                Matches, Matches_ = self.featureMatching(ImageSet[img], ImageSet[img+1], Features0, Features1, ANMSCorners0, ANMSCorners1, True)
                H, RANSAC_ = self.RANSAC(Matches, ImageSet[img], ImageSet[img+1], 1000, 5, True)
                if(H is not None):
                    I = self.stitchImages(ImageSet[img], ImageSet[img+1], H, True)
                else:
                    print("Not enough overlap, skipping image...")
                    continue

                self.saveResults(self.ImageCount, Corners_0, Corners_1, ANMS_0, ANMS_1, Matches_, RANSAC_, I)
                # cv2.imwrite(self.ResultPath + self.TestName + '_Stich_' + str(self.ImageCount) + '.png', I)
                self.ImageCount += 1
                ImageSet.append(I)
            
            ImageSet = ImageSet[-img-1:]

        PanoHalves.append(I)

        print("Generating final panorama...")
        PanoFirstHalf = PanoHalves[0]
        PanoSecondHalf = PanoHalves[1]

        ShiTomasiCorners0, Corners_0 = self.computeShiTomasiCorners(PanoFirstHalf, True)
        ANMSCorners0, _, ANMS_0 = self.ANMS(PanoFirstHalf, ShiTomasiCorners0, True)
        Features0 = self.featureDescriptor(PanoFirstHalf, ANMSCorners0, False)

        # HarrisCorners1 = self.computeHarrisCorners(self.ImageSet[img+1], True)
        ShiTomasiCorners1, Corners_0 = self.computeShiTomasiCorners(PanoSecondHalf, True)
        ANMSCorners1, _, ANMS_1 = self.ANMS(PanoSecondHalf, ShiTomasiCorners1, True)
        Features1 = self.featureDescriptor(PanoSecondHalf, ANMSCorners1, False)

        Matches, Matches_ = self.featureMatching(PanoFirstHalf, PanoSecondHalf, Features0, Features1, ANMSCorners0, ANMSCorners1, True)
        H, RANSAC_ = self.RANSAC(Matches, PanoFirstHalf, PanoSecondHalf, 2000, 10, True)
        I = self.stitchImages(PanoFirstHalf, PanoSecondHalf, H, True)

        self.saveResults(self.ImageCount, Corners_0, Corners_1, ANMS_0, ANMS_1, Matches_, RANSAC_, I)