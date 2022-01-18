#!/usr/env/bin python3

# Importing modules
import cv2
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

	def __init__(self, ImageSetPath, NumFeatures, ResultPath, ImageSetHeight=None, ImageSetWidth=None):
		self.ImageCount = 0
		self.ImageSetPath = ImageSetPath
		self.ResultPath = ResultPath
		# print(self.ImageSetPath)
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
		self.ImageSetHarrisCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ImageSetShiTomasiCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.HarrisCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth]))
		self.ShiTomasiCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth]))
		self.ImageSetLocalMaxima = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ImageSetANMS = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ANMSCorners = np.empty([0, self.NumFeatures, 3])
		self.Features = np.empty([0, self.NumFeatures, 64])
		self.Matches = np.empty([0, self.NumFeatures, 4])
		self.Inliers = np.empty([0, 0, 1])
		self.Homography = np.empty([0, 1, 3, 3])
		self.BlendedImage = None
		self.ImageSetRefId = None

		# Toggles
		self.Visualize = False

	def createImageSet(self):
		# cv2.resize(cv2.GaussianBlur(patch, (5,5), 0), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
		if(self.ImageSetResize):
			[self.ImageSet.append(cv2.resize(cv2.imread(self.ImageSetPath[img]), (self.ImageSetWidth, self.ImageSetHeight), interpolation=cv2.INTER_CUBIC)) for img in range(len(self.ImageSetPath))] # Reading images
		else:
			[self.ImageSet.append(cv2.imread(self.ImageSetPath[img])) for img in range(len(self.ImageSetPath))] # Reading images
		[self.ImageSetGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.ImageSet] # Converting images to grayscale
		self.ImageSet = np.array(self.ImageSet)
		self.ImageSetGray = np.float32(np.array(self.ImageSetGray))
		self.ImageSetRefId = int(len(self.ImageSet)/2) # Setting a reference to the anchor image

	def computeHarrisCorners(self, Image, Visualize):
		print("Computing Harris Corners...")
		# self.ImageSetHarrisCorners = np.copy(self.ImageSet)
		# for i in range(len(self.ImageSetGray)):

		ImageGray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
		CornerScore = cv2.cornerHarris(ImageGray, 2, 3, 0.00001) # Computing corner probability using Harris corners
		# CornerScore = cv2.normalize(CornerScore, None, -1.0, 1.0, cv2.NORM_MINMAX) # Normalizing
		CornerScore[CornerScore<0.001*CornerScore.max()] = 0
		CornerScore = cv2.dilate(CornerScore, None) # Dilating to mark corners
		HarrisCorners = np.copy(Image)
		HarrisCorners[CornerScore>0.001*CornerScore.max()]=[0,0,255] # Marking corners in RGB image
		# self.CornerScore.append(img)
		if(Visualize):
			# cv2.imshow("", img)
			cv2.imshow("Harris Corners", HarrisCorners)
			cv2.imshow("Corner Score", np.float32(CornerScore))
			cv2.waitKey(0)
		# self.CornerScore = np.float32(np.array(self.CornerScore))

		return CornerScore

	def computeShiTomasiCorners(self, Image, Visualize):
		print("Computing Shi-Tomasi Corners...")
		# self.ImageSetShiTomasiCorners = np.copy(self.ImageSet)
		# for img in range(len(self.ImageSetGray)):

		ImageGray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
		corners = cv2.goodFeaturesToTrack(ImageGray, self.NumFeatures, 0.01, 3) # Computing corners using the Shi-Tomasi method
		corners = np.int0(corners)
		print("ShiTomasiCorners: %d"%(len(corners)))

		ShiTomasiCorners = np.zeros(Image.shape[0:2])
		ImageSetShiTomasiCorners = np.copy(Image)
		for corner in corners: # Marking corners in RGB image
			x,y = corner.ravel()
			ShiTomasiCorners[y,x] = 255
			# cv2.circle(self.ImageSetShiTomasiCorners[img],(x,y),2,(0,0,255),-1)
			cv2.circle(ImageSetShiTomasiCorners,(x,y),2,(0,0,255),-1)

		# self.ImageSetShiTomasiCorners = np.insert(self.ImageSetShiTomasiCorners, len(self.ImageSetShiTomasiCorners), ImageSetShiTomasiCorners, axis=0)
		# self.ShiTomasiCorners = np.insert(self.ShiTomasiCorners, len(self.ShiTomasiCorners), ShiTomasiCorners, axis=0)

		if(Visualize):
			cv2.imshow("Shi-Tomasi Corners", ImageSetShiTomasiCorners)
			cv2.imshow("Corners", ShiTomasiCorners)
			cv2.waitKey(0)

		# self.ShiTomasiCorners = np.array(self.ShiTomasiCorners)
		# print(self.ShiTomasiCorners.shape)

		return ShiTomasiCorners

	def ANMS(self, Image, ImageCorners, Visualize):
		print("Applying ANMS...")

		# self.ImageSetLocalMaxima = np.copy(self.ImageSet)
		# self.ImageSetANMS = np.copy(self.ImageSet)
		# for img in range(len(ImageSetCorners)):

		ANMSCorners = list()
		local_maximas = peak_local_max(ImageCorners, min_distance=1)
		local_maximas = np.int0(local_maximas)
		print("Local Maximas: %d"%len(local_maximas))

		# print(self.NumFeatures, len(local_maximas))
		if(self.NumFeatures > len(local_maximas)):
			self.NumFeatures = len(local_maximas)

		r = [np.Infinity for i in range(len(local_maximas))]
		ED = 0

		for i in tqdm(range(len(local_maximas))):
			for j in range(len(local_maximas)):
				if(ImageCorners[local_maximas[j,0],local_maximas[j,1]] > ImageCorners[local_maximas[i,0],local_maximas[i,1]]):
					ED = math.sqrt((local_maximas[j,0] - local_maximas[i,0])**2 + (local_maximas[j,1] - local_maximas[i,1])**2)
					# print(ED)
				if(ED < r[i]):
					r[i] = ED
			ANMSCorners.append([r[i], local_maximas[i,0], local_maximas[i,1]])

		ANMSCorners = sorted(ANMSCorners, reverse=True)
		ANMSCorners = np.array(ANMSCorners[:self.NumFeatures])
		print("ANMS Corners: %d"%len(ANMSCorners))
		# self.ANMSCorners = np.insert(self.ANMSCorners, len(self.ANMSCorners), ANMSCorners, axis=0)
		# print(self.ANMSCorners[-1])

		ImageSetLocalMaxima = np.copy(Image)
		ImageSetANMS = np.copy(Image)

		for local_maxima in local_maximas: # Marking corners in RGB image
				y,x = local_maxima.ravel()
				cv2.circle(ImageSetLocalMaxima,(x,y),2,(0,255,0),-1)
				# cv2.circle(self.ImageSetANMS[img],(x,y),2,(0,255,0),-1)

		for i in range(self.NumFeatures): # Marking corners in RGB image
			cv2.circle(ImageSetANMS,(int(ANMSCorners[i][2]),int(ANMSCorners[i][1])),2,(0,0,255),-1)
		
		# self.ImageSetLocalMaxima = np.insert(self.ImageSetLocalMaxima, len(self.ImageSetLocalMaxima), ImageSetLocalMaxima, axis=0)
		# self.ImageSetANMS = np.insert(self.ImageSetANMS, len(self.ImageSetANMS), ImageSetANMS, axis=0)

		if(Visualize):
			# cv2.imshow("", self.ImageSet[self.ImageCount])
			cv2.imshow("Local Max", ImageSetLocalMaxima)
			cv2.imshow("ANMS", ImageSetANMS)
			cv2.waitKey(0)

		# self.ANMSCorners = np.array(self.ANMSCorners)
		# print(self.ANMSCorners.shape)

		return ANMSCorners, ImageSetLocalMaxima, ImageSetANMS

	def featureDescriptor(self, Image, key_points, Visualize):
		print("Retrieving feature patches...")

		# for img in range(len(self.ImageSetGray)):
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
				# cv2.imshow("ANMS", self.ImageSetANMS[img])
				cv2.imshow("Patch", patch)
				cv2.imshow("Patch gauss", patch_gauss)
				cv2.waitKey(0)

		features = np.array(features)
		# self.Features = np.insert(self.Features, len(self.Features), features, axis=0)

		# self.Features = np.array(self.Features)
		# print(self.Features.shape)
		return features

	def featureMatching(self, Image0, Image1, Features0, Features1, ANMSCorners0, ANMSCorners1, Visualize):
		print("Matching features...")
		# for img in range(len(self.ImageSet)-1):
			
		SSDs = list()
		matches = list()
		if(len(Features0) > len(Features1)):
			N = len(Features1)
		else:
			N = len(Features0)
		features = np.arange(N).tolist()

		if(Visualize):
			temp = np.hstack((Image0, Image1))
		
		for i in tqdm(range(N)):
			SSDs.clear()
			for j in features:
				SSDs.append([sum((Features0[i] - Features1[j])**2), ANMSCorners1[j][1], ANMSCorners1[j][2]])

			SSDs = sorted(SSDs)
			# print(SSDs)
			# print([self.ANMSCorners[img][i][1], self.ANMSCorners[img][i][2], SSDs[0][1], SSDs[0][2]])
			matches.append([ANMSCorners0[i][1], ANMSCorners0[i][2], SSDs[0][1], SSDs[0][2]])
			# input('q')
			# features.remove(SSDs[0][1])

			if(Visualize):
				temp = cv2.circle(temp,(int(ANMSCorners0[i][2]), int(ANMSCorners0[i][1])),2,(0,0,255),-1)
				temp = cv2.circle(temp,(int(SSDs[0][2])+Image0.shape[1], int(SSDs[0][1])),2,(0,0,255),-1)
				temp = cv2.line(temp, (int(ANMSCorners0[i][2]), int(ANMSCorners0[i][1])), (int(SSDs[0][2])+Image0.shape[1], int(SSDs[0][1])), (0,255,0), 1)
				cv2.imshow("Matches", temp)
				# cv2.imshow("2", temp2)

		if(Visualize):
			cv2.waitKey(0)

		print("Matches: %d"%len(matches))

		matches = np.array(matches)
		# self.Matches = np.insert(self.Matches, len(self.Matches), matches, axis=0)

		# self.Matches = np.array(self.Matches)
		# print(self.Matches.shape)

		return matches

	def RANSAC(self, Matches, Image0, Image1, iterations, threshold, Visualize):
		print("Performing RANSAC...")
		# for img in range(len(self.ImageSet)-1):

		# print(Matches)

		max_inliers = 0
		best_H = None
		Inliers = list()
		features = np.arange(len(Matches)).tolist()
		# print(features)
		# input('q')

		# pbar = tqdm()
		for i in tqdm(range(iterations)):
		# while(max_inliers<30):

			feature_pairs = np.random.choice(features, 4, replace=False)
			p1 = list()
			p2 = list()
			for j in range(len(feature_pairs)):
				p1.append([Matches[feature_pairs[j]][1], Matches[feature_pairs[j]][0]])
				p2.append([Matches[feature_pairs[j]][3], Matches[feature_pairs[j]][2]])

			# p1 = np.array(p1)
			# p2 = np.array(p2)
			# print("p1", p1)
			# print("p2", p2)

			H = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p2))
			Hp1 = np.dot(H, np.vstack((Matches[:,1], Matches[:,0], np.ones([1,len(Matches)]))))
			Hp1 = np.array(Hp1/(Hp1[2]+1e-20)).transpose()
			Hp1 = np.delete(Hp1, 2, 1)
			p2_ = list()
			[p2_.append([Matches[x][3], Matches[x][2]]) for x in range(len(Matches))]
			p2_ = np.array(p2_)

			# print("Matches: ", self.Matches[img])
			# print("Hp1", Hp1)
			# print("p2_", p2_)
			# input('q')

			SSD = list()
			[SSD.append(sum((p2_[x] - Hp1[x])**2)) for x in range(len(Matches))]

			SSD = np.array(SSD)
			# print(SSD)
			SSD[SSD <= threshold] = 1
			SSD[SSD > threshold] = 0

			inliers = np.sum(SSD)

			if(inliers > max_inliers):
				max_inliers = inliers
				Inliers = np.where(SSD == 1)
				best_H = H
				# print("Inliers: %d"%max_inliers)
				# print("H: ", H)

			# pbar.update(1)

		# pbar.close()

		print("Inliers: %d"%max_inliers)
		# print("Homography Matrix: ", best_H)

		if(Visualize):
			temp = np.hstack((Image0, Image1))
			for i in Inliers[0]:
				temp = cv2.circle(temp,(int(Matches[i][1]), int(Matches[i][0])),2,(0,0,255),-1)
				temp = cv2.circle(temp,(int(Matches[i][3])+Image0.shape[1], int(Matches[i][2])),2,(0,0,255),-1)
				temp = cv2.line(temp, (int(Matches[i][1]), int(Matches[i][0])), (int(Matches[i][3])+Image0.shape[1], int(Matches[i][2])), (0,255,0), 1)
				# print((int(self.ANMSCorners[img][i][1]), int(self.ANMSCorners[img][i][2])), (int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1]), int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])))
			cv2.imshow("", temp)
			cv2.waitKey(0)

		# Inliers = np.array(Inliers[0]).reshape((-1,1))
		# self.Inliers = np.insert(self.Inliers, len(self.Inliers), Inliers, axis=0)
		self.Homography = np.insert(self.Homography, len(self.Homography), np.array([best_H]), axis=0)

		# self.Inliers = np.array(self.Inliers, dtype=object)
		# self.Homography = np.array(self.Homography)

		return best_H

	def stitchImages(self, Image0, Image1, H, Visualize):
		print("Blending Images...")

		h0, w0 = Image0.shape[:2]
		h1, w1 = Image1.shape[:2]

		c0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
		c1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

		# print(c0)
		# print(c1)

		# print("Homography Matrix", H)

		c0_ = cv2.perspectiveTransform(c0, H)
		# print(c0_)

		points_on_merged_images = np.concatenate((c0_, c1), axis = 0)
		points_on_merged_images_ = []

		for p in range(len(points_on_merged_images)):
			points_on_merged_images_.append(points_on_merged_images[p].ravel())

		points_on_merged_images_ = np.array(points_on_merged_images_)

		x_min, y_min = np.int0(np.min(points_on_merged_images_, axis = 0))
		x_max, y_max = np.int0(np.max(points_on_merged_images_, axis = 0))
		
		# print("min, max")
		# print(x_min, y_min)
		# print(x_max, y_max)

		H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

		image0_transformed = cv2.warpPerspective(Image0, np.dot(H_translate, H), (x_max-x_min, y_max-y_min))

		images_stitched = image0_transformed.copy()
		# print(images_stitched.shape)
		# print("test", -y_min, -y_min+h1, -x_min, -x_min+w1)
		images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = Image1

		# indices = np.where(Image1 == [0,0,0])
		# y = indices[0] + -y_min 
		# x = indices[1] + -x_min 

		# images_stitched[y,x] = image0_transformed[y,x]

		if(Visualize):
			# cv2.imshow("IMG", self.ImageSet[img])
			# cv2.imshow("Ref", self.ImageSet[img+1])
			# cv2.imshow("Transformed", image0_transformed)
			cv2.imshow("Stiched", images_stitched)
			cv2.waitKey(0)

		return images_stitched

	def generatePanorama(self, Visualize):
		print("Generating Panorama...")

		self.createImageSet()
		I = self.ImageSet[1]

		ShiTomasiCorners0 = self.computeShiTomasiCorners(self.ImageSet[0], False)
		ANMSCorners0, _, _ = self.ANMS(self.ImageSet[0], ShiTomasiCorners0, False)
		Features0 = self.featureDescriptor(self.ImageSet[0], ANMSCorners0, False)

		# HarrisCorners1 = self.computeHarrisCorners(self.ImageSet[img+1], True)
		ShiTomasiCorners1 = self.computeShiTomasiCorners(I, False)
		ANMSCorners1, _, _ = self.ANMS(I, ShiTomasiCorners1, False)
		Features1 = self.featureDescriptor(I, ANMSCorners1, False)

		Matches = self.featureMatching(self.ImageSet[0], I, Features0, Features1, ANMSCorners0, ANMSCorners1, False)
		H = self.RANSAC(Matches, self.ImageSet[0], I, 5000, 5, False)
		I = self.stitchImages(self.ImageSet[0], I, H, True)
		
		for img in range(2, 4):
			print("Stitching Frames %d & %d"%(img, img+1))

			# HarrisCorners0 = self.computeHarrisCorners(self.ImageSet[img], True)
			ShiTomasiCorners0 = self.computeShiTomasiCorners(self.ImageSet[img], False)
			ANMSCorners0, _, _ = self.ANMS(self.ImageSet[img], ShiTomasiCorners0, False)
			Features0 = self.featureDescriptor(self.ImageSet[img], ANMSCorners0, False)

			# HarrisCorners1 = self.computeHarrisCorners(self.ImageSet[img+1], True)
			ShiTomasiCorners1 = self.computeShiTomasiCorners(I, False)
			ANMSCorners1, _, _ = self.ANMS(I, ShiTomasiCorners1, False)
			Features1 = self.featureDescriptor(I, ANMSCorners1, False)

			Matches = self.featureMatching(self.ImageSet[img], I, Features0, Features1, ANMSCorners0, ANMSCorners1, False)
			H = self.RANSAC(Matches, self.ImageSet[img], I, 5000, 5, False)
			I = self.stitchImages(self.ImageSet[img], I, H, True)

			# Matches = self.featureMatching(I, self.ImageSet[img], Features1, Features0, ANMSCorners1, ANMSCorners0, False)
			# H = self.RANSAC(Matches, I, self.ImageSet[img], 5000, 5, False)
			# I = self.stitchImages(I, self.ImageSet[img], H, True)

		return

		for img in range(self.ImageSetRefId+1, len(self.ImageSet)):
			print("Stitching Frames %d & %d"%(img, img+1))

			ShiTomasiCorners0 = self.computeShiTomasiCorners(self.ImageSet[img], False)
			ANMSCorners0, _, _ = self.ANMS(self.ImageSet[img], ShiTomasiCorners0, False)
			Features0 = self.featureDescriptor(self.ImageSet[img], ANMSCorners0, False)

			ShiTomasiCorners1 = self.computeShiTomasiCorners(I, False)
			ANMSCorners1, _, _ = self.ANMS(I, ShiTomasiCorners1, False)
			Features1 = self.featureDescriptor(I, ANMSCorners1, False)

			Matches = self.featureMatching(I, self.ImageSet[img], Features0, Features1, ANMSCorners0, ANMSCorners1, False)
			H = self.RANSAC(Matches, self.ImageSet[img], I, 5000, 5, False)

			I = self.stitchImages(self.ImageSet[img], I, H, True)

			# ShiTomasiCorners0 = self.computeShiTomasiCorners(self.ImageSet[img+2], False)
			# ANMSCorners0, _, _ = self.ANMS(self.ImageSet[img+2], ShiTomasiCorners0, False)
			# Features0 = self.featureDescriptor(self.ImageSet[img+2], ANMSCorners0, False)

			# ShiTomasiCorners1 = self.computeShiTomasiCorners(I, False)
			# ANMSCorners1, _, _ = self.ANMS(I, ShiTomasiCorners1, False)
			# Features1 = self.featureDescriptor(I, ANMSCorners1, False)

			# Matches = self.featureMatching(self.ImageSet[img+2], I, Features0, Features1, ANMSCorners0, ANMSCorners1, False)
			# H = self.RANSAC(Matches, self.ImageSet[img+2], I, 5000, 5, False)

			# I = self.blendImages(self.ImageSet[img+2], I, H, True)

			# if(Visualize):
				# cv2.imshow("IMG", self.ImageSet[img])
				# cv2.imshow("IMG Gray", self.ImageSetGray[img])
				# cv2.waitKey(0)