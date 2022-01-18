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

	def __init__(self, ImageSetPath, NumFeatures):
		self.ImageCount = 0
		self.ImageSetPath = ImageSetPath
		self.NumFeatures = NumFeatures
		self.ImageSetHeight = cv2.imread(ImageSetPath[0]).shape[0]
		self.ImageSetWidth = cv2.imread(ImageSetPath[0]).shape[1]
		self.ImageSet = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ImageSetGray = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth]))
		self.ImageSetHarrisCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ImageSetShiTomasiCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.HarrisCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth]))
		self.ShiTomasiCorners = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth]))
		self.ImageSetLocalMaxima = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ImageSetANMS = np.uint8(np.empty([0, self.ImageSetHeight, self.ImageSetWidth, 3]))
		self.ANMSCorners = np.empty([0, self.NumFeatures, 3])
		self.Features = np.empty([0, self.NumFeatures, 64])
		self.Matches = list()
		self.Inliers = list()
		self.Homography = list()
		self.BlendedImage = None
		self.ImageSetRefId = None

		# Toggles
		self.Visualize = False

	def createImageSet(self, ImageSet, N=None, N_best=0, height=None, width=None):
		if(N == None):
			N = len(ImageSet)
		if(not height and not width):
			height = cv2.imread(ImageSet[0]).shape[0]
			width = cv2.imread(ImageSet[0]).shape[1]
		[self.ImageSet.append(cv2.imread(ImageSet[img])) for img in range(N)] # Reading images
		[self.ImageSetGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.ImageSet] # Converting images to grayscale
		self.ImageSet = np.array(self.ImageSet) 
		self.ImageSetGray = np.float32(np.array(self.ImageSetGray))
		self.ImageSetRefId = int(len(self.ImageSet)/2) # Setting a reference to the anchor image
		self.N_best = N_best

		# Initializing other lists
		self.HarrisCorners = np.empty([0, height, width, 3])
		self.ImageSetShiTomasiCorners = np.uint8(np.empty([0, height, width, 3]))
		self.ShiTomasiCorners = np.uint8(np.empty([0, height, width]))
		self.ImageSetLocalMaxima = np.uint8(np.empty([0, height, width, 3]))
		self.ImageSetANMS = np.uint8(np.empty([0, height, width, 3]))
		self.ANMSCorners = np.empty([0, N_best, 3])
		self.Features = np.empty([0, N_best, 64])

	def computeHarrisCorners(self, Visualize):
		print("Computing Harris Corners...")
		self.ImageSetHarrisCorners = np.copy(self.ImageSet)
		for i in range(len(self.ImageSetGray)):
			img = cv2.cornerHarris(self.ImageSetGray[i], 2, 3, 0.00001) # Computing corner probability using Harris corners
			# img = cv2.normalize(img, None, -1.0, 1.0, cv2.NORM_MINMAX) # Normalizing
			img[img<0.001*img.max()] = 0
			img = cv2.dilate(img, None) # Dilating to mark corners
			self.ImageSetHarrisCorners[i][img>0.001*img.max()]=[0,0,255] # Marking corners in RGB image
			self.HarrisCorners.append(img)
			if(Visualize):
				# cv2.imshow("", img)
				cv2.imshow("Harris Corners", self.ImageSetHarrisCorners[i])
				cv2.imshow("Corner Score", self.HarrisCorners[i])
				cv2.waitKey(0)
		self.HarrisCorners = np.float32(np.array(self.HarrisCorners))

	def computeShiTomasiCorners(self, Image, Visualize):
		print("Computing Shi-Tomasi Corners...")
		# self.ImageSetShiTomasiCorners = np.copy(self.ImageSet)
		# for img in range(len(self.ImageSetGray)):

		ImageGray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
		corners = cv2.goodFeaturesToTrack(ImageGray, self.NumFeatures, 0.01, 10) # Computing corners using the Shi-Tomasi method
		corners = np.int0(corners)
		print("ShiTomasiCorners in Image %d: %d"%(0, len(corners)))

		ShiTomasiCorners = np.zeros(Image.shape[0:2])
		ImageSetShiTomasiCorners = np.copy(Image)
		for corner in corners: # Marking corners in RGB image
			x,y = corner.ravel()
			ShiTomasiCorners[y,x] = 255
			# cv2.circle(self.ImageSetShiTomasiCorners[img],(x,y),2,(0,0,255),-1)
			cv2.circle(ImageSetShiTomasiCorners,(x,y),2,(0,0,255),-1)

		self.ImageSetShiTomasiCorners = np.insert(self.ImageSetShiTomasiCorners, self.ImageCount, ImageSetShiTomasiCorners, axis=0)
		self.ShiTomasiCorners = np.insert(self.ShiTomasiCorners, self.ImageCount, ShiTomasiCorners, axis=0)

		if(Visualize):
			cv2.imshow("Shi-Tomasi Corners", self.ImageSetShiTomasiCorners[-1])
			cv2.imshow("Corners", self.ShiTomasiCorners[-1])
			cv2.waitKey(0)

		# self.ShiTomasiCorners = np.array(self.ShiTomasiCorners)
		print(self.ShiTomasiCorners.shape)

	def ANMS(self, Image, ImageSetCorners, N_best, Visualize):
		print("Applying ANMS...")

		print(ImageSetCorners.shape)
		# self.ImageSetLocalMaxima = np.copy(self.ImageSet)
		# self.ImageSetANMS = np.copy(self.ImageSet)
		# for img in range(len(ImageSetCorners)):
		ANMSCorners = list()
		local_maximas = peak_local_max(ImageSetCorners[self.ImageCount], min_distance=1)
		local_maximas = np.int0(local_maximas)
		print("local_maximas: %d"%len(local_maximas))

		if(N_best < len(local_maximas)):
			print('test')
			N_best = len(local_maximas)

		r = [np.Infinity for i in range(len(local_maximas))]
		ED = 0

		for i in tqdm(range(len(local_maximas))):
			for j in range(len(local_maximas)):
				if(ImageSetCorners[self.ImageCount][local_maximas[j,0],local_maximas[j,1]] > ImageSetCorners[self.ImageCount][local_maximas[i,0],local_maximas[i,1]]):
					ED = math.sqrt((local_maximas[j,0] - local_maximas[i,0])**2 + (local_maximas[j,1] - local_maximas[i,1])**2)
					# print(ED)
				if(ED < r[i]):
					r[i] = ED
			ANMSCorners.append([r[i], local_maximas[i,0], local_maximas[i,1]])

		ANMSCorners = sorted(ANMSCorners, reverse=True)
		ANMSCorners = np.array(ANMSCorners[:N_best])
		print("ANMS Corners: %d"%len(ANMSCorners))
		self.ANMSCorners = np.insert(self.ANMSCorners, self.ImageCount, ANMSCorners, axis=0)
		# print(self.ANMSCorners[-1])

		ImageSetLocalMaxima = np.copy(self.ImageSet[self.ImageCount])
		ImageSetANMS = np.copy(self.ImageSet[self.ImageCount])

		for local_maxima in local_maximas: # Marking corners in RGB image
				y,x = local_maxima.ravel()
				cv2.circle(ImageSetLocalMaxima,(x,y),2,(0,255,0),-1)
				# cv2.circle(self.ImageSetANMS[img],(x,y),2,(0,255,0),-1)

		for i in range(N_best): # Marking corners in RGB image
			cv2.circle(ImageSetANMS,(int(ANMSCorners[i][2]),int(ANMSCorners[i][1])),2,(0,0,255),-1)
		
		self.ImageSetLocalMaxima = np.insert(self.ImageSetLocalMaxima, self.ImageCount, ImageSetLocalMaxima, axis=0)
		self.ImageSetANMS = np.insert(self.ImageSetANMS, self.ImageCount, ImageSetANMS, axis=0)

		if(Visualize):
			# cv2.imshow("", self.ImageSet[self.ImageCount])
			cv2.imshow("Local Max", self.ImageSetLocalMaxima[-1])
			cv2.imshow("ANMS", self.ImageSetANMS[-1])
			cv2.waitKey(0)

		# self.ANMSCorners = np.array(self.ANMSCorners)
		print(self.ANMSCorners.shape)

	def featureDescriptor(self, key_points, Visualize):
		print("Retrieving feature patches...")

		# for img in range(len(self.ImageSetGray)):
		patch_size = 40
		features = list()
		for point in range(len(key_points[self.ImageCount])):
			patch = np.uint8(np.array(neighbors(self.ImageSetGray[self.ImageCount], 20, int(key_points[self.ImageCount][point][1]), int(key_points[self.ImageCount][point][2]))))
			patch_gauss = cv2.resize(cv2.GaussianBlur(patch, (5,5), 0), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
			patch_gauss = (patch_gauss - patch_gauss.mean())/patch_gauss.std()
			features.append(patch_gauss.flatten())
			if(Visualize):
				temp = cv2.circle(np.copy(self.ImageSet[self.ImageCount]),(int(key_points[self.ImageCount][point][2]), int(key_points[self.ImageCount][point][1])),2,(0,0,255),-1)
				cv2.imshow("Feature", temp)
				# cv2.imshow("ANMS", self.ImageSetANMS[img])
				cv2.imshow("Patch", patch)
				cv2.imshow("Patch gauss", patch_gauss)
				cv2.waitKey(0)

		features = np.array(features)
		self.Features = np.insert(self.Features, self.ImageCount, features, axis=0)

		# self.Features = np.array(self.Features)
		# print(self.Features.shape)

	def featureMatching(self, Visualize):
		print("Matching features...")
		# for img in range(len(self.ImageSet)-1):
			
		SSDs = list()
		matches = list()
		features = np.arange(len(self.Features[self.ImageCount])).tolist()
		temp = np.hstack((self.ImageSet[self.ImageCount], self.ImageSet[self.ImageCount+1]))
		for i in tqdm(range(len(self.Features[self.ImageCount]))):
			SSDs.clear()
			for j in features:
				SSDs.append([sum((self.Features[self.ImageCount][i] - self.Features[self.ImageCount+1][j])**2), self.ANMSCorners[self.ImageCount+1][j][1], self.ANMSCorners[self.ImageCount+1][j][2]])

			SSDs = sorted(SSDs)
			# print([self.ANMSCorners[img][i][1], self.ANMSCorners[img][i][2], SSDs[0][1], SSDs[0][2]])
			matches.append([self.ANMSCorners[self.ImageCount][i][1], self.ANMSCorners[self.ImageCount][i][2], SSDs[0][1], SSDs[0][2]])
			# input('q')
			# features.remove(SSDs[0][1])

			if(Visualize):
				# temp = np.hstack((self.ImageSet[img], self.ImageSet[img+1]))
				temp = cv2.circle(temp,(int(self.ANMSCorners[self.ImageCount][i][2]), int(self.ANMSCorners[self.ImageCount][i][1])),2,(0,0,255),-1)
				temp = cv2.circle(temp,(int(SSDs[0][2])+self.ImageSet[self.ImageCount].shape[1], int(SSDs[0][1])),2,(0,0,255),-1)
				temp = cv2.line(temp, (int(self.ANMSCorners[self.ImageCount][i][2]), int(self.ANMSCorners[self.ImageCount][i][1])), (int(SSDs[0][2])+self.ImageSet[self.ImageCount].shape[1], int(SSDs[0][1])), (0,255,0), 1)
				cv2.imshow("1", temp)
				# cv2.imshow("2", temp2)

		print("Matches: %d", len(matches))
		cv2.waitKey(0)

		matches = np.array(matches)
		print(matches.shape)
		# self.Matches.append(matches)

		# self.Matches = np.array(self.Matches)
		# print(self.Matches.shape)

	def RANSAC(self, iterations, threshold, Visualize):
		print("Performing RANSAC...")
		for img in range(len(self.ImageSet)-1):

			max_inliers = 0
			best_H = None
			Inliers = list()
			features = np.arange(len(self.Features[img])).tolist()
			for i in tqdm(range(iterations)):

				feature_pairs = np.random.choice(features, 4, replace=False)
				p1 = list()
				p2 = list()
				for j in range(len(feature_pairs)):
					p1.append([self.Matches[img][feature_pairs[j]][1], self.Matches[img][feature_pairs[j]][0]])
					p2.append([self.Matches[img][feature_pairs[j]][3], self.Matches[img][feature_pairs[j]][2]])

				# p1 = np.array(p1)
				# p2 = np.array(p2)

				H = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p2))
				Hp1 = np.dot(H, np.vstack((self.Matches[img][:,1], self.Matches[img][:,0], np.ones([1,len(self.Matches[img])]))))
				Hp1 = np.array(Hp1/(Hp1[2]+1e-20)).transpose()
				Hp1 = np.delete(Hp1, 2, 1)
				p2_ = list()
				[p2_.append([self.Matches[img][x][3], self.Matches[img][x][2]]) for x in range(len(self.Matches[img]))]
				p2_ = np.array(p2_)

				# print("Matches: ", self.Matches[img])
				# print("p2_", p2_)
				# print("Hp1", Hp1)
				# input('q')

				SSD = list()
				[SSD.append(sum((p2_[x] - Hp1[x])**2)) for x in range(len(self.Matches[img]))]

				SSD = np.array(SSD)
				SSD[SSD <= threshold] = 1
				SSD[SSD > threshold] = 0

				inliers = np.sum(SSD)

				if(inliers > max_inliers):
					max_inliers = inliers
					Inliers = np.where(SSD == 1)
					best_H = H
					# print("Inliers: %d"%max_inliers)
					# print("H: ", H)

			print("Inliers: %d"%max_inliers)
			print("Homography Matrix: ", best_H)
			if(Visualize):
				temp = np.hstack((self.ImageSet[img], self.ImageSet[img+1]))
				for i in Inliers[0]:
					temp = cv2.circle(temp,(int(self.Matches[img][i][1]), int(self.Matches[img][i][0])),2,(0,0,255),-1)
					temp = cv2.circle(temp,(int(self.Matches[img][i][3])+self.ImageSet[img].shape[1], int(self.Matches[img][i][2])),2,(0,0,255),-1)
					temp = cv2.line(temp, (int(self.Matches[img][i][1]), int(self.Matches[img][i][0])), (int(self.Matches[img][i][3])+self.ImageSet[img].shape[1], int(self.Matches[img][i][2])), (0,255,0), 1)
					# print((int(self.ANMSCorners[img][i][1]), int(self.ANMSCorners[img][i][2])), (int(self.ANMSCorners[img+1][self.Matches[img][i][1]][1]), int(self.ANMSCorners[img+1][self.Matches[img][i][1]][2])))
				cv2.imshow("", temp)
				cv2.waitKey(0)

			self.Inliers.append(np.array(Inliers[0]).reshape((-1,1)))
			self.Homography.append(np.array([best_H]))

		self.Inliers = np.array(self.Inliers, dtype=object)
		self.Homography = np.array(self.Homography)

	def generatePanorama(self):
		print("Generating Panorama...")

		print(self.ImageSetPath)
		for img in range(len(self.ImageSetPath)):

			self.ImageSet = np.insert(self.ImageSet, img, cv2.imread(self.ImageSetPath[img]), axis=0)
			print(self.ImageSet.shape)
			self.computeShiTomasiCorners(self.ImageSet[-1], True)


	def blendImages(self, Visualize):
		print("Blending Images...")
		# self.BlendedImage = np.copy(self.ImageSet[0])
		for img in range(len(self.ImageSet)-1):

			self.computeShiTomasiCorners(self.N_best, False)
			self.ANMS(self.ShiTomasiCorners, self.N_best, False)
			self.featureDescriptor(self.ANMSCorners, False)

			# self.computeShiTomasiCorners(self.N_best, False)
			# self.ANMS(self.ShiTomasiCorners, self.N_best, False)
			# self.featureDescriptor(self.ANMSCorners, False)

			# self.featureMatching(True)

			# self.ImageCount += 1

			# h0, w0 = self.ImageSet[img].shape[:2]
			# h1, w1 = self.ImageSet[img+1].shape[:2]

			# c0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1, 1, 2)
			# c1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

			# # H = np.ones((3,3))
			# # if(img < self.ImageSetRefId):
			# # 	for i in range(img+1):
			# # 		H = H*self.Homography[i][0]
			# # else:
			# # 	H = np.linalg.inv(self.Homography[img][0])

			# print("Homography Matrix", self.Homography[img][0])

			# c0_ = cv2.perspectiveTransform(c0, self.Homography[img][0])

			# points_on_merged_images = np.concatenate((c0_, c1), axis = 0)
			# points_on_merged_images_ = []

			# for p in range(len(points_on_merged_images)):
			# 	points_on_merged_images_.append(points_on_merged_images[p].ravel())

			# points_on_merged_images_ = np.array(points_on_merged_images_)

			# x_min, y_min = np.int0(np.min(points_on_merged_images_, axis = 0))
			# x_max, y_max = np.int0(np.max(points_on_merged_images_, axis = 0))
			
			# print("min, max")
			# print(x_min, y_min)
			# print(x_max, y_max)

			# H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

			# image0_transformed = cv2.warpPerspective(self.ImageSet[img], np.dot(H_translate, self.Homography[img][0]), (x_max-x_min, y_max-y_min))

			# images_stitched = image0_transformed.copy()
			# print(images_stitched.shape)
			# print("test", -y_min, -y_min+h1, -x_min, -x_min+w1)
			# images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = self.ImageSet[img+1]

			# indices = np.where(self.ImageSet[img+1] == [0,0,0])
			# y = indices[0] + -y_min 
			# x = indices[1] + -x_min 

			# images_stitched[y,x] = image0_transformed[y,x]

			# self.BlendedImage = images_stitched

			# self.ImageSet[img+1] = np.copy(images_stitched)

			# H += 1

			# if(Visualize):
			# 	# cv2.imshow("IMG", self.ImageSet[img])
			# 	# cv2.imshow("Ref", self.ImageSet[img+1])
			# 	# cv2.imshow("Transformed", image0_transformed)
			# 	cv2.imshow("Stiched", images_stitched)
			# 	cv2.waitKey(0)