#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import math
import os
from scipy import ndimage as ndi
from skimage.feature import peak_local_max, corner_peaks
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

from Helper import *


class MyAutoPano():

	def __init__(self):
		self.ImageSet = list()
		self.ImageSetGray = list()
		self.ImageSetHarrisCorners = list()
		self.ImageSetShiTomasiCorners = list()
		self.HarrisCorners = list()
		self.ShiTomasiCorners = list()
		self.ImageSetLocalMaxima = list()
		self.ImageSetANMS = list()
		self.ANMSCorners = list()
		self.Features = list()

		# Toggles
		self.Visualize = False

	def createImageSet(self, ImageSet):
		[self.ImageSet.append(cv2.imread(img)) for img in ImageSet]
		[self.ImageSetGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.ImageSet]
		self.ImageSet = np.array(self.ImageSet)
		self.ImageSetGray = np.float32(np.array(self.ImageSetGray))

	def computeHarrisCorners(self):
		self.ImageSetHarrisCorners = np.copy(self.ImageSet)
		for i in range(len(self.ImageSetGray)):
			img = cv2.cornerHarris(self.ImageSetGray[i], 2, 3, 0.001) # Computing corner probability using Harris corners
			# img = cv2.normalize(img, None, -1.0, 1.0, cv2.NORM_MINMAX) # Normalizing
			img[img<0.001*img.max()] = 0
			img = cv2.dilate(img, None) # Dilating to mark corners
			self.ImageSetHarrisCorners[i][img>0.001*img.max()]=[0,0,255] # Marking corners in RGB image
			self.HarrisCorners.append(img)
			if(self.Visualize):
				# cv2.imshow("", img)
				cv2.imshow("Harris", self.ImageSetHarrisCorners[i])
				cv2.imshow("cimg", self.HarrisCorners[i])
				cv2.waitKey(0)
		self.HarrisCorners = np.float32(np.array(self.HarrisCorners))

	def computeShiTomasiCorners(self):
		self.ImageSetShiTomasiCorners = np.copy(self.ImageSet)
		for img in range(len(self.ImageSetGray)):
			corners = cv2.goodFeaturesToTrack(self.ImageSetGray[img], 0, 0.01, 10) # Computing corners using the Shi-Tomasi method
			corners = np.int0(corners)
			for corner in corners: # Marking corners in RGB image
				x,y = corner.ravel()
				cv2.circle(self.ImageSetShiTomasiCorners[img],(x,y),2,(0,0,255),-1)
			if(self.Visualize):
				cv2.imshow("Shi-Tomasi", self.ImageSetShiTomasiCorners[img])
				cv2.waitKey(0)

	def anms(self, ImageSetCorners, N_best):
		self.ImageSetLocalMaxima = np.copy(self.ImageSet)
		self.ImageSetANMS = np.copy(self.ImageSet)
		for img in range(len(ImageSetCorners)):
			ANMSCorners = list()
			local_maximas = peak_local_max(ImageSetCorners[img], min_distance=10)
			local_maximas = np.int0(local_maximas)

			r = [np.Infinity for i in range(len(local_maximas))]
			ED = 0

			for i in range(len(local_maximas)):
				for j in range(len(local_maximas)):
					if(ImageSetCorners[img][local_maximas[j,0],local_maximas[j,1]] > ImageSetCorners[img][local_maximas[i,0],local_maximas[i,1]]):
						ED = math.sqrt((local_maximas[j,0] - local_maximas[i,0])**2 + (local_maximas[j,1] - local_maximas[i,1])**2)
					if(ED < r[i]):
						r[i] = ED
				ANMSCorners.append([r[i], local_maximas[i,0], local_maximas[i,1]])

			ANMSCorners = sorted(ANMSCorners, reverse=True)
			ANMSCorners = np.array(ANMSCorners[:N_best])
			self.ANMSCorners.append(ANMSCorners)

			for local_maxima in local_maximas: # Marking corners in RGB image
				y,x = local_maxima.ravel()
				cv2.circle(self.ImageSetLocalMaxima[img],(x,y),2,(0,255,0),-1)
				# cv2.circle(self.ImageSetANMS[img],(x,y),2,(0,255,0),-1)

			for i in range(N_best): # Marking corners in RGB image
				cv2.circle(self.ImageSetANMS[img],(int(ANMSCorners[i][2]),int(ANMSCorners[i][1])),2,(0,0,255),-1)

			if(self.Visualize):
				cv2.imshow("Local Max", self.ImageSetLocalMaxima[img])
				cv2.imshow("ANMS", self.ImageSetANMS[img])
				cv2.waitKey(0)

		self.ANMSCorners = np.array(self.ANMSCorners)

	def featureDescriptor(self, key_points):
		for img in range(len(self.ImageSetGray)):
			patch_size = 40
			features = list()
			for point in range(len(key_points[img])):
				print(key_points[img][point])
				patch = np.uint8(np.array(neighbors(self.ImageSetGray[img], 20, int(key_points[img][point][1]), int(key_points[img][point][2]))))
				patch_gauss = cv2.resize(cv2.GaussianBlur(patch, (5,5), 0), None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
				patch_gauss = (patch_gauss - patch_gauss.mean())/patch_gauss.std()
				features.append(patch_gauss.flatten())
				if(self.Visualize):
					temp = cv2.circle(np.copy(self.ImageSet[img]),(int(key_points[img][point][2]), int(key_points[img][point][1])),2,(0,0,255),-1)
					cv2.imshow("Feature", Feature)
					# cv2.imshow("ANMS", self.ImageSetANMS[img])
					cv2.imshow("Patch", patch)
					cv2.imshow("Patch gauss", patch_gauss)
					cv2.waitKey(0)

			features = np.array(features)
			self.Features.append(features)

		self.Features = np.array(self.Features)
		print(self.Features.shape)