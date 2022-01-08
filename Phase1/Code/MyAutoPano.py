#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import os


class MyAutoPano():

	def __init__(self):
		self.ImageSet = []
		self.ImageSetGray = []
		self.ImageSetHarrisCorners = []
		self.ImageSetShiTomasiCorners = []
		self.HarrisCorners = []
		self.ShiTomasiCorners = []

		# Toggles
		self.Visualize = False

	def createImageSet(self, ImageSet):
		[self.ImageSet.append(cv2.imread(img)) for img in ImageSet]
		[self.ImageSetGray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in self.ImageSet]
		self.ImageSet = np.array(self.ImageSet)
		self.ImageSetGray = np.float32(np.array(self.ImageSetGray))

	def computeHarrisCorners(self):
		self.ImageSetHarrisCorners = self.ImageSet
		for i in range(len(self.ImageSetGray)):
			img = cv2.cornerHarris(self.ImageSetGray[i], 2, 3, 0.08)
			img = cv2.normalize(img, None, -1.0, 1.0, cv2.NORM_MINMAX)
			img = cv2.dilate(img, None)
			self.ImageSetHarrisCorners[i][img>0.3*img.max()]=[0,0,255]
			if(self.Visualize):
				# cv2.imshow("", img)
				cv2.imshow("", self.ImageSetHarrisCorners[i])
				cv2.waitKey(0)

	def computeShiTomasiCorners(self):
		self.ImageSetShiTomasiCorners = self.ImageSet
		for i in range(len(self.ImageSetGray)):
			corners = cv2.goodFeaturesToTrack(self.ImageSetGray[i], 0, 0.01, 10)
			corners = np.int0(corners)
			for corner in corners:
				x,y = corner.ravel()
				cv2.circle(self.ImageSetShiTomasiCorners[i],(x,y),2,(0,0,255),-1)
			if(self.Visualize):
				cv2.imshow("", self.ImageSetShiTomasiCorners[i])
				cv2.waitKey(0)