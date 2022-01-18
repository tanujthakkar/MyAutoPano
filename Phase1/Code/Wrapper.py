#!/usr/env/bin python3

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import argparse
# Add any python libraries here

from MyAutoPano import MyAutoPano
from Helper import *


def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageSetPath', type=str, default="../Data/Train/Set1/", help='Path of the Image Set')
	Parser.add_argument('--NumImages', type=int, default=None, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--NumFeatures', type=int, default=400, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--ResultPath', type=str, default="../Data/Train/Results/", help='Path to save the generated results')
	
	Args = Parser.parse_args()
	NumImages = Args.NumImages
	NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	pano = MyAutoPano(readImageSet(Args.ImageSetPath), Args.NumFeatures, Args.ResultPath, 300, 400)
	pano.generatePanorama(True)
	# pano.Visualize = True
	# pano.Visualize = False
	# pano.createImageSet(readImageSet(Args.ImageSet), NumImages, NumFeatures)

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	# pano.computeHarrisCorners(False)
	# pano.computeShiTomasiCorners(pano.ImageSetGray[0], NumFeatures, True)
	
	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	# pano.ANMS(pano.HarrisCorners, NumFeatures, False)
	# pano.ANMS(pano.ShiTomasiCorners, NumFeatures, False)

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	# pano.featureDescriptor(pano.ANMSCorners, False)

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""
	# pano.featureMatching(False)

	"""
	Refine: RANSAC, Estimate Homography
	"""
	# pano.RANSAC(5000, 5, True)


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	# H = pano.test(pano.ImageSet[0], pano.ImageSet[1])
	# pano.stitchImagePairs(pano.ImageSet[0], pano.ImageSet[1], pano.Homography[0][0])
	# pano.blendImages(True)
	
if __name__ == '__main__':
	main()