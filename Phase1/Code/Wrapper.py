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
	Parser.add_argument('--ImageSet', type=str, default="../Data/Train/Set1/", help='Path of the Image Set')
	Parser.add_argument('--NumFeatures', type=int, default=100, help='Number of best features to extract from each image, Default:100')
	
	Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	pano = MyAutoPano()
	pano.createImageSet(readImageSet(Args.ImageSet))
	pano.Visualize = True

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	pano.computeHarrisCorners()
	# pano.computeShiTomasiCorners()

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""


	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

	
if __name__ == '__main__':
	main()