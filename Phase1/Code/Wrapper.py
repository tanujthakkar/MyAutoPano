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

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--ImageSetPath', type=str, default="../Data/Train/Set1/", help='Path of the Image Set')
	Parser.add_argument('--NumImages', type=int, default=None, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--NumFeatures', type=int, default=400, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--ResultPath', type=str, default="../Data/Train/Results/", help='Path to save the generated results')
	Parser.add_argument('--TestName', type=str, default="Test", help="Name of the test case to store results")
	Parser.add_argument('--SaveResults', action='store_true', help='Toggle to save generated results')
	Parser.add_argument('--UseHarris', action='store_true', help='Toggle to use Harris corners instead of Shi-Tomasi')
	
	Args = Parser.parse_args()
	NumImages = Args.NumImages
	NumFeatures = Args.NumFeatures
	ResultPath = Args.ResultPath
	TestName = Args.TestName
	SaveResults = Args.SaveResults
	UseHarris = Args.UseHarris

	pano = MyAutoPano(readImageSet(Args.ImageSetPath), Args.NumFeatures, Args.ResultPath, TestName, 1.0, 1.0)
	pano.generatePanorama(True)
	
if __name__ == '__main__':
	main()