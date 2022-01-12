#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import os


def readImageSet(ImageSet):
	return [os.path.join(ImageSet,  f) for f in sorted(os.listdir(ImageSet))]

def neighbors(a, radius, row_number, column_number):
	 return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
				for j in range(column_number-1-radius, column_number+radius)]
					for i in range(row_number-1-radius, row_number+radius)]

def remap(img, min_, max_):
	return cv2.normalize(np.float32(img), None, min_, max_, cv2.NORM_MINMAX)