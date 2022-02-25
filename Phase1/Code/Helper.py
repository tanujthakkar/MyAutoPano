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

def crop_image(Image):

    img = Image.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[len(contours)-1])
    crop = img[y:y+h,x:x+w]

    return crop