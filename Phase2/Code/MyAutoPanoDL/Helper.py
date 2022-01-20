#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import os
import re
from PIL import Image


def readImageSet(ImageSet):
	return [os.path.join(ImageSet,  f) for f in sorted(os.listdir(ImageSet), key=lambda x:int(re.sub("\D","",x)))]

def getPatch(a, radius, row_number, column_number):
	 return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
				for j in range(column_number-1-radius, column_number+radius)]
					for i in range(row_number-1-radius, row_number+radius)]

def load_image(path):
    img = Image.open(path)
    return img

def preprocess_image(cv_img):
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')

def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray((img).astype(np.uint8))
    im.save(path)