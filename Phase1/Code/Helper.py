#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import os


def readImageSet(ImageSet):
	return [os.path.join(ImageSet,  f) for f in sorted(os.listdir(ImageSet))]