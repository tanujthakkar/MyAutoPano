#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import os
import re
from PIL import Image
import tensorflow.keras.backend as K
import tensorflow as tf


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

def remap(x, oMin, oMax, iMin, iMax):
    # Taken from https://stackoverflow.com/questions/929103/convert-a-number-range-to-another-range-maintaining-ratios
    #range check
    if oMin == oMax:
        print("Warning: Zero input range")
        return None

    if iMin == iMax:
        print("Warning: Zero output range")
        return None

     # portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    result = np.add(np.divide(np.multiply(x - iMin, oMax - oMin), iMax - iMin), oMin)

    return result

def preprocess_H4_data(H4, rho=32):
    H4 = remap(H4, -1.0, 1.0, -rho, rho)
    return H4.reshape(8)

def deprocess_H4_data(H4, rho=32):
    H4 = remap(H4, -rho, rho, -1.0, 1.0)
    return np.int32(H4.reshape(4,2))

def L2_loss(y_true, y_pred):
    # print(tf.reduce_sum((y_pred - y_true)**2, axis=0))
    # return tf.math.sqrt(tf.reduce_sum((tf.math.squared_difference(y_pred, y_true))))
    # return K.mean((y_pred-y_true)**2)
    # return tf.reduce_sum((y_pred - y_true)**2)/8
    return tf.reduce_mean(tf.reduce_sum((y_pred - y_true)**2, axis=1))

def main():
    print(remap(9.125, -1.0, 1.0, -32, 32))

if __name__ == '__main__':
    main()