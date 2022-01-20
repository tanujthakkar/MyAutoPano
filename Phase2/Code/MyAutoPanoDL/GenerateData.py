#!/usr/env/bin python3

# Importing modules
import cv2
import numpy as np
import math
import os
import sys
from tqdm import tqdm
import argparse
from PIL import Image
import random

from Helper import *

sys.dont_write_bytecode = True


def generateData(DatasetPath, Resize, PatchSize, MaxPerturbation, Tolerance, NumPatches, SavePatches, SavePath, Visualize):

    PatchePerImage = int(NumPatches/len(DatasetPath))
    print("Total Images: %d"%len(DatasetPath))
    print("Total Patches: %d"%NumPatches)
    print("Patches per Image: %d"%PatchePerImage)

    if(SavePatches):
        os.makedirs(SavePath, exist_ok=True)
        print("Saving genereated patches to: %s"%SavePath)

    for img in tqdm(DatasetPath):
        for patch in range(PatchePerImage):
            ImageA = cv2.resize(cv2.imread(img), (int(Resize[1]), int(Resize[0])), interpolation=cv2.INTER_CUBIC) # Input ImageA
            ImageAGray = cv2.cvtColor(ImageA, cv2.COLOR_BGR2GRAY)

            P_xmin = int((PatchSize[0]/2 + MaxPerturbation))
            P_xmax = int(Resize[0] - (PatchSize[0]/2 + MaxPerturbation))
            P_ymin = int((PatchSize[1]/2 + MaxPerturbation))
            P_ymax = int(Resize[1] - (PatchSize[1]/2 + MaxPerturbation))
            P = [random.randint(P_xmin, P_xmax), random.randint(P_ymin, P_ymax)] # Random patch center [row, column]

            PatchA = np.uint8(np.array(getPatch(ImageAGray, int(PatchSize[0]/2), P[0], P[1]))) # Random patch from ImageA

            C_A = np.float32([[P[1]-PatchSize[1]/2, P[0]-PatchSize[0]/2],
                             [P[1]-PatchSize[1]/2, P[0]+PatchSize[0]/2],
                             [P[1]+PatchSize[1]/2, P[0]+PatchSize[0]/2],
                             [P[1]+PatchSize[1]/2, P[0]-PatchSize[0]/2]]).reshape(-1, 2) # Corners of the patch [column, row]
            C_A -= 1

            rho = np.random.randint(-32, 32, size=(4,1)) # Random perturbation

            C_B = np.copy(C_A) + rho # Perturbated corners

            H_AB = cv2.getPerspectiveTransform(np.float32(C_A), np.float32(C_B)) # Homography from C_A to C_B
            H_BA = np.linalg.inv(H_AB) # Homography from C_B to C_A

            ImageB = cv2.warpPerspective(ImageA, H_BA, (ImageA.shape[1], ImageA.shape[0])) # ImageA warped with respect to H_BA
            ImageBGray = cv2.cvtColor(ImageB, cv2.COLOR_BGR2GRAY)

            PatchB = np.uint8(np.array(getPatch(ImageBGray, int(PatchSize[0]/2), P[0], P[1]))) # Warped Patch

            PatchStack = np.dstack((PatchA, PatchB)) # Stacking patches channel-wise for network input
            print(PatchStack.shape)

            H4 = (C_B - C_A)
            print(C_B)
            print(C_A)
            print(np.float32(H4).shape)

            if(Visualize):
                # ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(Boundary)], True, (255, 255, 255), 2)
                ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(C_A)], True, (255, 0, 0), 2)
                ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(C_B)], True, (0, 0, 255), 2)
                ImageB = cv2.polylines(np.uint8(ImageB), [np.int32(C_A)], True, (0, 255, 0), 2)
                
                cv2.imshow("ImageA", ImageA)
                cv2.imshow("ImageA Gray", ImageAGray)
                cv2.imshow("PatchA", PatchA)
                cv2.imshow("ImageB", ImageB)
                cv2.imshow("PatchB", PatchB)
                cv2.waitKey(0)

            # return PatchStack, H_AB, H4


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DatasetPath', type=str, default="../Data/Train/", help='Path of the ImageA Dataset')
    Parser.add_argument('--Resize', default=[240, 320], help='Target size of genereate images (height, width)')
    Parser.add_argument('--PatchSize', default=[128, 128], help='Target size of the generated patches')
    Parser.add_argument('--MaxPerturbation', type=int, default=32, help='Maximum perturbation to generate random homography estimates')
    Parser.add_argument('--Tolerance', type=int, default=10, help='Tolerance of patch center selection')
    Parser.add_argument('--NumPatches', type=int, default=50000, help='Total number of patches to be generated')
    Parser.add_argument('--SavePatches', type=bool, default=False, help='Toggle to save generated patches')
    Parser.add_argument('--SavePath', type=str, default="../Data/Patches/Train", help='Path to store the generated patches')

    Args = Parser.parse_args()
    DatasetPath = Args.DatasetPath
    Resize = Args.Resize.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    Resize = np.array([int(i) for i in Resize])
    PatchSize = Args.PatchSize.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    PatchSize = np.array([int(i) for i in PatchSize])
    MaxPerturbation = Args.MaxPerturbation
    NumPatches = Args.NumPatches
    Tolerance = Args.Tolerance
    SavePatches = Args.SavePatches
    SavePath = Args.SavePath
    Visualize = True

    generateData(readImageSet(DatasetPath), Resize, PatchSize, MaxPerturbation, Tolerance, NumPatches, SavePatches, SavePath, Visualize)


if __name__ == '__main__':
    main()