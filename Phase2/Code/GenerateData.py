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
from datetime import datetime
import tensorflow as tf

from Helper import *
from Misc.MiscUtils import *

sys.dont_write_bytecode = True


def generatePatchSet(ImagePath, Resize, PatchSize, MaxPerturbation, Tolerance, SavePatchSet, SavePath, PatchCount, Visualize):

    ImageA = cv2.resize(cv2.imread(ImagePath), (int(Resize[1]), int(Resize[0])), interpolation=cv2.INTER_CUBIC) # Input ImageA
    ImageAGray = cv2.cvtColor(ImageA, cv2.COLOR_BGR2GRAY)

    P_xmin = int((PatchSize[0]/2 + MaxPerturbation))
    P_xmax = int(Resize[0] - (PatchSize[0]/2 + MaxPerturbation))
    P_ymin = int((PatchSize[1]/2 + MaxPerturbation))
    P_ymax = int(Resize[1] - (PatchSize[1]/2 + MaxPerturbation))
    P = [random.randint(P_xmin, P_xmax), random.randint(P_ymin, P_ymax)] # Random patch center [row, column]

    PatchA = np.uint8(ImageAGray[P[0]-int(PatchSize[0]/2):P[0]+int(PatchSize[0]/2), P[1]-int(PatchSize[1]/2):P[1]+int(PatchSize[1]/2)])

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

    PatchB = np.uint8(ImageBGray[P[0]-int(PatchSize[0]/2):P[0]+int(PatchSize[0]/2), P[1]-int(PatchSize[1]/2):P[1]+int(PatchSize[1]/2)])

    PatchStack = np.dstack((PatchA, PatchB)) # Stacking patches channel-wise for network input
    # print(PatchStack.shape)

    H4 = (C_B - C_A)
    # print(C_B)
    # print(C_A)
    # print(np.float32(H4).shape)

    HC_A = np.dot(H_BA, np.vstack((C_A[:,0], C_A[:,1], np.ones([1,len(C_A)]))))
    HC_A = np.array(HC_A/(HC_A[2]+1e-20)).transpose()
    HC_A = np.delete(HC_A, 2, 1)

    Test = C_A + H4

    # print("H4", H4)
    # print("H_AB", H_AB)

    if(Visualize):
        # ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(Boundary)], True, (255, 255, 255), 2)
        ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(C_A)], True, (255, 0, 0), 2)
        ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(C_B)], True, (0, 0, 255), 2)
        # ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(Test)], True, (0, 255, 0), 2)
        ImageB = cv2.polylines(np.uint8(ImageB), [np.int32(C_A)], True, (0, 0, 255), 2)
        ImageB = cv2.polylines(np.uint8(ImageB), [np.int32(HC_A)], True, (255, 0, 0), 2)
        
        cv2.imshow("ImageA", ImageA)
        cv2.imshow("PatchA", PatchA)
        cv2.imshow("ImageB", ImageB)
        cv2.imshow("PatchB", PatchB)
        cv2.waitKey(0)

    if(SavePatchSet):
        os.makedirs(SavePath, exist_ok=True)
        os.makedirs(os.path.join(SavePath, 'PatchA'), exist_ok=True)
        os.makedirs(os.path.join(SavePath, 'PatchB'), exist_ok=True)
        cv2.imwrite(os.path.join(SavePath, 'PatchA', '%d.png'%PatchCount), PatchA)
        cv2.imwrite(os.path.join(SavePath, 'PatchB', '%d.png'%PatchCount), PatchB)

    return PatchStack, H4, H_AB, rho


def generateResult(ImageA, H4, H_AB, Resize, PatchSize, MaxPerturbation, Visualize):

    ImageA = cv2.resize(cv2.imread(ImageA), (int(Resize[1]), int(Resize[0])), interpolation=cv2.INTER_CUBIC) # Input ImageA
        
    H_BA = np.linalg.inv(H_AB)
    ImageB = cv2.warpPerspective(ImageA, H_BA, (ImageA.shape[1], ImageA.shape[0])) # ImageA warped with respect to H_BA

    P_xmin = int((PatchSize[0]/2 + MaxPerturbation))
    P_xmax = int(Resize[0] - (PatchSize[0]/2 + MaxPerturbation))
    P_ymin = int((PatchSize[1]/2 + MaxPerturbation))
    P_ymax = int(Resize[1] - (PatchSize[1]/2 + MaxPerturbation))
    P = [random.randint(P_xmin, P_xmax), random.randint(P_ymin, P_ymax)] # Random patch center [row, column]

    C_A = np.float32([[P[1]-PatchSize[1]/2, P[0]-PatchSize[0]/2],
                     [P[1]-PatchSize[1]/2, P[0]+PatchSize[0]/2],
                     [P[1]+PatchSize[1]/2, P[0]+PatchSize[0]/2],
                     [P[1]+PatchSize[1]/2, P[0]-PatchSize[0]/2]]).reshape(-1, 2) # Corners of the patch [column, row]
    C_A -= 1

    # print("C_A", C_A)

    # print("H4", H4)

    C_B = C_A + H4

    # print("C_B", C_B)

    HC_A = np.dot(H_BA, np.vstack((C_A[:,0], C_A[:,1], np.ones([1,len(C_A)]))))
    HC_A = np.array(HC_A/(HC_A[2]+1e-20)).transpose()
    HC_A = np.delete(HC_A, 2, 1)

    Test = C_A + H4

    ImageA = cv2.polylines(np.uint8(ImageA), [np.int32(C_A)], True, (255, 0, 0), 2)
    ImageB = cv2.polylines(np.uint8(ImageB), [np.int32(C_B)], True, (255, 0, 0), 2)
    ImageB = cv2.polylines(np.uint8(ImageB), [np.int32(HC_A)], True, (0, 255, 0), 2)
    Result = np.hstack((ImageA, ImageB))
 
    if(Visualize):
        cv2.imshow("ImageA", ImageA)
        cv2.imshow("ImageB", ImageB)
        cv2.imshow("Result", Result)
        cv2.waitKey(0)

    return np.uint8(Result)

def generateDataset(DatasetPath, Resize, PatchSize, MaxPerturbation, Tolerance, NumPatches, SavePatches, SavePath, Visualize):

    PatchePerImage = int(NumPatches/len(DatasetPath))
    print("Total Images: %d"%len(DatasetPath))
    print("Total Patches: %d"%NumPatches)
    print("Patches per Image: %d"%PatchePerImage)

    if(SavePatches):
        os.makedirs(SavePath, exist_ok=True)
        print("Saving genereated patches to: %s"%SavePath)

    H4_data = np.empty([0, 4, 2])
    H_AB_data = np.empty([0, 3, 3])
    rho_data = np.empty([0, 4, 1])

    PatchCount = 1
    for img in tqdm(range(1, len(DatasetPath)+1)):
        for patch in range(1, PatchePerImage+1):
            PatchSet, H4, H_AB, rho = generatePatchSet(DatasetPath[img-1], Resize, PatchSize, MaxPerturbation, Tolerance, SavePatches, SavePath, PatchCount, Visualize)
            H4_data = np.insert(H4_data, PatchCount-1, H4, axis=0)
            H_AB_data = np.insert(H_AB_data, PatchCount-1, H_AB, axis=0)
            rho_data = np.insert(rho_data, PatchCount-1, rho, axis=0)
            PatchCount += 1

    with open (os.path.join(SavePath, 'H4.npy'), 'wb') as f:
        np.save(f, H4_data)
    with open (os.path.join(SavePath, 'H_AB.npy'), 'wb') as f:
        np.save(f, H_AB_data)
    with open (os.path.join(SavePath, 'rho.npy'), 'wb') as f:
        np.save(f, rho_data)

    with open (os.path.join(SavePath, 'H4.npy'), 'rb') as f:
        H4_data_test = np.load(f)
    with open (os.path.join(SavePath, 'H_AB.npy'), 'rb') as f:
        H_AB_data_test = np.load(f)
    with open (os.path.join(SavePath, 'rho.npy'), 'rb') as f:
        rho_data_test = np.load(f)

    print("H4 Data Test", H4_data_test)
    print("H_AB Data Test", H_AB_data_test)
    print("rho Data Test", rho_data_test)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DatasetPath', type=str, default="../Data/Train/", help='Path of the ImageA Dataset')
    Parser.add_argument('--Resize', default=[240, 320], help='Target size of genereate images (height, width)')
    Parser.add_argument('--PatchSize', default=[128, 128], help='Target size of the generated patches')
    Parser.add_argument('--MaxPerturbation', type=int, default=32, help='Maximum perturbation to generate random homography estimates')
    Parser.add_argument('--Tolerance', type=int, default=10, help='Tolerance of patch center selection')
    Parser.add_argument('--NumPatches', type=int, default=50000, help='Total number of patches to be generated')
    Parser.add_argument('--SavePatches', type=bool, default=False, help='Toggle to save generated patches')
    Parser.add_argument('--SavePath', type=str, default="../Data/Patches/Train", help='Path to store the generated patches')
    Parser.add_argument('--GenerateDataset', type=bool, default=False, help='Toggle to generate dataset or single patch set')
    Parser.add_argument('--Visualize', type=bool, default=False, help='Toggle to visualize outputs')

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
    GenerateDataset = Args.GenerateDataset
    Visualize = Args.Visualize

    if(GenerateDataset):
        generateDataset(readImageSet(DatasetPath), Resize, PatchSize, MaxPerturbation, Tolerance, NumPatches, SavePatches, SavePath, Visualize)
    else:
        generatePatchSet(readImageSet(DatasetPath)[0], Resize, PatchSize, MaxPerturbation, Tolerance, SavePatches, SavePath, 0, Visualize)

if __name__ == '__main__':
    main()