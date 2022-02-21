#!/usr/env/bin python3

"""
CMSC733 Spring 2021: Classical and Deep Learning Approaches for Geometric Computer Vision
Project 1: MyAutoPano: Phase 2


Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing modules
import time
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import sys
import argparse
import cv2
from PIL import Image
import pandas as pd
import traceback
import seaborn as sns
from sklearn.metrics import confusion_matrix

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam
from tensorflow.train import Checkpoint, CheckpointManager

from Network.HomographyNet import HomographyNet
from Helper import *
from Misc.MiscUtils import *
from GenerateData import *

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()


class Test():

    def __init__(self, Model, DatasetPath, BatchSize, CheckpointPath):
        self.Model = Model
        self.DatasetPath = DatasetPath
        self.BatchSize = BatchSize
        self.CheckpointPath = CheckpointPath

        self.TestingData = None

    def evaluate(self):

        running_loss = 0.0
        inference_time = 0.0

        for batch in tqdm(range(int(len(self.PatchA_testing_data)/self.BatchSize))):
            X = np.empty([0, 128, 128, 2]) # Input to the network
            y = np.empty([0, 8]) # Ground Truth

            for index in range(self.BatchSize):
                X = np.insert(X, index, np.dstack((preprocess_image(load_image(self.PatchA_testing_data[(batch*self.BatchSize)+index])), preprocess_image(load_image(self.PatchB_testing_data[(batch*self.BatchSize)+index])))), axis=0)
                y = np.insert(y, index, preprocess_H4_data(self.H4_testing_data[(batch*self.BatchSize)+index]), axis=0)

            tick = time.time()

            y_ = self.Model(X, training=False) # Prediction of the network
            toc = time.time()
            inference_time += (toc - tick)
            
            loss = L2_loss(y, y_) # L2 loss between prediction (y_) and ground truth (y)

            running_loss += loss

            result = generateResult(self.TestingData[batch], deprocess_H4_data(y_[0].numpy()), self.H_AB_testing_data[batch], [240, 320], [128,128], 32, True)
        # cv2.imwrite(os.path.join(self.ResultsPath, 'Val', '%d.png'%self.Epoch), result)
        running_loss = running_loss/(len(self.PatchA_testing_data)/self.BatchSize) # Mean loss of the epoch
        print("Testing Loss: %f"%(running_loss))
        inference_time = inference_time/(len(self.PatchA_testing_data)/self.BatchSize)
        print("Average Inference Time: %f"%(inference_time))

        return running_loss, inference_time

    def test(self):

        print("GPU %d"%tf.test.is_gpu_available()) # Checking GPU availability

        self.Model.summary() # Summary of the model

        if(self.CheckpointPath):
            self.TrainingDir = os.path.join('../Training', self.CheckpointPath) # Initializing the training directory for current training instance
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)
        else:
            print("CHECKPOINT DIRECTORY NOT FOUND!")

        learning_rate = 1E-4 # Learning rate of 0.005 from the paper
        self.Optimizer = Adam(learning_rate=learning_rate)

        # Creating a Checkpoint Manager
        ckpt = Checkpoint(step=tf.Variable(0), model=self.Model, optimizer=self.Optimizer)
        ckpt_manager = CheckpointManager(ckpt, self.CheckpointPath, max_to_keep=None)

        # Loading Testing Data
        self.TestingData = readImageSet('../Data/Test/')
        self.PatchA_testing_data = readImageSet(os.path.join(self.DatasetPath, 'Test', 'PatchA')) # Gettings relative paths to PatchA testing images
        self.PatchB_testing_data = readImageSet(os.path.join(self.DatasetPath, 'Test', 'PatchB')) # Gettings relative paths to PatchB testing images
        with open (os.path.join(self.DatasetPath, 'Test', 'H4.npy'), 'rb') as f:
            self.H4_testing_data = np.load(f)
        with open (os.path.join(self.DatasetPath, 'Test', 'H_AB.npy'), 'rb') as f:
            self.H_AB_testing_data = np.load(f)
        
        ckpt.restore(ckpt_manager.latest_checkpoint) # Restoring checkpoint if available
        if(ckpt_manager.latest_checkpoint):
            self.Model = ckpt.model # Restoring model state
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
            self.Optimizer.lr.assign(ckpt.optimizer.lr) # Restoring learning rate of the optimizer
            print("Restored Learnign Rate: {}".format(self.Optimizer.lr))
        else:
            print("Initializing from scratch.")

        start = time.time() # Training instance start time

        try:
            testing_epoch_loss, testing_inference_time = self.evaluate() # Test the model

            end = time.time()

            print("Took %.03f minutes to test"%((end-start)/60))
        except KeyboardInterrupt:
            print(traceback.format_exc())

            end = time.time()

            print("Took %.03f minutes to test"%((end-start)/60))


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', type=str, default="Supervised", help='Select between Supervised and Unsupervised model for testing (Supervised, Unsupervised)')
    Parser.add_argument('--DatasetPath', type=str, default="../Data/Patches_50000/", help='Path to the testing dataset')
    Parser.add_argument('--BatchSize', type=int, default=1, help='batch_size')
    Parser.add_argument('--CheckpointPath', type=str, default=None, help='Checkpoint for testing')

    Args = Parser.parse_args()
    ModelType = Args.ModelType # Model to test
    DatasetPath = Args.DatasetPath # Path to the dataset folder containing 'Train' and 'Val' folders
    BatchSize = Args.BatchSize # Batchsize to be used in training
    CheckpointPath = Args.CheckpointPath # Folder name of trianing instance in the TrainingDir 

    if(ModelType == "Supervised"):
        Model = HomographyNet() # Creating an instance of the HomographyNet model
        test_model = Test(Model, DatasetPath, BatchSize, CheckpointPath)
        test_model.test()

if __name__ == '__main__':
    main()