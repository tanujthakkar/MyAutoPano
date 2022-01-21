#!/usr/env/bin python3

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

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.train import Checkpoint, CheckpointManager

from Network.HomographyNet import HomographyNet
from Helper import *
from Misc.MiscUtils import *



def TrainSupervised(DatasetPath, Epochs, BatchSize, TrainingDir, CheckpointPath, CheckpointEpoch):

    print("GPU %d"%tf.test.is_gpu_available()) # Checking GPU availability

    HM = HomographyNet() # Creating an instance of the HomographyNet model
    # HM.summary() # Summary of the model

    if(CheckpointPath):
        start_epoch = CheckpointEpoch + 1
        TrainingDir = os.path.join('../Training', CheckpointPath)
        CheckpointPath = os.path.join(TrainingDir, 'Checkpoints')
        ResultsPath = os.path.join(TrainingDir, 'Results')
        print("Training Data Path: %s"%TrainingDir)
        print("Checkpoints Path: %s"%CheckpointPath)
        print("Results Path: %s"%ResultsPath)

        StartEpoch = CheckpointEpoch + 1
    else:
        TrainingDir = os.path.join(TrainingDir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(TrainingDir)
        CheckpointPath = os.path.join(TrainingDir, 'Checkpoints')
        os.makedirs(CheckpointPath)
        ResultsPath = os.path.join(TrainingDir, 'Results')
        os.makedirs(ResultsPath)
        print("Training Data Path: %s"%TrainingDir)
        print("Checkpoints Path: %s"%CheckpointPath)
        print("Results Path: %s"%ResultsPath)

        StartEpoch = 1

    learning_rate = 5E-3 # Learning rate of 0.005 from the paper
    momentum = 0.9 # Momentum of SGD from the paper
    opt = SGD(momentum=momentum, learning_rate=learning_rate) # Using Stochastic Gradient Descent as optimizer for training

    # Creating a Checkpoint Manager
    ckpt = Checkpoint(step=tf.Variable(1), model=HM, optimizer=opt)
    ckpt_manager = CheckpointManager(ckpt, CheckpointPath, max_to_keep=None)

    training_data = readImageSet(os.path.join(DatasetPath, 'Train')) # Getting relative paths to training images
    validation_data = readImageSet(os.path.join(DatasetPath, 'Val')) # Gettings relative paths to validation images

    ckpt.restore(ckpt_manager.latest_checkpoint) # Restoring checkpoint if available
    if(ckpt_manager.latest_checkpoint):
        HM = ckpt.model # Restoring model state
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    opt.lr.assign(learning_rate) # Restoring learning rate of the optimizer

    try:
        for epoch in range(StartEpoch, Epochs):
            print("Epoch %d of %d"%((epoch), Epochs))

            if(epoch%30000 == 0):
                learning_rate = learning_rate / 10
                print("Updated Learning Rate: %f"%learning_rate)
                opt.lr.assign(learning_rate)

                fit()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Model', type=str, default="Supervised", help='Select between Supervised and Unsupervised model for training (Supervised, Unsupervised)')
    Parser.add_argument('--DatasetPath', type=str, default="../Data/", help='Path to the training and validation dataset')
    Parser.add_argument('--Epochs', type=int, default=90000, help='Epochs to train the model for')
    Parser.add_argument('--BatchSize', type=int, default=2, help='batch_size')
    Parser.add_argument('--TrainingDir', type=str, default="../Training/", help='Path to save the training and validation results')
    Parser.add_argument('--CheckpointPath', type=str, default=None, help='Checkpoint for inference/resuming training')
    Parser.add_argument('--CheckpointEpoch', type=int, default=0, help='Checkpoint epoch to resume training from')

    Args = Parser.parse_args()
    Model = Args.Model
    DatasetPath = Args.DatasetPath
    Epochs = Args.Epochs
    BatchSize = Args.BatchSize
    TrainingDir = Args.TrainingDir
    CheckpointPath = Args.CheckpointPath
    CheckpointEpoch = Args.CheckpointEpoch

    if(Model == "Supervised"):
        TrainSupervised(DatasetPath, Epochs, BatchSize, TrainingDir, CheckpointPath, CheckpointEpoch)


if __name__ == '__main__':
    main()