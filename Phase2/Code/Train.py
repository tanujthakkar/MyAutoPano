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
import traceback

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.train import Checkpoint, CheckpointManager

from Network.HomographyNet import HomographyNet
from Helper import *
from Misc.MiscUtils import *
from GenerateData import *

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class TrainSupervised():

    def __init__(self, DatasetPath, Epochs, BatchSize, TrainingDir, CheckpointPath=None, CheckpointEpoch=None):
        self.DatasetPath = DatasetPath
        self.Epochs = Epochs + 1
        self.BatchSize = BatchSize
        self.TrainingDir = TrainingDir
        self.CheckpointPath = CheckpointPath
        self.CheckpointEpoch = CheckpointEpoch

        self.Model = None
        self.Optimizer = None
        self.Epoch = 0 # Current epoch of the training operation
        self.TrainingData = None
        self.PatchA_training_data = None
        self.PatchB_training_data = None
        self.H4_training_data = None
        self.H_AB_training_data = None
        self.PatchA_validation_data = None
        self.PatchB_validation_data = None
        self.H4_validation_data = None
        self.H_AB_validation_data = None

    def fit(self):

        running_loss = 0.0

        for batch in tqdm(range(int(len(self.PatchA_training_data[:10])/self.BatchSize))):
            X = np.empty([0, 128, 128, 2]) # Input to the network
            y = np.empty([0, 8]) # Ground Truth

            for index in range(self.BatchSize):
                X = np.insert(X, index, np.dstack((preprocess_image(load_image(self.PatchA_training_data[(batch*self.BatchSize)+index])), preprocess_image(load_image(self.PatchB_training_data[(batch*self.BatchSize)+index])))), axis=0)
                y = np.insert(y, index, preprocess_H4_data(self.H4_training_data[(batch*self.BatchSize)+index]), axis=0)

            with tf.GradientTape() as tape:
                y_ = self.Model(X, training=True) # Prediction of the network
                loss = L2_loss(y, y_) # L2 loss between prediction (y_) and ground truth (y)

            grads = tape.gradient(loss, self.Model.trainable_variables)
            self.Optimizer.apply_gradients(zip(grads, self.Model.trainable_variables))

            running_loss += loss

        print("Epoch Final Prediction: ", y_)
        generateResult(self.TrainingData[2], deprocess_H4_data(y_.numpy()), self.H_AB_training_data[batch], [240, 320], [128,128], 32, True)
        running_loss = running_loss/len(self.PatchA_training_data[:10]) # Mean loss of the epoch
        print("Training Loss: %f"%running_loss)

        return running_loss

    def validate(self):

        running_loss = 0.0

        for batch in tqdm(range(int(len(self.PatchA_validation_data)/self.BatchSize))):
            X = np.empty([0, 128, 128, 2]) # Input to the network
            y = np.empty([0, 8]) # Ground Truth

            for index in range(self.BatchSize):
                X = np.insert(X, index, np.dstack((preprocess_image(load_image(self.PatchA_validation_data[(batch*self.BatchSize)+index])), preprocess_image(load_image(self.PatchB_validation_data[(batch*self.BatchSize)+index])))), axis=0)
                y = np.insert(y, index, preprocess_H4_data(self.H4_validation_data[(batch*self.BatchSize)+index]), axis=0)

            y_ = self.Model(X, training=False) # Prediction of the network
            loss = L2_loss(y, y_) # L2 loss between prediction (y_) and ground truth (y)

            running_loss += loss

        print("Epoch Final Prediction: ", y_)
        running_loss = running_loss/len(self.PatchA_validation_data) # Mean loss of the epoch
        print("Validation Loss: %f"%running_loss)

        return running_loss

    def train(self):

        print("GPU %d"%tf.test.is_gpu_available()) # Checking GPU availability

        HM = HomographyNet() # Creating an instance of the HomographyNet model
        # HM.summary() # Summary of the model
        self.Model = HM

        if(self.CheckpointPath):
            self.TrainingDir = os.path.join('../Training', self.CheckpointPath) # Initializing the training directory for current training instance
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)

            StartEpoch = self.CheckpointEpoch + 1 # Initializing starting epoch of the training operation
        else:
            self.TrainingDir = os.path.join(self.TrainingDir, datetime.now().strftime("%Y%m%d-%H%M%S")) # Initializing the training directory for current training instance
            os.makedirs(self.TrainingDir)
            self.CheckpointPath = os.path.join(self.TrainingDir, 'Checkpoints') # Initializing checkpoint directory of current training instance
            os.makedirs(self.CheckpointPath)
            self.ResultsPath = os.path.join(self.TrainingDir, 'Results') # Initializing results directory of current training instance
            os.makedirs(self.ResultsPath)
            print("Training Data Path: %s"%self.TrainingDir)
            print("Checkpoints Path: %s"%self.CheckpointPath)
            print("Results Path: %s"%self.ResultsPath)

            StartEpoch = 1 # Initializing starting epoch of the training operation

        learning_rate = 1E-3 # Learning rate of 0.005 from the paper
        momentum = 0.9 # Momentum of SGD from the paper
        # opt = SGD(momentum=momentum, learning_rate=learning_rate) # Using Stochastic Gradient Descent as optimizer for training
        opt = Adam(learning_rate=learning_rate)
        self.Optimizer = opt

        # Creating a Checkpoint Manager
        ckpt = Checkpoint(step=tf.Variable(0), model=self.Model, optimizer=self.Optimizer)
        ckpt_manager = CheckpointManager(ckpt, self.CheckpointPath, max_to_keep=None)

        self.TrainingData = readImageSet('../Data/Train/')

        # Loading Traning Data
        self.PatchA_training_data = readImageSet(os.path.join(self.DatasetPath, 'Train', 'PatchA')) # Getting relative paths to PatchA training images
        self.PatchB_training_data = readImageSet(os.path.join(self.DatasetPath, 'Train', 'PatchB')) # Gettings relative paths to PatchB training images
        with open (os.path.join(self.DatasetPath, 'Train', 'H4.npy'), 'rb') as f:
            self.H4_training_data = np.load(f)
        with open (os.path.join(self.DatasetPath, 'Train', 'H_AB.npy'), 'rb') as f:
            self.H_AB_training_data = np.load(f)
        # print(H4_training_data, H4_training_data.shape)

        # Loading Validation Data
        self.PatchA_validation_data = readImageSet(os.path.join(self.DatasetPath, 'Val', 'PatchA')) # Gettings relative paths to PatchA validation images
        self.PatchB_validation_data = readImageSet(os.path.join(self.DatasetPath, 'Val', 'PatchB')) # Gettings relative paths to PatchB validation images
        with open (os.path.join(self.DatasetPath, 'Val', 'H4.npy'), 'rb') as f:
            self.H4_validation_data = np.load(f)
        with open (os.path.join(self.DatasetPath, 'Val', 'H_AB.npy'), 'rb') as f:
            self.H_AB_validation_data = np.load(f)
        # print(H4_validation_data, H4_validation_data.shape)

        ckpt.restore(ckpt_manager.latest_checkpoint) # Restoring checkpoint if available
        if(ckpt_manager.latest_checkpoint):
            self.Model = ckpt.model # Restoring model state
            print("Restored from {}".format(ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.Optimizer.lr.assign(learning_rate) # Restoring learning rate of the optimizer

        start = time.time() # Training instance start time

        x = list()
        y1 = list()
        y2 = list()

        try:
            for epoch in range(StartEpoch, self.Epochs):
                print("Epoch %d of %d"%((epoch), self.Epochs-1))
                self.Epoch = epoch # Updating current epoch

                if(epoch%100 == 0):
                    learning_rate = learning_rate / 2
                    print("Updated Learning Rate: %f"%learning_rate)
                    self.Optimizer.lr.assign(learning_rate)

                training_epoch_loss = self.fit()
                # validation_epoch_loss = self.validate()
                validation_epoch_loss = 0

                ckpt.step.assign_add(1)
                if(epoch%100 == 0):
                    save_path = ckpt_manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                data = [epoch, np.mean(self.Optimizer.lr.numpy()), training_epoch_loss, validation_epoch_loss]
                if(epoch == 1):
                    df = pd.DataFrame([data], columns = ['Epochs', 'Learning Rate','Training Loss', 'Validation Loss'])
                    df.to_csv(os.path.join(self.TrainingDir, 'model_state.csv'), mode='a')
                else:
                    df = pd.DataFrame([data])
                    df.to_csv(os.path.join(self.TrainingDir, 'model_state.csv'), header=False, mode='a')

                x.append(epoch)
                y1.append(training_epoch_loss)
                y2.append(validation_epoch_loss)
                plt.plot(x, y1)
                plt.plot(x, y2)
                plt.pause(0.3)

            plt.show()
            plt.savefig(os.path.join(self.TrainingDir, 'Loss.png'))
            end = time.time()
            print("Took %.03f minutes to train"%((end-start)/60))
        except Exception:
            print(traceback.format_exc())
            save_path = ckpt_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            end = time.time()
            print("Took %.03f minutes to train"%((end-start)/60))


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelType', type=str, default="Supervised", help='Select between Supervised and Unsupervised model for training (Supervised, Unsupervised)')
    Parser.add_argument('--DatasetPath', type=str, default="../Data/Patches/", help='Path to the training and validation dataset')
    Parser.add_argument('--Epochs', type=int, default=90000, help='Epochs to train the model for')
    Parser.add_argument('--BatchSize', type=int, default=1, help='batch_size')
    Parser.add_argument('--TrainingDir', type=str, default="../Training/", help='Path to save the training and validation results')
    Parser.add_argument('--CheckpointPath', type=str, default=None, help='Checkpoint for inference/resuming training')
    Parser.add_argument('--CheckpointEpoch', type=int, default=0, help='Checkpoint epoch to resume training from')

    Args = Parser.parse_args()
    ModelType = Args.ModelType
    DatasetPath = Args.DatasetPath # Path to the dataset folder containing 'Train' and 'Val' folders
    Epochs = Args.Epochs # Total number of epochs to run the training for
    BatchSize = Args.BatchSize # Batchsize to be used in training
    TrainingDir = Args.TrainingDir # Path to the directory where training data has to be stored
    CheckpointPath = Args.CheckpointPath # Folder name of trianing instance in the TrainingDir 
    CheckpointEpoch = Args.CheckpointEpoch # Number of epochs for which training has been completed (will resume from CheckpointEpoch + 1)

    if(ModelType == "Supervised"):
        train_supervised = TrainSupervised(DatasetPath, Epochs, BatchSize, TrainingDir, CheckpointPath, CheckpointEpoch)
        train_supervised.train()

if __name__ == '__main__':
    main()