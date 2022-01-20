#!/usr/env/bin python3

# Importing modules
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import sys
import numpy as np

from tensorflow.keras.models import Model

# Don't generate pyc codes
sys.dont_write_bytecode = True


def HomographyNet(InputShape=(128,128,2), Filters=64, KernelSize=(3,3), PoolSize=(2,2), Strides=2):
    '''
        Convolutional Neural Network (CNN) based on the HomographyNet put forward in - https://arxiv.org/pdf/1606.03798.pdf
    '''

    inputs = Input(shape=InputShape)

    x = Conv2D(filters=Filters, kernel_size=KernelSize, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=Filters, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=PoolSize, strides=Strides, padding='same')(x)


    x = Conv2D(filters=Filters, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=Filters, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=PoolSize, strides=Strides, padding='same')(x)


    x = Conv2D(filters=Filters*2, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=Filters*2, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=PoolSize, strides=Strides, padding='same')(x)


    x = Conv2D(filters=Filters*2, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=Filters*2, kernel_size=KernelSize, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(8)(x)

    outputs = x

    model = Model(inputs=inputs, outputs=outputs, name='HomographyNet')

    return model


def main():
    HM = HomographyNet()
    HM.summary()


if __name__ == '__main__':
    main()