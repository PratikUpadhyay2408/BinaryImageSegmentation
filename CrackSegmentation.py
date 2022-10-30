import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import Unet
from DataLoader import DataLoader

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K

""" Global dataset parameters """
TRAIN_IMG_DIR = r'G:\AI\BinaryImageSegmentation\Data\train\images'
TRAIN_MASK_DIR = r'G:\AI\BinaryImageSegmentation\Data\train\masks'
TEST_IMG_DIR = r'G:\AI\BinaryImageSegmentation\Data\test\images'
TEST_MASK_DIR= r'G:\AI\BinaryImageSegmentation\Data\test\masks'
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 64
NUM_CLASSES = 1
IMG_HEIGHT = 128
IMG_WIDTH  = 128
CHANNELS   = 3


Train = DataLoader(
                    root_img_dir = TRAIN_IMG_DIR,
                    root_mask_dir = TRAIN_MASK_DIR,
                    width = IMG_WIDTH,
                    height = IMG_HEIGHT,
                    batch_size = BATCH_SIZE,
                    shuffle_buffer_size= SHUFFLE_BUFFER_SIZE
                   )

Test = DataLoader(
                    root_img_dir = TEST_IMG_DIR,
                    root_mask_dir = TEST_MASK_DIR,
                    width = IMG_WIDTH,
                    height = IMG_HEIGHT,
                    batch_size = BATCH_SIZE,
                    shuffle_buffer_size= SHUFFLE_BUFFER_SIZE
                  )


Train_data = Train.makedataset().take(500)
Val_data = Train_data.take(100)
Test_data = Test.makedataset()



lr = 1e-4
epochs = 10


model = Unet.unet_model( (IMG_HEIGHT, IMG_WIDTH, CHANNELS),32, NUM_CLASSES)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1)
]

model.fit(
           Train_data,
           batch_size=BATCH_SIZE,
           shuffle=True,
           epochs=epochs,
           callbacks=callbacks,
           validation_data=Val_data
          )