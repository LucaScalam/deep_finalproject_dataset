from __future__ import print_function
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Conv2DTranspose, Activation
from metrics_extra import seg_metrics, mean_iou, mean_dice
from keras.optimizers import Adam
from keras import backend as K
from generator_v2 import data_generator
from callbacks import TrainCheck
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np



dir_path = os.path.dirname(os.path.realpath('__file__'))

# Parametros para el cluster
parser = argparse.ArgumentParser()
parser.add_argument("-TB", "--train_batch", type=int,required=False, default=4)
parser.add_argument("-VB", "--val_batch", type=int,required=False, default=1)
parser.add_argument("-LI", "--lr_init", type=float,required=False, default=1e-4)
parser.add_argument("-LD", "--lr_decay", type=float, required=False, default=5e-4)
args = parser.parse_args()
TRAIN_BATCH = args.train_batch
VAL_BATCH = args.val_batch
lr_init = args.lr_init
lr_decay = args.lr_decay

#8 clases
labels = ['background', 'person', 'car', 'road',
'sidewalk', 'vegetation', 'building', 'sky']


input_shape=(256, 512, 3)

input_model = Input(input_shape)

#256 - block1
x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(input_model)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
x = BatchNormalization()(x)
block1 = Activation('relu')(x)
x = MaxPooling2D()(block1)
#128 - block2
x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
x = BatchNormalization()(x)
block2 = Activation('relu')(x)
x = MaxPooling2D()(block2)
#64 - block3
x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
x = BatchNormalization()(x)
block3 = Activation('relu')(x)
x = MaxPooling2D()(block3)
#32 - block4
x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
x = BatchNormalization()(x)
block4 = Activation('relu')(x)
x = MaxPooling2D()(block4)
#16 - block5
x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#32(despues del convTrans) - block6
x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = concatenate([x, block4])
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#64 - block7
x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = concatenate([x, block3])
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#128 - block8
x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = concatenate([x, block2])
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
#256 - block9
x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = concatenate([x, block1])
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

#softmax
x = Conv2D(len(labels), (3, 3), activation='softmax', padding='same')(x)

model = Model(input_model, x)
model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
              loss='categorical_crossentropy',
              metrics=['accuracy',mean_iou,mean_dice])




checkpoint = ModelCheckpoint(filepath='7_model.h5',
                             monitor='val_accuracy',
                             save_best_only=True,
                             verbose = 1)
model_name = 'unet_final'
train_check = TrainCheck(output_path='./img', model_name=model_name)


history = model.fit(data_generator('data/data.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=2780 // TRAIN_BATCH,
                              validation_data=data_generator('data/data.h5', VAL_BATCH, 'val'),
                              validation_steps=347 // VAL_BATCH,
                              callbacks=[checkpoint, train_check],
                              epochs=120,
                              verbose=1)

np.save('7.npy',history.history)
