import os
import re
import glob
import json
import shutil
import cv2 as cv
import numpy as np
import keras as ks
import tensorflow as tf 
import more_itertools as mit
import sklearn.model_selection as sks
import keras.utils as ku
import keras.layers as kl
from keras import backend as bk

params = {
    # Neural network parameters
    'batch_size': 256,
    'learn_rate': 0.0001,
    'decay': 0.001,
    'epochs': 100,
    'dropout': 0.75,
    'filters': 8,
    'dense': 256,
    'activation': 'relu',
    'patience': 5,

    # Animal to train on
    'animal': 'dog',

    # Seeds for train/test splits
    'seeds': [1, 2, 3, 4, 5]

}


ks.backend.tensorflow_backend.set_session(
    tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
)

model = ks.models.Sequential(
[kl.InputLayer(input_shape=(640, 640, 3))] +
list(mit.flatten([
    [
        kl.Conv2D(
            filters=params['filters'] * (2**n), kernel_size=(3, 3),
            activation=params['activation'], padding='same'
        ),
        kl.MaxPooling2D(pool_size=(2, 2))
    ] for n in range(7)
])) +
[kl.Flatten()] +
list(mit.flatten([
    [
        kl.Dense(params['dense'], activation=params['activation']),
        kl.Dropout(rate=params['dropout'])
    ] for n in range(2)
])) +
[kl.Dense(2, activation='softmax')]
)

model.compile(
    optimizer=ks.optimizers.Adam(lr=params['learn_rate'], decay=params['decay']),
    loss=ks.losses.binary_crossentropy, metrics=['accuracy']
)

model.summary()

bk.clear_session()


print('DONE')


