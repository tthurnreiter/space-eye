
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, Model, models, optimizers, metrics, callbacks, regularizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
from functools import partial
from sklearn.metrics import confusion_matrix
from pathlib import Path
import keras_tuner as kt


# download the data

train_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
validation_url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'

train_dir = '/.../training'
validation_dir = '/.../validation'

# copy the dataset from online storage to the local files folder
tf.keras.utils.get_file('training', train_url, extract=True, cache_subdir=train_dir)
tf.keras.utils.get_file('validation', validation_url, extract=True, cache_subdir=validation_dir)


# first we'll instantiate the data generator, including real-time image preprocessing and data augmentation
data_train_gen = image.ImageDataGenerator(rescale=1./255.,
                                          zoom_range=0.1,
                                          horizontal_flip=True)

data_validation_gen = image.ImageDataGenerator(rescale=1./255.)  # the validation data should not be augmented

# then, we use flow_from_directory to create a training and validation generator
train_gen = data_train_gen.flow_from_directory(train_dir,
                                         target_size=(300,300),  # default target size is 128 x 256
                                         class_mode='binary')  # determines the type of labels that are created

validation_gen = data_validation_gen.flow_from_directory(validation_dir,
                                              target_size=(300,300),  # default target size is 256 x 256
                                              class_mode='binary')


def build_model(hp):

  learning_rate = hp.Float('learning_rate', 0.00001, 0.001, sampling='log')  # sample from log space
  l2_kernel_c = hp.Float('l2_kernel_c', 0.00001, 0.01, sampling='log')
  l2_kernel_d = hp.Float('l2_kernel_d', 0.00001, 0.01, sampling='log')
  num_kernels_1 = hp.Choice('num_kernels_1', [32, 64, 128])  # sample from different choices
  num_kernels_2 = hp.Choice('num_kernels_2', [64, 128, 256])
  num_dense = hp.Choice('num_dense', [256, 512, 1024])

  adam = optimizers.RMSprop(learning_rate=learning_rate)

  denseLayer = partial(layers.Dense, use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_kernel_d))

  convLayer = partial(layers.Conv2D, kernel_size=3, use_bias=False, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(l2_kernel_c))

  batchLayer = partial(layers.BatchNormalization, center=True, scale=False)  # scale can be disabled when using ReLU

  poolLayer = partial(layers.MaxPooling2D, pool_size=(2, 2))

  relu = partial(layers.Activation, activation='relu')

  # network architecture
  input_ = layers.Input(shape=(300,300,3))

  # conv part
  conv_1_1 = convLayer(filters=num_kernels_1)(input_)
  batch_1_1 = batchLayer()(conv_1_1)
  act_1_1 = relu()(batch_1_1)

  conv_1_2 = convLayer(filters=num_kernels_1)(act_1_1)
  batch_1_2 = batchLayer()(conv_1_2)
  act_1_2 = relu()(batch_1_2)
  pool_1 = poolLayer()(act_1_2)

  conv_2_1 = convLayer(filters=num_kernels_1)(pool_1)
  batch_2_1 = batchLayer()(conv_2_1)
  act_2_1 = relu()(batch_2_1)

  conv_2_2 = convLayer(filters=num_kernels_1)(act_2_1)
  batch_2_2 = batchLayer()(conv_2_2)
  act_2_2 = relu()(batch_2_2)

  conv_2_3 = convLayer(filters=num_kernels_1)(act_2_2)
  res_1 = layers.Add()([pool_1, conv_2_3])
  batch_res_1 = batchLayer()(res_1)
  act_res_1 = relu()(batch_res_1)
  pool_2 = poolLayer()(act_res_1)

  conv_3_1 = convLayer(filters=num_kernels_2)(pool_2)
  batch_3_1 = batchLayer()(conv_3_1)
  act_3_1 = relu()(batch_3_1)

  conv_3_2 = convLayer(filters=num_kernels_2)(act_3_1)
  batch_3_2 = batchLayer()(conv_3_2)
  act_3_2 = relu()(batch_3_2)

  conv_3_3 = convLayer(filters=num_kernels_2)(act_3_2)
  res_input_2 = convLayer(kernel_size=1, filters=num_kernels_2)(pool_2)
  res_2 = layers.Add()([res_input_2, conv_3_3])
  batch_res_2 = batchLayer()(res_2)
  act_res_2 = relu()(batch_res_2)
  pool_3 = poolLayer()(act_res_2)

  flat = layers.Flatten()(pool_3)
  drop_flat = layers.Dropout(0.2)(flat)

  dense_1 = denseLayer(num_dense)(drop_flat)
  batch_d_1 = batchLayer()(dense_1)
  act_d_1 = relu()(batch_d_1)
  drop_1 = layers.Dropout(0.2)(act_d_1)

  dense_2 = denseLayer(num_dense)(drop_1)
  batch_d_2 = batchLayer()(dense_2)
  act_d_2 = relu()(batch_d_2)
  drop_2 = layers.Dropout(0.2)(act_d_2)

  output = denseLayer(1, activation='sigmoid')(drop_2)

  model = Model(inputs=[input_], outputs=[output])

  model.compile(loss="binary_crossentropy", optimizer=adam,
              metrics=["accuracy", metrics.Precision(), metrics.Recall()])

  return model

tf.keras.backend.clear_session()
log_dir = os.path.join('/.../tensorboard', time.strftime('%d_%m_%y-%H_%M'))

# to start tensorboard via console within the folder >> tensorboard --logdir=./tensorboard --port=6006
tensorboard_cb = callbacks.TensorBoard(log_dir)
checkpoint_cb = callbacks.ModelCheckpoint('/.../model', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping_cb = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)  # alternativ 'val_loss'
epochs = 30

tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=50)

print(tuner.search_space_summary(extended=True))

tuner.search(train_gen, epochs=epochs,
             validation_data=validation_gen,
             callbacks=[early_stopping_cb, tensorboard_cb], verbose=2)

print(tuner.results_summary())

