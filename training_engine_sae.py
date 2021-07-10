from __future__ import division

import cv2
import numpy as np
import random as rd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, UpSampling2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import image_data_format
import keras
import tensorflow as tf


# ===========================
#       SETTINGS
# ===========================

# gpu_options = tf.GPUOptions(
#     allow_growth=True,
#     per_process_gpu_memory_fraction=0.40
# )
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.tensorflow_backend.set_session(sess)
VALIDATION_SPLIT=0.2
# BATCH_SIZE = 16

# ===========================

def get_input_shape(height, width, channels = 3):
    if image_data_format() == 'channels_first':
        return (channels, height, width)
    else:
        return (height, width, channels)


def get_sae(height, width, pretrained_weights = None):
    ff = 32

    inputs = Input(shape=get_input_shape(height,width))
    conv1 = Conv2D(ff, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(ff, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(ff * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv7 = Conv2D(ff * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(ff * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = Concatenate(axis = 3)([conv2,up8])

    conv8 = Conv2D(ff * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(ff * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(ff * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = Concatenate(axis = 3)([conv1,up9])

    conv9 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(ff * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


def createGenerator(grs, gts, idx_label, patch_height, patch_width, batch_size):
    
    selected_page_idx = 0
    
    while(True):

        gr = grs[selected_page_idx]
        gt = gts[selected_page_idx][idx_label]
        selected_page_idx = np.random.randint(len(gr))

        # Compute where there is information of this layer
        x_coords, y_coords = np.where(gt == 1)
        coords_with_info = (x_coords, y_coords)

        gr_chunks = []
        gt_chunks = []

        num_coords = len(coords_with_info[0])

        index_coords_selected = [random.randint(0, num_coords) for _ in range(batch_size)]
        x_coords = coords_with_info[0][index_coords_selected]
        y_coords = coords_with_info[1][index_coords_selected]
        
        for i in range(batch_size):
            row = x_coords[i]
            col = y_coords[i]
            gr_sample = gr_img[row:row+patch_height, col:col+patch_width]
            gt_sample = gt_img[row:row+patch_height, col:col+patch_width]
            gr_chunks.append(gr_sample)
            gt_chunks.append(gt_sample)

        yield gr_chunks_arr, gt_chunks_arr


def getTrain(input_images, gts, num_labels, patch_height, patch_width, batch_size):
    generator_labels = []

    print('num_labels', num_labels)
    for idx_label in range(num_labels):
        print('idx_label', idx_label)
        generator_label = createGenerator(input_images, gts, idx_label, patch_height, patch_width, batch_size)
        generator_labels.append(generator_label)
        print(generator_labels)

    return generator_labels


def train_msae(input_images, gts, num_labels, height, width, output_path, epochs, max_samples_per_class, batch_size=16):

    # Create ground_truth
    print('Creating data generators...')
    generators = getTrain(input_images, gts, num_labels, height, width, batch_size)

    # Training loop
    for label in range(num_labels):
        print('Training a new model for label #{}'.format(str(label)))
        model = get_sae(
            height=height,
            width=width
        )

        model.summary()
        callbacks_list = [
            ModelCheckpoint(output_path['%s' % label], save_best_only=True, monitor='val_accuracy', verbose=1, mode='max'),
            EarlyStopping(monitor='val_accuracy', patience=3, verbose=0, mode='max')
        ]

        # Training stage
        model.fit_generator(
            generators[label],
            verbose=2,
            steps_per_epoch=max_samples_per_class//batch_size,
            validation_data=generators[label],
            validation_steps=100,
            callbacks=callbacks_list,
            epochs=epochs
        )

    return 0


# Debugging code
if __name__ == "__main__":
    print('Must be run from Rodan')
