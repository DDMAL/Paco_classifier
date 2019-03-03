import cv2
import numpy as np
import random as rd
from keras.models import Model
from keras.layers import Dropout, UpSampling2D, Concatenate
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.backend import image_data_format

# ===========================
#       SETTINGS
# ===========================

VALIDATION_SPLIT=0.2
BATCH_SIZE = 16

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


def getTrain(input_image, gt, patch_height, patch_width):
    X_train = []
    Y_train = {}

    # Initialize GT lists
    for key in gt:
        Y_train[key] = []

    hstride = patch_height // 2
    wstride = patch_width // 2

    # TODO Take into account margins
    for h in range(0, input_image.shape[0] - patch_height, hstride):
        for w in range(0, input_image.shape[1] - patch_width, wstride):
            x_sample = input_image[h:h + patch_height, w:w + patch_width]

            # Pre-process TODO: Check all the pipelines to do the same
            x_sample = (255. - x_sample) / 255.

            X_train.append(x_sample)

            for label in gt:
                Y_train[label].append(gt[label][h:h + patch_height, w:w + patch_width])

    return X_train, Y_train


def train_msae(input_image, gt, height, width, output_path, epochs):

    # Crop the image and create ground_truth
    [X_train, Y_train] = getTrain(input_image, gt,
                                  height, width)

    X_train = np.asarray(X_train)

    print 'Training created with ' + str(len(X_train)) + ' samples.'
    for label in Y_train:

        # Training configuration
        print 'Training a new model for ' + str(label)
        model = get_sae(
            height=height,
            width=width
        )

        model.summary()

        callbacks_list = [
                ModelCheckpoint(output_path[label], save_best_only=True, monitor='val_acc', verbose=1, mode='max'),
                EarlyStopping(monitor='val_acc', patience=3, verbose=0, mode='max')
                ]

        Y_train_label = np.expand_dims(np.asarray(Y_train[label]),axis=-1)

        print(X_train.shape)
        print(Y_train_label.shape)
        # Training stage
        model.fit(X_train, Y_train_label,
                  verbose=2,
                  batch_size=BATCH_SIZE,
                  validation_split=VALIDATION_SPLIT,
                  callbacks=callbacks_list,
                  epochs=epochs)

    return 0


# Debugging code
if __name__ == "__main__":
    print('Must be run from Rodan')


