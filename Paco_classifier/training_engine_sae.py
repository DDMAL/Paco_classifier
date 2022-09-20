from __future__ import division

import os	
import logging	

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dropout, UpSampling2D, Concatenate	
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Masking	
from tensorflow.keras.optimizers import Adam	
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint	
from tensorflow.keras.backend import image_data_format	
import tensorflow as tf

from .data_loader import FileSelectionMode, SampleExtractionMode, getTrain

def get_sae(height, width, pretrained_weights=None):
    ff = 32
    channels = 3
    kPIXEL_VALUE_FOR_MASKING = -1

    img_shape = (height, width, channels)
    if image_data_format() == "channels_first":
        img_shape = (channels, height, width)

    inputs = Input(shape=img_shape)
    mask = Masking(mask_value=kPIXEL_VALUE_FOR_MASKING)(inputs)

    conv1 = Conv2D(
        ff, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(mask)
    conv1 = Conv2D(
        ff, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(
        ff * 2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool1)
    conv2 = Conv2D(
        ff * 2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(
        ff * 8, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(pool2)
    conv7 = Conv2D(
        ff * 8, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv3)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(
        ff * 4, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(up8)
    merge8 = Concatenate(axis=3)([conv2, up8])

    conv8 = Conv2D(
        ff * 4, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge8)
    conv8 = Conv2D(
        ff * 4, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(
        ff * 2, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(up9)
    merge9 = Concatenate(axis=3)([conv1, up9])

    conv9 = Conv2D(
        ff * 2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(merge9)
    conv9 = Conv2D(
        ff * 2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(
        optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
    )

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model

def get_stride(patch_height, patch_width):
    return patch_height // 2, patch_width // 2

def get_steps_per_epoch(inputs, number_samples_per_class, patch_height, patch_width, batch_size, sample_extraction_mode):

    if sample_extraction_mode == SampleExtractionMode.RANDOM:
        return number_samples_per_class // batch_size
    elif sample_extraction_mode == SampleExtractionMode.SEQUENTIAL:
        hstride, wstride = get_stride(patch_height, patch_width)
        number_samples = 0

        for idx_file in range(len(inputs["Image"])):
            gr = inputs["Image"][0][idx_file]
            number_samples += ((gr.shape[0] - patch_height) // hstride) * ((gr.shape[1] - patch_width) // wstride)

        return number_samples // batch_size
    else:
        raise Exception(
            'The sample extraction mode does not exist.\n'
        )    

def train_msae(
    inputs,
    num_labels,
    height,
    width,
    output_path,
    file_selection_mode,
    sample_extraction_mode,
    epochs,
    number_samples_per_class,
    batch_size=16,
    patience=15,
    models=None
):

    # Create ground_truth
    print("Creating data generators...")
    generators = getTrain(inputs, height, width, batch_size, file_selection_mode, sample_extraction_mode)
    generators_validation = getTrain(inputs, height, width, batch_size, FileSelectionMode.DEFAULT, SampleExtractionMode.RANDOM)

    # Training loop
    for label in range(num_labels):
        print("Training a new model for label #{}".format(str(label)))
        # Pretrained weights
        model_name = "Background Model" if label == 0 else "Model {}".format(label)
        if models and model_name in models:
            model = load_model(models[model_name][0]['resource_path'])
        else:
            model = get_sae(height=height, width=width)
        new_output_path = os.path.join(output_path[str(label)])
        callbacks_list = [
            ModelCheckpoint(
                new_output_path,
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1,
                mode="max",
            ),
            EarlyStopping(monitor="val_accuracy", patience=patience, verbose=0, mode="max"),
        ]

        steps_per_epoch = get_steps_per_epoch(inputs, number_samples_per_class, height, width, batch_size, sample_extraction_mode)

        # Training stage
        model.fit(
            generators[label],
            verbose=2,
            steps_per_epoch=steps_per_epoch,
            validation_data=generators_validation[label],
            validation_steps=len(inputs.meta["Image"]),
            callbacks=callbacks_list,
            epochs=epochs
        )
        
        os.rename(new_output_path, output_path[str(label)])

    return 0


# Debugging code
if __name__ == "__main__":
    print("Must be run from Rodan")
