#!/bin/bash

#----------------------------------------------------------------------------------
#                               PARAMETERS:
#----------------------------------------------------------------------------------
PATH_IMAGES="datasets/images"                   #Folder with the images
PATH_REGIONS="datasets/regions"                 #Folder with the mask regions
PATH_BACKGROUND="datasets/layers/background"    #Folder with the background ground-truth data.
PATH_LAYERS=("datasets/layers/staff" "datasets/layers/neumes")                                              #List of folders of ground-truth data for each layer
OUTPUT_MODEL=("Models/model_background.hdf5" "Models/model_staff.hdf5" "Models/model_neumes.hdf5")          #List of paths for the output models
WINDOW_WIDTH=256                                #Width of the window to extract samples
WINDOW_HEIGHT=256                               #Height of the window to extract samples
BATCH_SIZE=8                                    #Batch size
MAX_EPOCHS=50                                   #Maximum number of epochs to be considered. The model will stop before if the the training process does not improve the results.
NUMBER_SAMPLES_PER_CLASS=1000                   #Number of samples to be extracted for each layer.
PATIENCE=15                                     #Number of consecutive epochs allowed without improvement. The training is stopped if the model does not improve this number of consecutive epochs. 
FILE_SELECTION_MODE="SHUFFLE"                   #Mode of the selection of the files in the training process. [RANDOM, SHUFFLE, DEFAULT]
SAMPLE_EXTRACTION_MODE="RANDOM"                 #Mode of extraction of samples. [RANDOM, SEQUENTIAL]

#----------------------------------------------------------------------------------

PARAM_PATH_LAYERS=""
for i in "${PATH_LAYERS[@]}" ; do
    PARAM_PATH_LAYERS+="-pgt "${i}" "
done
echo $PARAM_PATH_LAYERS


PARAM_OUTPUT_MODEL=""
for i in "${OUTPUT_MODEL[@]}" ; do
    PARAM_OUTPUT_MODEL+="-out "${i}" "
done
echo $PARAM_OUTPUT_MODEL

python -u fast_calvo_easy_training.py \
                    -psr   ${PATH_IMAGES} \
                    -prg   ${PATH_REGIONS} \
                    -pbg   ${PATH_BACKGROUND} \
                    ${PARAM_PATH_LAYERS} \
                    ${PARAM_OUTPUT_MODEL} \
                    -width ${WINDOW_WIDTH} \
                    -height ${WINDOW_HEIGHT} \
                    -b ${BATCH_SIZE} \
                    -e ${MAX_EPOCHS} \
                    -n ${NUMBER_SAMPLES_PER_CLASS} \
                    -pat ${PATIENCE} \
                    -fm ${FILE_SELECTION_MODE} \
                    -sm ${SAMPLE_EXTRACTION_MODE}
