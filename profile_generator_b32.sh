#!/bin/bash

#----------------------------------------------------------------------------------
#                               PARAMETERS:
#----------------------------------------------------------------------------------
PATH_IMAGES="dataset/MS73/training/images"
PATH_REGIONS="dataset/MS73/training/regions"
PATH_BACKGROUND="dataset/MS73/training/layers/bg"
PATH_LAYERS=("dataset/MS73/training/layers/neumes" "dataset/MS73/training/layers/staff")

OUTPUT_MODEL=("Results/model_background.hdf5" "Results/model_staff.hdf5" "Results/model_neumes.hdf5")          #List of paths for the output models
WINDOW_WIDTH=256                                #Width of the window to extract samples
WINDOW_HEIGHT=256                               #Height of the window to extract samples
BATCH_SIZE=32                                        #Batch size
MAX_EPOCHS=50
NUMBER_SAMPLES_PER_CLASS=1000
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

python -u profileMain.py \
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

