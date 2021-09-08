#!/bin/bash

#python -u py_dasae.py -path datasets -db1 sal -db2 dibco2016 -s 128 -l 5 -f 128 -gpu 0
#python -u py_dasae.py -path datasets -db1 dibco2016 -db2 palm0
#exit

#gpu=0

PATH_IMAGES="datasets/images"
PATH_REGIONS="datasets/regions"
PATH_BACKGROUND="datasets/layers/bg"
PATH_LAYERS=("datasets/layers/staff" "datasets/layers/neumes")
OUTPUT_MODEL=("Images/model0.hdf5" "Images/model1.hdf5" "Images/model2.hdf5")
WINDOW_WIDTH=64
WINDOW_HEIGHT=64
BATCH_SIZE=8
MAX_EPOCHS=5
NUMBER_SAMPLES_PER_CLASS=100
FILE_SELECTION_MODE="SHUFFLE"
SAMPLE_EXTRACTION_MODE="RANDOM"


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
                    -fm ${FILE_SELECTION_MODE} \
                    -sm ${SAMPLE_EXTRACTION_MODE}
