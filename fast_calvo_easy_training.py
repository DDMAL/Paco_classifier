"""Local Fast Trainer
This is the file for running Calvo Fast Trainer loaclly. Make sure
to have an 'Images' folder with the correct inputs in the same directory.
If not, you can change the values in 'inputs' and 'outputs'.

Simply run `python local_fast_trainer.py` to see the output.
This will call `training_engine_sae.py`.

It should generate 3 files in its current state. A background model,
a Model 0, and a Log File.

If you're running it in a Rodan container, this will be located in code/Rodan/rodan/jobs/Calvo_classifier
If the container is already running, try `docker exec -it [container_name] bash` to run the script without
stopping.
"""

import os
import cv2
import pdb
import sys
import logging
import argparse
import numpy as np

import Paco_classifier.training_engine_sae as training
import Paco_classifier.preprocess as preprocess

from ConfigParser import loadConfig

KEY_BACKGROUND_LAYER = "rgba PNG - Layer 0 (Background)"
KEY_SELECTED_REGIONS = "rgba PNG - Selected regions"
KEY_RESOURCE_PATH = "resource_path"

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./Configs/config.yaml")
    return parser.parse_args()

def list_files(directory, ext=None):
    """
    Return the list of files in folder
    ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
    """
    list_files =  [os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

    return sorted(list_files)

def init_input_dictionary(config):
    """
    Initialize the dictionary with the inputs
    """
    inputs = {}

    inputs["Image"] = []
    inputs[KEY_BACKGROUND_LAYER] = []
    inputs[KEY_SELECTED_REGIONS] = []

    idx_layer = 1
    for path_layer in config.path_layer:
            name_layer = "rgba PNG - Layer " + str(idx_layer)
            idx_layer+=1

            inputs[name_layer] = []

    list_src_files = list_files(config.path_src)

    for path_img in list_src_files:

        print (path_img)
        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_img
        inputs["Image"].append(dict_img)

        filename_img = os.path.basename(path_img)

        #list_bg_files = list_files(config.path_bg) #background

        path_bg = os.path.join(config.path_bg, filename_img)

        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_bg
        inputs[KEY_BACKGROUND_LAYER].append(dict_img)


        filename_without_ext, ext = os.path.splitext(filename_img)
        filename_png = filename_without_ext + ".png"

        path_regions = os.path.join(config.path_regions, filename_png)
        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_regions
        inputs[KEY_SELECTED_REGIONS].append(dict_img)
        

        idx_layer = 1
        for path_layer in config.path_layer:
            fullpath_layer = os.path.join(path_layer, filename_img)
            name_layer = "rgba PNG - Layer " + str(idx_layer)
            idx_layer+=1

            dict_img = {}
            dict_img[KEY_RESOURCE_PATH] = fullpath_layer
            inputs[name_layer].append(dict_img)
            
    return inputs

def init_output_dictionary(config):
    """
    Initialize the dictionary with the outputs
    """
    outputs = {}

    idx_model = 0
    for path_model in config.path_out:
        name_model = "Model " + str(idx_model)
        idx_model+=1
        outputs[name_model] = []

    idx_model = 0
    for path_model in config.path_out:
        name_model = "Model " + str(idx_model)
        idx_model += 1
        
        dict_img = {}
        dict_img[KEY_RESOURCE_PATH] = path_model
        outputs[name_model].append(dict_img)

    return outputs

def main():
    args = getArgs()
    config = loadConfig(args.config, verbose=True)

    # Fail if arbitrary layers are not equal before training occurs.
    inputs = init_input_dictionary(config)
    outputs = init_output_dictionary(config)

    input_ports = len([x for x in inputs if "Layer" in x])
    output_ports = len([x for x in outputs if "Model" in x or "Log file" in x])
    if input_ports not in [output_ports, output_ports - 1]: # So it still works if Log File is added as an output. 
        raise Exception(
            'The number of input layers "rgba PNG - Layers" does not match the number of'
            ' output "Adjustable models"\n'
            "input_ports: " + str(input_ports) + " output_ports: " + str(output_ports)
        )

    # Sanity check
    print ("Start preprocess")
    data_container = preprocess.preprocess(inputs, config.batch_size, config.patch_height, config.patch_width, config.number_samples_per_class)
    print ("After pre_training_check")

    # Create output models
    output_models_path = {}
    for i in range(input_ports):
        output_models_path[str(i)] = outputs["Model " + str(i)][0][KEY_RESOURCE_PATH]

    # Call in training function
    status = training.train_msae(
        inputs=data_container,
        num_labels=input_ports,
        height=config.patch_height,
        width=config.patch_width,
        output_path=output_models_path,
        file_selection_mode=config.file_selection_mode,
        sample_extraction_mode=config.sample_extraction_mode,
        epochs=config.max_epochs,
        number_samples_per_class=config.number_samples_per_class,
        batch_size=config.batch_size,
        patience=config.patience
    )

    print("Finishing the Fast CM trainer job.")

if __name__ == "__main__":
    main()
