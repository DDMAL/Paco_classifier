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

import logging
import os
import sys
import cv2
import numpy as np
import training_engine_sae as training
import pdb

batch_size = 1
patch_height = 32
patch_width = 256
max_number_of_epochs = 1
max_samples_per_class = 100


# Fail if arbitrary layers are not equal before training occurs.
inputs = {
    "Image": [{"resource_path": "Images/Halifax_Folio_42v.png"}],
    "rgba PNG - Background layer": [
        {"resource_path": "Images/042v_BackgroundForNeumes.png"}
    ],
    "rgba PNG - Layer 0": [{"resource_path": "Images/042v_Neumes.png"}],
    "rgba PNG - Selected regions": [
        {"resource_path": "Images/042v_SelectedRegion.png"}
    ],
}
outputs = {
    "Background Model": [{"resource_path": "Images/back.hdf5"}],
    "Model 0": [{"resource_path": "Images/model0.hdf5"}],
    "Log File": [{"resource_path": "Images/logfile"}],
}

input_ports = len([x for x in inputs if "rgba PNG" in x])
output_ports = len([x for x in outputs if "Model" in x or "Log" in x])
int_model = input_ports - 2 # This only has the number of "Model x" inputs.
if input_ports != output_ports:
    raise Exception(
        'The number of input layers "rgba PNG - Layers" does not match the number of'
        ' output "Adjustable models"\n'
        "input_ports: " + str(input_ports) + " output_ports: " + str(output_ports)
    )

# Required input ports
# TODO assert that all layers have the same number of inputs (otherwise it will crack afterwards)
number_of_training_pages = len(inputs["Image"])

input_images = []
gts = []

# Create output models
output_models_path = {
    "background": outputs["Background Model"][0]["resource_path"],
}

for idx in range(number_of_training_pages):
    input_image = cv2.imread(inputs["Image"][idx]["resource_path"], True)  # 3-channel
    background = cv2.imread(
        inputs["rgba PNG - Background layer"][idx]["resource_path"],
        cv2.IMREAD_UNCHANGED,
    )  # 4-channel
    regions = cv2.imread(
        inputs["rgba PNG - Selected regions"][idx]["resource_path"],
        cv2.IMREAD_UNCHANGED,
    )  # 4-channel

    # Create categorical ground-truth
    gt = {}
    regions_mask = regions[:, :, 3] == 255
    gt["background"] = (
        background[:, :, 3] == 255
    )  # background is already restricted to the selected regions (based on Pixel.js' behaviour)

    # Populate remaining inputs and outputs
    for i in range(int_model):
        file_obj = cv2.imread(
            inputs["rgba PNG - Layer {layer_num}".format(layer_num=i)][idx]["resource_path"],
            cv2.IMREAD_UNCHANGED,
        )
        file_mask = file_obj[:, :, 3] == 255
        gt[str(i)] = np.logical_and(file_mask, regions_mask)

    input_images.append(input_image)
    gts.append(gt)

for i in range(int_model):
    output_models_path[str(i)] = outputs["Model " + str(i)][0]["resource_path"]
    # THIS IS NOT TAKING INTO ACCOUNT ANY FILE NOT NAMED MODEL IE BACKGROUND AND LOG!!!!

# Call in training function
status = training.train_msae(
    input_images=input_images,
    gts=gts,
    num_labels=int_model,
    height=patch_height,
    width=patch_width,
    output_path=output_models_path,
    epochs=max_number_of_epochs,
    max_samples_per_class=max_samples_per_class,
    batch_size=batch_size,
)

# THIS IS ONLY CREATING THE MODEL 0 FILE!!!!!!
print("Finishing the Fast CM trainer job.")