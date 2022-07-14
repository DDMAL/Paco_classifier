import numpy as np
import cv2
import logging
import os.path as osp

logging.getLogger().setLevel(logging.INFO)

def getMaskFromRegion(region_path):
    """
    Load the the region layer from <region_path>, which is a ndarray with shape (W, H, 4).
    The last channel is the alpha channel. Get your mask using this channel!

    Return:
        A np.ndarray of shape (W, H) and type bool. Pixels in the selected region are True.
    """
    mask = cv2.imread(region_path, cv2.IMREAD_UNCHANGED)[..., -1]
    mask = (mask == 255)
    return mask

def writeCoordsToNpy(masked_layer, layer_path, patch_height, patch_width):
    """
    Extract the xy coordinate of labeled pixels in <masked_layer>. LABELED means that the value
    of a pixel is set to True. Keep extracted coordinates to a (#pixels, 2) ndarray and dump to .npy file. 
    Reading a npy file is supposed to run faster than np.where in each step.

    Params:
        masked_layer: ndarray of shape (W, H) and type bool. Labeled pixels are set True.
        layer_path: the path in dict[whatever keys you need]["resource_path"]. This is where you the layer is saved.
    
    Return:
        NONE. A ndarray of shape (#pixels, 2) is saved to <layer_path>+'.npy' with np.save(...).
    """
    # This is where the conversion starts
    x_coord, y_coord = np.where(masked_layer[:-patch_height, :-patch_width] == 1)
    coord = np.stack((x_coord, y_coord), axis=-1)

    *layer_path_prefix, layer_name = layer_path.split("/")
    filename = osp.join(*layer_path_prefix, "{}.npy".format(layer_name))
    np.save(filename, coord)

def preprocess(inputs, patch_height, patch_width):
    """
    Run a bunch of preprocessing steps in this function. Currently we have:
    1. Extract xy coordinates to stop using np.where in each step.
    """
    layer_key_list = [k for k in inputs.keys() if "Image" not in k and "regions" not in k]

    for idx in range(len(inputs["rgba PNG - Selected regions"])): # Select Region
        region_path = inputs["rgba PNG - Selected regions"][idx]["resource_path"]
        mask = getMaskFromRegion(region_path)

        for layer_key in layer_key_list: # Bg, neumes, staff. Exclude Image and Region
            # === Do not change this section, this section is meant to be identical to training_engine_sae.py ===
            layer_path = inputs[layer_key][idx]["resource_path"]
            layer = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED,)  # 4-channel

            TRANSPARENCY = 3
            layer = (layer[:, :, TRANSPARENCY] == 255)
            
            masked_layer = np.logical_and(layer, mask) # (W, H) with type bool. This is the Y
            # === === ===
            # 1. 
            writeCoordsToNpy(masked_layer, layer_path, patch_height, patch_width)