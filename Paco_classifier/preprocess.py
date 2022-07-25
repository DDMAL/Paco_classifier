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

def writeImgLayer(target, path):
    *path_prefix, target_name = path.split("/")
    filename = osp.join(*path_prefix, "{}-cropped.png".format(target_name))
    cv2.imwrite(filename, target)

    return filename

def preprocess(inputs, patch_height, patch_width):
    """
    Run a bunch of preprocessing steps in this function. Currently we have:
    1. extract X/Y/W/H from region mask and crop images and layers first
    """
    layer_key_list = [k for k in inputs.keys() if "Image" not in k and "regions" not in k]

    for idx in range(len(inputs["rgba PNG - Selected regions"])): # Select Region
        region_path = inputs["rgba PNG - Selected regions"][idx]["resource_path"]
        mask = getMaskFromRegion(region_path) 

        # Extract (x, y, w, h) from region mask
        X, Y, W, H = cv2.boundingRect(mask.astype(np.uint8))

        img_path = inputs["Image"][idx]["resource_path"]

        # Crop image and write image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[Y:Y+H, X:X+W, :]  # RGB, uint8
        new_path = writeImgLayer(img, img_path)
        inputs["Image"][idx]["resource_path"] = new_path

        for layer_key in layer_key_list: # Bg, neumes, staff. Exclude Image and Region
            layer_path = inputs[layer_key][idx]["resource_path"]

            # Crop layer and write layer
            layer = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)[Y:Y+H, X:X+W, :]  # 4-channel
            new_path = writeImgLayer(layer, layer_path)
            inputs[layer_key][idx]["resource_path"] = new_path
