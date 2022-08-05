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
    mask = open_image(region_path)[..., -1]
    mask = (mask == 255)
    return mask

def preprocess(inputs, batch_size, patch_height, patch_width, number_samples_per_class):
    """
    Run a bunch of preprocessing steps in this function. Currently we have:
    1. extract X/Y/W/H from region mask and crop images and layers first
    """
    # Check if batch size is less than number of samples per class
    logging.info("Checking batch size")
    checkBatch(batch_size, number_samples_per_class)
    # Check if all images are larger than or equal to patch height and patch width
    num_pages_training = len(inputs["Image"])

    layer_key_list = [k for k in inputs.keys() if "Image" not in k and "regions" not in k]
    check_empty_dict = {k: 0 for k in inputs.keys() if "Image" not in k and "regions" not in k}
    layer_dict = {k: [[],[]] for k in inputs.keys() if "regions" not in k}

    for idx in range(len(inputs["rgba PNG - Selected regions"])): # Select Region
        logging.info("Image {}".format(idx + 1))
        region_path = inputs["rgba PNG - Selected regions"][idx]["resource_path"]
        mask = getMaskFromRegion(region_path)

        # Extract (x, y, w, h) from region mask
        X, Y, W, H = cv2.boundingRect(mask.astype(np.uint8))

        img_path = inputs["Image"][idx]["resource_path"]

        # Crop image and write image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[Y:Y+H, X:X+W, :]  # RGB, uint8
        img = (255.-img) / 255.
        # Add image to layer dictionary
        layer_dict["Image"][0].append(img)
        
        for layer_key in layer_key_list: # Bg, neumes, staff. Exclude Image and Region
            logging.info("Checking {}".format(layer_key))
            layer_path = inputs[layer_key][idx]["resource_path"]
            # Crop layer and write layer
            layer = open_image(layer_path)[Y:Y+H, X:X+W, :]  # 4-channel
            # Check if image size is larger or equal to patch size
            check_size(layer, patch_height, patch_width)
            # Check if image is non-empty if it isn't original image
            empty, bg_mask = check_empty(layer)
            check_empty_dict[layer_key] += empty
            if not empty:
                layer_dict[layer_key][1].append(idx)
            # Add image to layer dictionary
            layer_dict[layer_key][0].append(bg_mask)
    
    for layer in check_empty_dict:
        # Check if an entire layer is does not only contain empty images
        if check_empty_dict[layer] >= num_pages_training:
            raise Exception('All images in layer {} are empty'.format(layer_key))
    
    return layer_dict

def check_size(img, patch_height, patch_width):
    if img.shape[0] < patch_height:
        raise ValueError('Patch height of {} is larger than image height of {}'.format(patch_height, img.shape[0]))
    if img.shape[1] < patch_width:
        raise ValueError('Patch height of {} is larger than image height of {}'.format(patch_width, img.shape[1]))

def check_empty(img):
    TRANSPARENCY = 3
    bg_mask = (img[:, :, TRANSPARENCY] == 255)
    
    return int(np.sum(bg_mask) == 0), bg_mask

def checkBatch(batch_size, number_samples_per_class):
    if batch_size > number_samples_per_class:
        raise ValueError("Not enough samples for on batch, got batchsize: {} and number_samples_per_class: {}".format(batch_size, number_samples_per_class))

def open_image(image_path):
    file_obj = cv2.imread(image_path, cv2.IMREAD_UNCHANGED,)  # 4-channel
    if file_obj is None : 
        raise Exception(
            'It is not possible to load the image\n'
            "Path: " + str(image_path)
        )
    return file_obj