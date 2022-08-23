import numpy as np
import cv2
import logging
import os.path as osp

from .data_loader import Data, DataContainer

logging.getLogger().setLevel(logging.INFO)

def bytes2Gb(b):
    return b / (1024**3)

def getMemoryLimit():
    """
    Read from /sys/fs/cgroup/memory/memory.limit_in_bytes to see the RAM size
    of the contaier.

    Return:
        RAM size in Gb

    from: https://carlosbecker.com/posts/python-docker-limits/
    """
    mem = 0
    if osp.isfile('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as limit:
            mem = int(limit.read()) # bytes
    return bytes2Gb(mem)

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
    1. Extract X/Y/W/H from the region mask and crop images and layers first
    2. The image width/height should be not less than the patch width/height.
    3. Keep a list to save the indices of nonempty layers. The model is trained on nonempty layers.
    4. Track the number of empty layers. It should be less than the number of images 
        (at least one trainable data).

    Return:
        {<rodan port name>:[[np.ndarray, ...], [int, int, ...]]}
            A dictionary with the rodan port names ('Image', 'rgba PNG - Layer 0 (Background)', ...)
            as keys and lists of two lists as their items.
            1st list: processed/cropped loaded images/layers represented as np.ndarray with shape (W, H, 3, dtype=float) or (W, H, dtyp=bool)
            2nd list: the indices of nonempty layers. The list is empty in dict['Image']
    """
    # Check if batch size is less than number of samples per class
    logging.info("Checking batch size")
    checkBatch(batch_size, number_samples_per_class)
    # Check if all images are larger than or equal to patch height and patch width
    num_pages_training = len(inputs["Image"])

    layer_key_list = [k for k in inputs.keys() if "Image" not in k and "regions" not in k]
    check_empty_dict = {k: 0 for k in inputs.keys() if "Image" not in k and "regions" not in k}

    # Get RAM limit
    ram_limit = max(getMemoryLimit() - 5, 0)
    data_container = DataContainer(ram_limit=ram_limit)

    for idx in range(len(inputs["rgba PNG - Selected regions"])): # Select Region
        logging.info("Image {}".format(idx + 1))
        region_path = inputs["rgba PNG - Selected regions"][idx]["resource_path"]

        mask = getMaskFromRegion(region_path)

        # Extract (x, y, w, h) from region mask
        X, Y, W, H = cv2.boundingRect(mask.astype(np.uint8))

        # Crop image and write image to npy
        img_path = inputs["Image"][idx]["resource_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[Y:Y+H, X:X+W, :]  # RGB, uint8

        # Creata data
        x_name = img_path.split("/")[-1]+"-{}".format(idx)
        img_W, img_H, img_C = img.shape
        if img_W < patch_width:
            img_W = patch_width*2
        if img_H < patch_height:
            img_H = patch_height*2
        img_path += ".npy"
        data_x = Data(x_name, img_path, bytes2Gb(img_W * img_H * img_C * img.itemsize))

        
        for layer_key in layer_key_list: # Bg, neumes, staff. Exclude Image and Region
            logging.info("Checking {}".format(layer_key))
            layer_path = inputs[layer_key][idx]["resource_path"]
            # Crop layer and write layer
            layer = open_image(layer_path)[Y:Y+H, X:X+W, :]  # 4-channel
            # Check if image size is larger or equal to patch size
            layer, pad = check_size(layer, patch_height, patch_width, layer_key)
            if "Background" in layer_key and pad:
                img_temp = np.copy(layer)[:,:,:3]
                img_temp[:img.shape[0],:img.shape[1]] = img
                img = img_temp
            # Check if image is non-empty if it isn't original image
            empty, bg_mask = check_empty(layer)
            layer_W, layer_H = bg_mask.shape
            layer_C = 1
            check_empty_dict[layer_key] += empty
            if not empty and bytes2Gb(layer_W*layer_H*layer_C*bg_mask.itemsize + img_W*img_H*img_C*img.itemsize) < ram_limit:
                # Save Y as npy file
                layer_path += ".npy"
                np.save(layer_path, bg_mask)

                # Add the XY pair to the data container
                data_y = Data(x_name, layer_path, bytes2Gb(layer_W * layer_H * layer_C * bg_mask.itemsize))
                data_container.addXYPair(x_name, layer_key, data_x, data_y)

        img = (255.-img) / 255.
        np.save(img_path, img)
    
    for layer in check_empty_dict:
        # Check if an entire layer is does not only contain empty images
        if check_empty_dict[layer] >= num_pages_training:
            raise Exception('All images in layer {} are empty'.format(layer_key))
    
    return data_container

def check_size(img, patch_height, patch_width, layer_name):
    height = img.shape[0]
    width = img.shape[1]
    if img.shape[0] >= patch_height and img.shape[1] >= patch_width:
        return img, False
    if img.shape[0] < patch_height:
        height = patch_height*2
    if img.shape[1] < patch_width:
        width = patch_width*2
    if "Background" in layer_name:
        unique = np.array(list(tuple(v) for m2d in img for v in m2d if v[3] == 255))
        list2d = np.random.randint(len(unique), size=(height, width))
        patch = unique[list2d, ...]
        patch[:img.shape[0],:img.shape[1]] = img  

        return patch, True

    else:
        patch = np.zeros((height, width, 4))
        patch[:img.shape[0],:img.shape[1]] = img

        return patch, True
    

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