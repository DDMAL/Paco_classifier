import os
import cv2
import numpy as np
import argparse

from Paco_classifier import recognition_engine as recognition

from ConfigParser import loadConfig

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./Configs/config.yaml")
    return parser.parse_args()

def evaluateRodan(settings:dict, model_paths:list, inputs:dict):
    """
    Almost copy-paste from rodan job. Do not touch this part for now.
    """
    # Settings, Move to evaluate()
    height = settings['Height']
    width = settings['Width']
    threshold = settings['Threshold']

    # Inner configuration
    mode = 'logical'

    # Pack model path together, Move to evaluate()
    # background_model = inputs['Background model'][0]['resource_path']
    # model_paths = [background_model]
    # for i in range(input_ports):
    #     model_paths += [inputs['Model %d' % i][0]['resource_path']]

    # Image input is a list of images, you can classify a list of images and this iterates on each image.
    for idx, _ in enumerate(inputs['Image']):

        """
        model_paths = [path_to_ckpt, path_to_ckpt, ...]
        image = np.ndarray
        height, width = patch height, patch width
        mode = 'logical'
        """

        # Process
        image_filepath = inputs['Image'][idx]['resource_path']
        print ("Process {}".format(image_filepath))
        image = cv2.imread(image_filepath, 1) # (W, H, 3, np.uint8)
        print ("Start process_image_msae")
        analyses = recognition.process_image_msae(image, model_paths, height, width, mode = mode) # np.ndarray (image_W, image_H), uint8
        print ("Finish")

        for id_label, _ in enumerate(model_paths): # For each ckpt
            if mode == 'masks':
                mask = ((analyses[id_label] > (threshold / 100.0)) * 255).astype('uint8')
            elif mode == 'logical':
                label_range = np.array(id_label, dtype=np.uint8)
                mask = cv2.inRange(analyses, label_range, label_range) # (4166, 2940), dtype('uint8')

            original_masked = cv2.bitwise_and(image, image, mask = mask) # (4166, 2940, 3), dtype('uint8')
            original_masked[mask == 0] = (255, 255, 255)

            # Alpha = 0 when background
            alpha_channel = np.ones(mask.shape, dtype=mask.dtype) * 255 # (4166, 2940), dtype('uint8')
            alpha_channel[mask == 0] = 0
            b_channel, g_channel, r_channel = cv2.split(original_masked) # (4166, 2940), dtype('uint8'), ...
            original_masked_alpha = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # This is the final output, np.ndarray (image_W, image_H, 4), uint8
            print ("Cont")
            cv2.imwrite (f"img{idx}-{id_label}.png", original_masked_alpha)

def evaluate(cfg):
    # Settings
    settings = {}
    settings['Height'] = cfg.patch_height
    settings['Width'] = cfg.patch_width
    settings['Threshold'] = cfg.threshold

    # model_paths
    model_paths = cfg.path_ckpt

    # Inputs
    inputs = {"Image":[]}
    for img_path in cfg.testset:
        inputs["Image"].append({'resource_path':img_path})

    evaluateRodan(settings, model_paths, inputs)

if __name__ == "__main__":
    args = getArgs()
    config = loadConfig(args.config, verbose=True)
    evaluate(config)
    print ("Pass")
