# -----------------------------------------------------------------------------
# Program Name:         calvo_trainer.py
# Program Description:  Rodan wrapper for Fast Calvo's classifier training
# -----------------------------------------------------------------------------

# Core
import logging
import os
import sys

# Third-party
from celery.utils.log import get_task_logger
import cv2
import numpy as np

# Project
from rodan.celery import app
from rodan.jobs.base import RodanTask
from . import training_engine_sae as training

"""Wrap Patchwise (Fast) Calvo classifier training in Rodan."""

logger = get_task_logger(__name__)


class FastCalvoTrainer(RodanTask):
    name = "Training model for Patchwise Analysis of Music Document"
    author = "Jorge Calvo-Zaragoza, Francisco J. Castellanos, Gabriel Vigliensoni, and Ichiro Fujinaga"
    description = "The job performs the training of many Selection Auto-Encoder model for the pixelwise analysis of music document images."
    enabled = True
    category = "OMR - Layout analysis"
    interactive = False

    settings = {
        'title': 'Training parameters',
        'type': 'object',
        'properties': {
            'Batch Size': {
                'type': 'integer',
                'minimum': 1,
                'default': 16,
                'maximum': 64,
            },
            'Maximum number of training epochs': {
                'type': 'integer',
                'minimum': 1,
                'default': 15
            },
            'Maximum number of samples per label': {
                'type': 'integer',
                'minimum': 100,
                'default': 2000
            },            
            'Patch height': {
                'type': 'integer',
                'minimum': 32,
                'default': 256
            },
            'Patch width': {
                'type': 'integer',
                'minimum': 32,
                'default': 256
            },
        },
        'job_queue': 'GPU'
    }

    input_port_types = (
        {'name': 'Image', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgb+png','image/rgb+jpg']},
        {'name': 'rgba PNG - Background layer', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Selected regions', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        # We did not go this route because it would be more difficult for the user to track layers
        # {'name': 'rgba PNG - Layers', 'minimum': 1, 'maximum': 10, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 0', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 1', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 2', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 3', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 4', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 5', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 6', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 7', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 8', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Layer 9', 'minimum': 0, 'maximum': 1, 'resource_types': ['image/rgba+png']},
    )

    output_port_types = (
        {'name': 'Background Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Log File', 'minimum': 1, 'maximum': 1, 'resource_types': ['text/plain']},
        # We did not go this route because it would be more difficult for the user to track layers
        # {'name': 'Adjustable models', 'minimum': 1, 'maximum': 10, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 0', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 1', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 2', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 3', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 4', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 5', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 6', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 7', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 8', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Model 9', 'minimum': 0, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
    )


    def run_my_task(self, inputs, settings, outputs):
        oldouts = sys.stdout, sys.stderr
        if len(outputs['Log File']) > 0:
            handler = logging.FileHandler(outputs['Log File'][0]['resource_path'])
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            )
            logger.addHandler(handler)
        try:
            # Settings
            batch_size = settings['Batch Size']
            patch_height = settings['Patch height']
            patch_width = settings['Patch width']
            max_number_of_epochs = settings['Maximum number of training epochs']
            max_samples_per_class = settings['Maximum number of samples per label']

            rlevel = app.conf.CELERY_REDIRECT_STDOUTS_LEVEL
            app.log.redirect_stdouts_to_logger(logger, rlevel)

            # Required input ports
            input_image = cv2.imread(inputs['Image'][0]['resource_path'], True) # 3-channel
            background = cv2.imread(inputs['rgba PNG - Background layer'][0]['resource_path'], cv2.IMREAD_UNCHANGED) # 4-channel
            regions = cv2.imread(inputs['rgba PNG - Selected regions'][0]['resource_path'], cv2.IMREAD_UNCHANGED) # 4-channel
            
            # Fail if arbitrary layers are not equal before training occurs.
            input_ports = len([x for x in inputs if x[:-1] == 'rgba PNG - Layer '])
            output_ports = len([x for x in outputs if x[:5] == 'Model'])
            if input_ports != output_ports:
                raise Exception(
                    'The number of input layers "rgba PNG - Layers" does not match the number of'
                    ' output "Adjustable models"\n'
                    'input_ports: %d output_ports: %d' % (input_ports, output_ports)
                )

            # Create categorical ground-truth
            gt = {}
            regions_mask = (regions[:, :, 3] == 255)
            gt['background'] = (background[:, :, 3] == 255) # background is already restricted to the selected regions (based on Pixel.js' behaviour)

            # Create output models
            output_models_path = {
                'background': outputs['Background Model'][0]['resource_path'],
            }

            # Populate remaining inputs and outputs
            for i in range(input_ports):
                file_obj = cv2.imread(inputs['rgba PNG - Layer %d' % i][0]['resource_path'], cv2.IMREAD_UNCHANGED)
                file_mask = (file_obj[:, :, 3] == 255)
                gt['%s' % i] = np.logical_and(file_mask, regions_mask)
                output_models_path['%s' % i] = outputs['Model %d' % i][0]['resource_path']

            # Call in training function
            status = training.train_msae(
                input_image=input_image,
                gt=gt,
                height=patch_height,
                width=patch_width,
                output_path=output_models_path,
                epochs=max_number_of_epochs,
                max_samples_per_class=max_samples_per_class,
                batch_size=batch_size,
            )

            print('Finishing the Fast CM trainer job.')
            return True
        finally:
            sys.stdout, sys.stderr = oldouts

    def my_error_information(self, exc, traceback):
        pass
