#-----------------------------------------------------------------------------
# Program Name:         calvo_classifier.py
# Program Description:  Rodan wrapper for Calvo's classifier
#-----------------------------------------------------------------------------

import cv2
import numpy as np
import os

from rodan.jobs.base import RodanTask
from . import recognition_engine as recognition


"""Wrap Fast Calvo classifier in Rodan."""
    
class FastCalvoClassifier(RodanTask):
    name = "Fast Pixelwise Analysis of Music Document"
    author = "Jorge Calvo-Zaragoza, Gabriel Vigliensoni, and Ichiro Fujinaga"
    description = "Given a pre-trained Convolutional neural network, the job performs a (fast) pixelwise analysis of music document images." 
    enabled = True
    category = "OMR - Layout analysis"
    interactive = False
    
    settings = {
        'title': 'Parameters',
        'type': 'object',
        'properties': {
            'Height': {
                'type': 'integer',
                'minimum': 1,
                'default': 256
            },
            'Width': {
                'type': 'integer',
                'minimum': 1,
                'default': 256
            },
            'Threshold': {
                'type': 'integer',
                'minimum': 0,
                'maximum': 100,
                'default': 50
            }
        },
        'job_queue': 'Python3'
    }
    
    input_port_types = (
        {'name': 'Image', 'minimum': 1, 'maximum': 100, 'resource_types': lambda mime: mime.startswith('image/')},
        {'name': 'Background model',    'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5'] },
        {'name': 'Symbol model',        'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5'] },
        {'name': 'Staff-line model',    'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5'] },
        {'name': 'Text model',          'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5'] },
    )
    output_port_types = (    
        {'name': 'Background', 'minimum': 0, 'maximum': 100, 'resource_types': ['image/rgba+png']},
        {'name': 'Music symbol', 'minimum': 0, 'maximum': 100, 'resource_types': ['image/rgba+png']},
        {'name': 'Staff lines', 'minimum': 0, 'maximum': 100, 'resource_types': ['image/rgba+png']},
        {'name': 'Text', 'minimum': 0, 'maximum': 100, 'resource_types': ['image/rgba+png']}        
    )
    
    
    

    """
    Entry point
    """
    def run_my_task(self, inputs, settings, outputs):
        # Inner configuration
        mode = 'logical'

        # Ports
        background_model = inputs['Background model'][0]['resource_path']
        symbol_model = inputs['Symbol model'][0]['resource_path']  
        staff_model = inputs['Staff-line model'][0]['resource_path']  
        text_model = inputs['Text model'][0]['resource_path']

        model_paths = [background_model, symbol_model, staff_model, text_model]
        
        # Settings        
        height =  settings['Height']
        width =  settings['Width']
        threshold = settings['Threshold']         
        
        for idx in range(len(inputs['Image'])):
            image_filepath = inputs['Image'][idx]['resource_path']
            # Process
            image = cv2.imread(image_filepath,True)            
            
            analyses = recognition.process_image_msae(image,model_paths,height,width, mode = mode)        
            
            for id_label in range(len(model_paths)):
                if mode == 'masks':
                    mask = ((analyses[id_label] > (threshold/100.0))*255).astype('uint8')
                elif mode == 'logical':
                    label_range = np.array(id_label, dtype=np.uint8)
                    mask = cv2.inRange(analyses, label_range, label_range)
      
                original_masked = cv2.bitwise_and(image,image,mask = mask)      
                original_masked[mask == 0] = (255, 255, 255)
            
                # Alpha = 0 when background
                alpha_channel = np.ones(mask.shape, dtype=mask.dtype)*255
                alpha_channel[mask == 0] = 0            
                b_channel, g_channel, r_channel = cv2.split(original_masked)
                original_masked_alpha = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
                if id_label == 0:
                    port = 'Background'            
                elif id_label == 1:
                    port = 'Music symbol'
                elif id_label == 2:
                    port = 'Staff lines'
                elif id_label == 3:
                    port = 'Text'                
                    
                if port in outputs:
                    cv2.imwrite(outputs[port][idx]['resource_path']+'.png',original_masked_alpha)   
                    os.rename(outputs[port][idx]['resource_path']+'.png',outputs[port][idx]['resource_path'])
       
        return True
