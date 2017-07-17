import keras
import cv2
import numpy as np

from keras.models import load_model
from keras.backend import image_data_format

def process_image(image, model_path, vspan, hspan):   
    """
    Takes a document image path and a pre-trained model path
    and returns the process image (with logical labels).
    """    
      
    model = load_model(model_path)
  
    [height, width, channels] = image.shape

    output = np.zeros( (height, width), 'uint8')
    
    for row in range(vspan,vspan-height-1):
        for col in range(hspan,hspan-width-1):
            
            sample = image[row-vspan:row+vspan+1,col-hspan:col+hspan+1]
            
            if image_data_format() == 'channels_first':
                sample = np.asarray(sample).reshape(1, 3, vspan*2 + 1, hspan*2 + 1)
            else:
                sample = np.asarray(sample).reshape(1, vspan*2 + 1, hspan*2 + 1, 3)            
            
            prediction = model.predict(sample)[0]
            label = np.argmax(prediction)
            
            output[row][col] = label
        
    return output
