# Calvo-classifier

Repository of the Rodan wrapper for Calvo classifier

# Rodan Jobs definition
This repository includes the following Rodan Jobs:
- `Pixelwise Analysis of Music Document` in **calvo_classifier.py**
- `Training model for Pixelwise Analysis of Music Document` in **calvo_trainer.py**
- `Fast Pixelwise Analysis of Music Document` in **fast_calvo_classifier.py**
  - Available in the **python3** Rodan queue.
- `Training model for Patchwise Analysis of Music Document` in **fast_calvo_trainer.py**
  - Available in the **python3** Rodan queue.

# Installation Dependencies

## Python dependencies:

  * h5py (2.7.0)
  * html5lib (0.9999999)
  * Keras (2.0.5)
  * numpy (1.15.1)
  * scipy (0.19.1)
  * setuptools (36.0.1)
  * six (1.10.0)
  * Tensorflow (1.5)
  * opencv-python (3.2.0.7)

## Keras configuration

Calvo's classifier needs *Keras* and *TensorFlow* to be installed. It can be easily done through **pip**. 

*Keras* works over both Theano and Tensorflow, so after the installation check **~/.keras/keras.json** so that it looks like:

~~~
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
~~~


# Mode of use for training the model

There are two options:
  * Through the module **fast_calvo_easy_training.py**. The parameters should be provided by console.
  * Through the script **script_run_training.sh**. The parameters should be provided within this script, by modifying the corresponding values.

Both are ready for receiving different parameters.
  * **-psr** `Path to the folder with the original images.`
  * **-prg** `Path to the folder with the mask regions.`
  * **-pbg** `Path to the folder with the background ground-truth data.`
  * **-pgt** `A repeatable option for a path to the folder with ground-truth data for each layer (other than background).`
  * **-out** `A repeteable option for a path for the output model for each layer (including the background).`
  * **-width** `Width of the window to extract samples.`
  * **-height** `Height of the window to extract samples`
  * **-b** `Batch size`
  * **-e** `Maximum number of epochs to be considered. The model will stop before if the the training process does not improve the results.`
  * **-n** `Number of samples to be extracted for each layer.`
  * **-fm** `Mode of the selection of the files in the training process. Possible values: [RANDOM, SHUFFLE, SEQUENTIAL] (Default: SHUFFLE)`
  * **-sm** `Mode of extraction of samples. Possible values: [RANDOM, SEQUENTIAL] (Default: RANDOM)`
  
Note that the parameters **-pgt** and **-out** indicate the path to the folders containing respective files and the number of the paths of both parameters must be the same. For example:

~~~
  python -u fast_calvo_easy_training.py  
            -psr datasets/images  
            -prg datasets/regions  
            -pbg datasets/layers/bg  
            -pgt datasets/layers/staff  
            -pgt datasets/layers/neumes  
            -out Models/model_background.hdf5  
            -out Models/model_staff.hdf5  
            -out Models/model_neumes.hdf5  
            -width 256  
            -height 256  
            -b 8  
            -e 50  
            -n 1000  
            -fm SHUFFLE  
            -sm RANDOM  
~~~

When using **script_run_training.sh**, in console only it is necessary to run the script:
~~~
     ./script_run_training.sh`
~~~

Within that script, there are a set of parameters with an example of use. Each one of these variables matches with a parameter of the python code.

  * **PATH_IMAGES**="datasets/images"  
  * **PATH_REGIONS**="datasets/regions"  
  * **PATH_BACKGROUND**="datasets/layers/bg"  
  * **PATH_LAYERS**=("datasets/layers/staff" "datasets/layers/neumes")  
  * **OUTPUT_MODEL**=("Models/model_background.hdf5" "Models/model_staff.hdf5" "Models/model_neumes.hdf5")  
  * **WINDOW_WIDTH**=256  
  * **WINDOW_HEIGHT**=256  
  * **BATCH_SIZE**=8  
  * **MAX_EPOCHS**=50  
  * **NUMBER_SAMPLES_PER_CLASS**=1000  
  * **FILE_SELECTION_MODE**="SHUFFLE"  
  * **SAMPLE_EXTRACTION_MODE**="RANDOM"  


**IMPORTANT**: Currently, these scripts require that the data have a specific folder structure. It is necessary to have a folder for images, another for the region masks, another for the background layer, and an additional folder for each layer different to the background. For example:

  - **datasets** `Parent folder.`
    - **images** `Folder for images.`
    - **regions** `Folder for the region masks.`
    - **background** `Folder for the background layer.`
    - **staff** `Folder for the staff layer.`
    - **neume** `Folder for the neume notation layer.
    - **layer_x** `Folder for the x layer, etc.`

With this structure, each image have exactly the same name of the associated data in each folder. For example, if there is an image with the name **image_01.png**, the structure should be:
  - **datasets**
    - **images** 
      - image_01.png
      - example_99.png
    - **regions**
      - image_01.png
      - example_99.png
    - **background**
      - image_01.png
      - example_99.png
    - **staff**
      - image_01.png
      - example_99.png
    - **neume**
      - image_01.png
      - example_99.png
