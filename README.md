# Calvo-classifier

Repository of the Rodan wrapper for Calvo classifier

# Dependencies

Python dependencies:

  * h5py (2.7.0)
  * html5lib (0.9999999)
  * Keras (2.0.5)
  * numpy (1.13.1)
  * scipy (0.19.1)
  * setuptools (36.0.1)
  * six (1.10.0)
  * Theano (0.9.0)
  * opencv-python (3.2.0.7)

# Keras configuration

Calvo's classifier needs *Keras* and *Theano* to be installed. It can be easily done through **pip**. 

*Keras* works over both Theano and Tensorflow, so after the installation check **~/.keras/keras.json** so that it looks like:

~~~
{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
~~~

