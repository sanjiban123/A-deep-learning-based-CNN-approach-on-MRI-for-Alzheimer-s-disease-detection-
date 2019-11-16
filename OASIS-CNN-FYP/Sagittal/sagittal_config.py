#################################################################################################################
# IMPORTING LIBRARIES
#################################################################################################################

import numpy as np
from numpy import argmax

# SEED
np.random.seed(42)

import itertools
import matplotlib.pyplot as plt

import h5py

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve

import keras 
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

# ONLY FOR LOCAL RUNNING
if cloud_var == 0:
    from keras.utils.vis_utils import plot_model

##################################################################################################################
# READING DATA FROM HDF5 FILE AND STORING IN NUMPY ARRAY
##################################################################################################################

hf = h5py.File('OASIS_sag_grey_ren_data.hdf5', 'r')
X = hf.get('Images_2D')
y = hf.get('Labels')
X = np.array(X)
y = np.array(y)
hf.close()
X = X[0:235, 0:176, 16:192]
