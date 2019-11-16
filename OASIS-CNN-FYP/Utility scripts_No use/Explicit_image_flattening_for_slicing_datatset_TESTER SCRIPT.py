import numpy as np
import h5py

##################################################################################################################
# READING DATA FROM HDF5 FILE AND STORING IN NUMPY ARRAY
##################################################################################################################

hf = h5py.File('OASIS_cor_grey_ren_data.hdf5', 'r')
X = hf.get('Images_2D')
y = hf.get('Labels')
X = np.array(X)
y = np.array(y)
hf.close()

##################################################################################################################
# SELECTING SMALL DATA FOR MACHINE TO RUN LOCALLY
##################################################################################################################

# MODIFY FOR DEBUGGING
no_imgs = 235
X = X[0:no_imgs,45:135,45:135]
# y= y[0:no_imgs]
X_1 = []
for ite in range(0, 235):
    temp = X[ite].flatten()
    X_1.append(temp)

X = np.array(X_1)


