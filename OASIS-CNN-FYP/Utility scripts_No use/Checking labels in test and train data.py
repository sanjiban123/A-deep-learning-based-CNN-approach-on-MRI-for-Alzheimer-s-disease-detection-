import numpy as np
from numpy import argmax
import h5py
from sklearn.model_selection import train_test_split

hf = h5py.File('OASIS_cor_grey_ren_data.hdf5', 'r')
X = hf.get('Images_2D')
y = hf.get('Labels')
X = np.array(X)
y = np.array(y)
hf.close()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

c = []
d = []
for ite in range(len(y_train)):
    if y_train[ite] == 0 :
        c.append(0)
    else :
        d.append(1)

print(len(c))
print(len(d))

c = []
d = []
for ite in range(len(y_test)):
    if y_test[ite] == 0 :
        c.append(0)
    else :
        d.append(1)

print(len(c))
print(len(d))

# y_train = y_train.astype('float32')
# y_test = y_test.astype('float32')
# a = y_train[y_train < 1 ].sum()
# b = y_train[y_train == 1].sum()

# print("zeros : ", a)
# print("ones  : ", b)
# print(y_train)