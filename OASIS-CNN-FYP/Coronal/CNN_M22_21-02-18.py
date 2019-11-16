#################################################################################################################
# 
#	FILE NAME   : CNN_M22_21-02-18.py
#	DESCRIPTION : Attempt 22 of basic CNN model. InceptionV3 architecture. INCOMPLETE
#               https://github.com/fchollet/deep-learning-models/blob/master/inception_v3.py
#	Author      : 1)Aarti Susan Kuruvilla
#			      2)Raghav Sikaria
#
#################################################################################################################

# INITIALIZING MODEL NAME
modelName = "CNN_M22_21-02-18"

#################################################################################################################
# IMPORTING LIBRARIES
#################################################################################################################

import numpy as np
from numpy import argmax

# SEED
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import itertools
import matplotlib.pyplot as plt

import h5py

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve

import keras 
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard

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
# no_imgs = 235
# X = X[0:no_imgs,45:135,45:135]
# y= y[0:no_imgs]

##################################################################################################################
# TRAIN TEST SPLIT
##################################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##################################################################################################################
# DATA NORMALISATION/STANDARDISATION
##################################################################################################################

# CONVERTING DATA EXPLICITLY INTO FLOAT32 TYPE AND NORMALISING THEM BY DIVIDING BY 120
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train[X_train > 120] = 120
# X_train /= 120
# X_test[X_test > 120] = 120
# X_test /= 120

# RESHAPING DATA TO FEED INTO CNN LAYERS
img_rows, img_cols = 176, 176
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# CONVERTING OUTPUT LABELS INTO CATEGORICAL DATA TO FEED INTO CNN
# num_classes = 2
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

##################################################################################################################
# CNN MODULE
##################################################################################################################

# KERAS CALLBACK VARIABLE TO STORE EVENT LOG FILE FOR FEEDING INTO TENSORBOARD
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Outputs/' + str(modelName), histogram_freq=0, write_graph=True, write_images=True)

# INITIALIZING MODEL
model = Sequential()

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))

model.add(Convolution2D(80, 1, 1, activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(192, 3, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3), strides=(2,2)))

model.add(Dense(1, activation='sigmoid'))

# COMPILING MODEL
model.compile(loss='mean_squared_logarithmic_error', optimizer='Adam', metrics=['accuracy'])

# FITTING THE MODEL ON TRAINING DATA USING TESTING DATA AS VALIDATION DATA
# STORING THE RETURNED HISTORY OBJECT FOR CALLING HISTORY FUNCTION LATER
# PASSING CALLBACK VARIABLE TO UPDATE EVENT LOG FILE                   
history = model.fit(X_train, y_train,epochs=40, batch_size=5, verbose=1, validation_data=(X_test, y_test), callbacks=[tbCallBack])

##################################################################################################################
# SAVING MODEL AS JSON ( SAVES ONLY MODEL ARCHITECTURE CONFIGURATION )
##################################################################################################################

model_json = model.to_json()
with open('Outputs/' + str(modelName) + '/' + str(modelName) + "_JSONArchitecture.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()

##################################################################################################################
# SAVING AND LOADING CNN MODEL
##################################################################################################################

# SAVING MODEL AS H5 FILE ( SAVES MODEL CONFIGURATION, ARCHITECTURE, STATE AND WEIGHTS )
model.save('Outputs/' + str(modelName) + '/' + str(modelName) + "_ModelDetails.h5")

# DELETING EXISTING MODEL
# del model

# # LOADING MODEL FROM THE SAVED H5 FILE
# model = load_model('Outputs/' + str(modelName) + '/' + str(modelName) + "_ModelDetails.h5")

# ##################################################################################################################
# # SAVING MODEL ARCHITECTURE AND TRAINING AND VALIDATION DATA FROM HISTORY FUNCTION AS IMAGES
# ##################################################################################################################

plot_model(model, to_file='Outputs/' + str(modelName) + '/' + str(modelName) + "_modelPlot.png", show_shapes=True, show_layer_names=True)

# PLOTS FOR ACCURACY AND LOSS FROM HISTORY FUNCTION
# FOR ACCURACY
plt.figure(figsize=(10, 10), dpi=150)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Outputs/' + str(modelName) + '/' + str(modelName) + '_Accuracy_Graph.png', dpi=150)

# FOR LOSS
plt.figure(figsize=(10, 10), dpi=150)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Outputs/' + str(modelName) + '/' + str(modelName) + '_Loss_Graph.png', dpi=150)

# ##################################################################################################################
# # PREDICTING VALUES FOR TRAINING AND TESTING DATASET ON TRAINED MODEL
# ##################################################################################################################

y_pred_tr= model.predict_classes(X_train)
y_pred_te = model.predict_classes(X_test)
# print("y_pred_tr: ", y_pred_tr)
# print("y_pred_te: ", y_pred_te)

##################################################################################################################
# WRITING METRICS TO TEXT FILE FROM EVALUATE FUNCTION
##################################################################################################################

file = open('Outputs/' + str(modelName) + '/' + str(modelName) + "_Metrics.txt", "w")
file.write('\nMETRICS FROM MODEL EVALUATION FUNCTION\n')

# STORING SCORE FROM MODEL EVALUATE FUNCTION FOR TRAINING DATA
score = model.evaluate(X_train, y_train,verbose=1)
file.write('Train loss     : ' + str(score[0]) + '\n')
file.write('Train accuracy : ' +  str(score[1]) + '\n')

# STORING SCORE FROM MODEL EVALUATE FUNCTION FOR TESTING DATA
score = model.evaluate(X_test, y_test,verbose=1)
file.write('Test loss      : ' + str(score[0]) + '\n')
file.write('Test accuracy  : ' + str(score[1]) + '\n\n\n')
file.close()

##################################################################################################################
# FUNCTIONS FOR METRIC CALCULATION AND PLOTTING CONFUSION MATRIX
##################################################################################################################

# PLOTTING CONFUSION MATRIX
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# CALCULATING METRICS
def MetricCalc(parameter_a, parameter_b, dataset_type):

    tn, fp, fn, tp = confusion_matrix(parameter_a , parameter_b).ravel()
    cnf_matrix = confusion_matrix(parameter_a, parameter_b)
    plt.figure()
    
    # SAVING CONFUSION MATRIX PLOT AS AN IMAGE
    plot_confusion_matrix(cnf_matrix, classes=[0, 1])
    plt.savefig('Outputs/' + str(modelName) + '/' + str(modelName) + '_CNFMatrix' + str(dataset_type) + '.png', dpi=150)

    file.write('True Positive          : '+ str(tp) +'\n')
    file.write('True Negative          : '+ str(tn) +'\n')
    file.write('False Positive         : '+ str(fp) +'\n')
    file.write('False Negative         : '+ str(fn) +'\n')
                                        
    Actual_No = tn + fp                
    Actual_Yes = fn + tp
    Predicted_No = tn + fn
    Predicted_Yes = fp + tp
    Total = fp + fn + tn + tp

    Accuracy = (tp + tn)/Total
    Misclassification_Rate = (fp + fn)/Total
    True_Positive_Rate = tp/Actual_Yes
    False_Positive_Rate = fp/Actual_No
    Specificity = tn/Actual_No
    Precision = tp/Predicted_Yes
    Prevalence = Actual_Yes/Total

    file.write('Accuracy               : '+ str(Accuracy) +'\n')
    file.write('Misclassification Rate : '+ str(Misclassification_Rate) +'\n')
    file.write('True Positive Rate     : '+ str(True_Positive_Rate) +'\n')
    file.write('False Positive Rate    : '+ str(False_Positive_Rate) +'\n')
    file.write('Specificity            : '+ str(Specificity) +'\n')
    file.write('Precision              : '+ str(Precision) +'\n')
    file.write('Prevalence             : '+ str(Prevalence) +'\n')
    file.write('Precision Score        : '+ str(precision_score(parameter_a , parameter_b)) +'\n')
    file.write('Recall Score           : '+ str(recall_score(parameter_a , parameter_b)) +'\n')
    file.write('F1 Score               : '+ str(f1_score(parameter_a , parameter_b)) +'\n')
    file.write('Cohen Kappa Score      : '+ str(cohen_kappa_score(parameter_a , parameter_b)) +'\n\n\n')

    # ## ROC CURVE PLOTTING
    # fpr, tpr, thresholds = roc_curve(parameter_a, parameter_b)
    # print('fpr, tpr, thresholds: ', fpr, tpr, thresholds)
    # plt.figure(figsize=(8, 8), dpi=150)
    # plt.xlabel("FPR", fontsize=14)
    # plt.ylabel("TPR", fontsize=14)
    # plt.title("ROC Curve: " +str(dataset_type), fontsize=14)
    # plt.plot(fpr, tpr, color='yellow', linewidth=2, label='AD')
    # x = [0.0, 1.0]
    # plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')
    # plt.xlim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    # plt.legend(fontsize=10, loc='best')
    # plt.tight_layout()
    # plt.savefig("CNN_1_ROC_Curve_"+str(dataset_type), dpi=150)
    # plt.show()

##################################################################################################################
# CONVERTING OUTPUT LABELS BACK TO SIMPLE CLASS OUTPUTS FROM CATEGORICAL DATA
##################################################################################################################

# y_test = argmax(y_test, axis=1)
# y_train = argmax(y_train, axis=1)
# y_pred_tr = argmax(y_pred_tr, axis=1)
# y_pred_te = argmax(y_pred_te, axis=1)

##################################################################################################################
# WRITING ALL OTHER METRICS TO TEXT FILE
##################################################################################################################

file = open('Outputs/' + str(modelName) + '/' + str(modelName) + "_Metrics.txt", "a")
file.write('FORMULAS:\n')
file.write('Actual_No                 = tn + fp\n')
file.write('Actual_Yes                = fn + tp\n')
file.write('Predicted_No              = tn + fn\n')
file.write('Predicted_Yes             = fp + tp\n')
file.write('Total                     = fp + fn + tn + tp\n')
file.write('Accuracy                  = (tp + tn)/Total\n')
file.write('Misclassification_Rate    = (fp + fn)/Total\n')
file.write('True_Positive_Rate        = tp/Actual_Yes\n')
file.write('False_Positive_Rate       = fp/Actual_No\n')
file.write('Specificity               = tn/Actual_No\n')
file.write('Precision                 = tp/Predicted_Yes\n')
file.write('Prevalance                = Actual_Yes/Total\n\n\n')

file.write('TRAINING DATA METRICS:\n')
MetricCalc(y_train, y_pred_tr, "Train_data")
file.write('TESTING DATA METRICS:\n')
MetricCalc(y_test, y_pred_te, "Test_data")
file.close()

###################################################################################################################
