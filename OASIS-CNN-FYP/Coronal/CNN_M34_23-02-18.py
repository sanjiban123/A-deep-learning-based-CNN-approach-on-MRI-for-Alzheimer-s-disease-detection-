#################################################################################################################
# 
#	FILE NAME   : CNN_M34_23-02-18.py
#	DESCRIPTION : Attempt 33 of basic CNN model. Run locally on Laptop.
#	Author      : 1)Aarti Susan Kuruvilla
#			      2)Raghav Sikaria
#
#################################################################################################################

# INITIALIZING MODEL NAME
modelName = "CNN_M34_23-02-18"

#################################################################################################################
# IMPORTING LIBRARIES
#################################################################################################################

import numpy as np
from numpy import argmax

# # SEED
# import tensorflow as tf
# import random as rn
# import os
# os.environ['PYTHONHASHSEED'] = '0'
# np.random.seed(42)
# rn.seed(42)
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# from keras import backend as K
# tf.set_random_seed(42)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

import itertools
import matplotlib.pyplot as plt

import h5py

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve

import keras 
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
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
no_imgs = 235
X = X[0:no_imgs,45:135,45:135]
y= y[0:no_imgs]

##################################################################################################################
# TRAIN TEST SPLIT
##################################################################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##################################################################################################################
# DATA NORMALISATION/STANDARDISATION
##################################################################################################################

# # print("train x : ", X_train.shape)
# # print("test  x : ", X_test.shape)

# normalised_X = []
# image = []
# image = np.array(image)
# for ite in range(0,71):
#     image = X_test[ite]
#     print(image.shape)
#     max_val = np.amax(image)
#     min_val = np.amin(image)
#     mean_val = np.mean(image)
#     pix_count_val = (image > 140).sum()
#     print(image.shape, " MinV:", min_val, " MaxV:", max_val, " Mean:", mean_val, " pixCount:", pix_count_val)
#     image[image > 140] = 140
#     image = (0.0143 * image) - 1
#     max_val = np.amax(image)
#     min_val = np.amin(image)
#     mean_val = np.mean(image)
#     pix_count_val = (image > 140).sum()
#     print(image.shape, " MinV:", min_val, " MaxV:", max_val, " Mean:", mean_val, " pixCount:", pix_count_val)
#     normalised_X.append(image)
# X_test = np.array(normalised_X)    
# # X_test = normalised_X

# normalised_X = []
# image = []
# image = np.array(image)

# for ite in range(0,164):
#     image = X_train[ite]
#     image[image > 140] = 140
#     image = (0.0143 * image) - 1
#     normalised_X.append(image)
# X_train = np.array(normalised_X)
# # X_train = normalised_X

# print("train x : ", X_train.shape)
# print("test  x : ", X_test.shape)

# CONVERTING DATA EXPLICITLY INTO FLOAT32 TYPE AND NORMALISING THEM BY DIVIDING BY 250
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# RESHAPING DATA TO FEED INTO CNN LAYERS
img_rows, img_cols = 90, 90
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# CONVERTING OUTPUT LABELS INTO CATEGORICAL DATA TO FEED INTO CNN (NECESSARY FOR CATEGORICAL CROSS ENTROPY)
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

##################################################################################################################
# CNN MODULE
##################################################################################################################

# KERAS CALLBACK VARIABLE TO STORE EVENT LOG FILE FOR FEEDING INTO TENSORBOARD
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Outputs/' + str(modelName), histogram_freq=0, write_graph=True, write_images=True)

# INITIALIZING MODEL
model = Sequential()

# CONVOLUTIONAL LAYERS
model.add(Convolution2D(12, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Convolution2D(4, kernel_size=(5, 5), activation='relu'))
model.add(Convolution2D(2, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(8, kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(4, kernel_size=(3, 3), activation='relu'))
# model.add(Convolution2D(2, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))

# FLATTENING THE OUTPUT FROM PREVIOUS CONVOLUTIONAL LAYER TO FEED INTO NEXT DENSE NEURAL LAYER
model.add(Flatten())

# DENSE NEURAL NETWORK LAYERS
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

# COMPILING MODEL
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# FITTING THE MODEL ON TRAINING DATA USING TESTING DATA AS VALIDATION DATA
# STORING THE RETURNED HISTORY OBJECT FOR CALLING HISTORY FUNCTION LATER
# PASSING CALLBACK VARIABLE TO UPDATE EVENT LOG FILE                   
history = model.fit(X_train, y_train,epochs=30, batch_size=5, verbose=1, validation_data=(X_test, y_test), callbacks=[tbCallBack])

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

##################################################################################################################
# SAVING MODEL ARCHITECTURE AND TRAINING AND VALIDATION DATA FROM HISTORY FUNCTION AS IMAGES
##################################################################################################################

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

# y_pred_tr= model.predict_classes(X_train)
# y_pred_te = model.predict_classes(X_test)
y_pred_tr= model.predict(X_train)
y_pred_te = model.predict(X_test)
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
    file.write('Cohen Kappa Score      : '+ str(cohen_kappa_score(parameter_a , parameter_b)) +'\n')
    file.write('No. of actual 0s       : '+ str(Actual_No) +'\n')
    file.write('No. of predicted 0s    : '+ str(Predicted_No) +'\n')
    file.write('No. of actual 1s       : '+ str(Actual_Yes) +'\n')
    file.write('No. of predicted 1s    : '+ str(Predicted_Yes) +'\n\n\n')

def PredictedValues(parameter_a, parameter_b, dataset_type):
    counter = 1
    file.write(str(dataset_type) + '\n')
    file.write("S. No.".ljust(10) + "  |  " + "Actual".rjust(10) + "  |  " + "Predicted".rjust(10) + "\n")
    for ite in range(len(parameter_a)):
        file.write(str(counter).ljust(10) + "  |  " + str(parameter_a[ite]).rjust(10) + "  |  " + str(parameter_b[ite]).rjust(10) + "\n")  
        counter = counter + 1
    file.write("\n\n\n")


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

y_test = argmax(y_test, axis=1)
y_train = argmax(y_train, axis=1)
y_pred_tr = argmax(y_pred_tr, axis=1)
y_pred_te = argmax(y_pred_te, axis=1)

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

file.write('TRAINING DATA CLASS LABELS:\n')
PredictedValues(y_train, y_pred_tr, "Training Data")
file.write('TESTING DATA CLASS LABELS:\n')
PredictedValues(y_test, y_pred_te, "Testing Data")

file.close()

###################################################################################################################
