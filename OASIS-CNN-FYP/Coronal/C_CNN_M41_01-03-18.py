#################################################################################################################
# 
#	FILE NAME   : C_CNN_M41_04-03-18.py
#	DESCRIPTION : Attempt 38 of basic CNN model. Run locally on Laptop.
#	Author      : 1)Aarti Susan Kuruvilla
#			      2)Raghav Sikaria
#
#################################################################################################################

# INITIALIZING MODEL NAME
modelName = "C_CNN_M41_04-03-18"

# 1 = True, 0 = False
cloud_var = 0
categorical_var = 0
image_datagen_var = 0
do_random = 1
set_var_by_rescaling_target_classes = 1
label_for_class_0 = 0.2
label_for_class_1 = 1

##################################################################################################################
# CALL CONFIG
##################################################################################################################

exec(open('coronal_config.py').read())
exec(open('coronal_randomizer.py').read())

##################################################################################################################
# DATA NORMALISATION/STANDARDISATION
##################################################################################################################

# CONVERTING DATA EXPLICITLY INTO FLOAT32 TYPE AND NORMALISING THEM BY DIVIDING BY 250
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# RESHAPING DATA TO FEED INTO CNN LAYERS
img_rows, img_cols = 176, 176
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


# CONVERTING OUTPUT LABELS INTO CATEGORICAL DATA TO FEED INTO CNN (NECESSARY FOR CATEGORICAL CROSS ENTROPY)
if categorical_var == 1:
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

##################################################################################################################
# CNN MODULE
##################################################################################################################

# KERAS CALLBACK VARIABLE TO STORE EVENT LOG FILE FOR FEEDING INTO TENSORBOARD
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Coronal Outputs/' + str(modelName), histogram_freq=0, write_graph=True, write_images=True)

# INITIALIZING MODEL
model = Sequential()

# CONVOLUTIONAL LAYERS
model.add(Convolution2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Convolution2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
# model.add(Convolution2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# FLATTENING THE OUTPUT FROM PREVIOUS CONVOLUTIONAL LAYER TO FEED INTO NEXT DENSE NEURAL LAYER
model.add(Flatten())

# DENSE NEURAL NETWORK LAYERS
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))

# LAST DENSE LAYER
if categorical_var == 1:
    model.add(Dense(num_classes, activation='sigmoid'))
else:
    model.add(Dense(1, activation='sigmoid'))

# COMPILING MODEL
if categorical_var == 1:
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
else:
    model.compile(loss='mean_absolute_error', optimizer='RMSProp', metrics=['accuracy'])

# FITTING THE MODEL ON TRAINING DATA USING TESTING DATA AS VALIDATION DATA
# STORING THE RETURNED HISTORY OBJECT FOR CALLING HISTORY FUNCTION LATER
# PASSING CALLBACK VARIABLE TO UPDATE EVENT LOG FILE  
if image_datagen_var == 0:
    history = model.fit(X_train, y_train, epochs=20, batch_size=5, verbose=2, validation_data=(X_test, y_test), callbacks=[tbCallBack])
else:
    datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True)
    datagen.fit(X_train)
    # Fits the model on batches with real-time data augmentation
    if cloud_var == 0:
        history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=40), steps_per_epoch=5*len(X_train), epochs=2, verbose=1, validation_data=(X_test, y_test), callbacks=[tbCallBack])
    else:
        history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=40), steps_per_epoch=60*len(X_train), epochs=5, verbose=1, validation_data=(X_test, y_test), callbacks=[tbCallBack])            
##################################################################################################################
# CALL MODEL SAVE
##################################################################################################################

exec(open('coronal_model_save.py').read())

##################################################################################################################
# PREDICTING VALUES FOR TRAINING AND TESTING DATASET ON TRAINED MODEL
##################################################################################################################

# FOR CATEGORICAL DATA
if categorical_var == 1:
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

# FOR NON CATEGORICAL DATA
else:
    # y_pred_tr = model.predict_classes(X_train)
    # y_pred_te = model.predict_classes(X_test)

    y_pred_tr = model.predict(X_train) 
    y_pred_te = model.predict(X_test)
    
    y_pred_tr_class = model.predict_classes(X_train)
    y_pred_te_class = model.predict_classes(X_test)
    # if set_var_by_rescaling_target_classes == 1:
    #     temp_y_pred_tr = []
    #     temp_y_pred_te = [] 
        
    #     for ite in range(len(y_pred_te)):
    #         if y_pred_te[ite] == 0:
    #             temp_y_pred_te.append(0.4)
    #         else:
    #             temp_y_pred_te.append(0.6)
        
    #     for ite in range(len(y_pred_tr)):
    #         if y_pred_tr[ite] == 0:
    #             temp_y_pred_tr.append(0.4)
    #         else:
    #             temp_y_pred_tr.append(0.6)

    #     y_pred_te = np.array(temp_y_pred_te) 
    #     y_pred_tr = np.array(temp_y_pred_tr)
    
    # if set_var_by_rescaling_target_classes == 1:
    #     temp_y_train = []
    #     temp_y_test  = [] 
        
    #     for ite in range(len(y_test)):
    #         if y_test[ite] == label_for_class_0:
    #             temp_y_test.append(0)
    #         else:
    #             temp_y_test.append(1)
        
    #     for ite in range(len(y_train)):
    #         if y_train[ite] == label_for_class_0:
    #             temp_y_train.append(0)
    #         else:
    #             temp_y_train.append(1)

    #     y_test = np.array(temp_y_test) 
    #     y_train = np.array(temp_y_train)

    print("y test : \n", y_test)
    print("y train : \n", y_train)
    print("y pred te : \n", y_pred_te.ravel())
    print("y pred tr : \n", y_pred_tr.ravel())
    print("y pred te classes : \n", y_pred_te_class.ravel())
    print("y pred tr classes : \n", y_pred_tr_class.ravel())
    
    exec(open('variance_inflation_metric_helper.py').read())

    # print("y pred te : \n", y_pred_te)
    # print("y pred tr : \n", y_pred_tr)

##################################################################################################################
# CONVERTING OUTPUT LABELS BACK TO SIMPLE CLASS OUTPUTS FROM CATEGORICAL DATA
##################################################################################################################

# FOR CATEGORICAL DATA
if categorical_var == 1:
    y_test    = argmax(y_test, axis=1)
    y_train   = argmax(y_train, axis=1)
    y_pred_tr = argmax(y_pred_tr, axis=1)
    y_pred_te = argmax(y_pred_te, axis=1)

##################################################################################################################
# CALL METRIC FUNCTIONS
##################################################################################################################

exec(open('coronal_metric_functions.py').read())