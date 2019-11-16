##################################################################################################################
# SAVING MODEL AS JSON ( SAVES ONLY MODEL ARCHITECTURE CONFIGURATION )
##################################################################################################################

model_json = model.to_json()
with open('Sagittal Outputs/' + str(modelName) + '/' + str(modelName) + "_JSONArchitecture.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()

##################################################################################################################
# SAVING AND LOADING CNN MODEL
##################################################################################################################

# SAVING MODEL AS H5 FILE ( SAVES MODEL CONFIGURATION, ARCHITECTURE, STATE AND WEIGHTS )
model.save('Sagittal Outputs/' + str(modelName) + '/' + str(modelName) + "_ModelDetails.h5")

# DELETING EXISTING MODEL
# del model

# # LOADING MODEL FROM THE SAVED H5 FILE
# model = load_model('Sagittal Outputs/' + str(modelName) + '/' + str(modelName) + "_ModelDetails.h5")

##################################################################################################################
# SAVING MODEL ARCHITECTURE AND TRAINING AND VALIDATION DATA FROM HISTORY FUNCTION AS IMAGES
##################################################################################################################

# ONLY FOR LOCAL RUNNING
if cloud_var == 0:
    plot_model(model, to_file='Sagittal Outputs/' + str(modelName) + '/' + str(modelName) + "_modelPlot.png", show_shapes=True, show_layer_names=True)

# PLOTS FOR ACCURACY AND LOSS FROM HISTORY FUNCTION
# FOR ACCURACY
plt.figure(figsize=(10, 10), dpi=150)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Sagittal Outputs/' + str(modelName) + '/' + str(modelName) + '_Accuracy_Graph.png', dpi=150)

# FOR LOSS
plt.figure(figsize=(10, 10), dpi=150)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Sagittal Outputs/' + str(modelName) + '/' + str(modelName) + '_Loss_Graph.png', dpi=150)
