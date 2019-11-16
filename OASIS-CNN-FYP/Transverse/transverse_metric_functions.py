##################################################################################################################
# WRITING METRICS TO TEXT FILE FROM EVALUATE FUNCTION
##################################################################################################################

file = open('Transverse Outputs/' + str(modelName) + '/' + str(modelName) + "_Metrics.txt", "w")
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
    plt.savefig('Transverse Outputs/' + str(modelName) + '/' + str(modelName) + '_CNFMatrix' + str(dataset_type) + '.png', dpi=150)

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
# WRITING ALL OTHER METRICS TO TEXT FILE
##################################################################################################################

file = open('Transverse Outputs/' + str(modelName) + '/' + str(modelName) + "_Metrics.txt", "a")
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
