
file = open('Coronal Outputs/' + str(modelName) + '/' + str(modelName) + "_Variance_helper_temp_Metrics.txt", "w")
def PredictedValues(parameter_a, parameter_b, dataset_type):
    counter = 1
    file.write(str(dataset_type) + '\n')
    file.write("S. No.".ljust(10) + "  |  " + "Actual".rjust(10) + "  |  " + "Predicted".rjust(10) + "\n")
    for ite in range(len(parameter_a)):
        file.write(str(counter).ljust(10) + "  |  " + str(parameter_a[ite]).rjust(10) + "  |  " + str(parameter_b[ite]).rjust(10) + "\n")  
        counter = counter + 1
    file.write("\n\n\n")

PredictedValues(y_test, y_pred_te, "Testing Data")

file.close()