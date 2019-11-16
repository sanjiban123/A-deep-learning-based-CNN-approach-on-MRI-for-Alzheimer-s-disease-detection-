
##################################################################################################################
# READING DATA FROM HDF5 FILE AND STORING IN NUMPY ARRAY
##################################################################################################################

if do_random == 1:
    X_label0=[]
    y_label0=[]
    X_label1=[]
    y_label1=[]

    for ite in range(len(y)):
        if y[ite] == 0:
            X_label0.append(X[ite])
            y_label0.append(y[ite])
        else:
            X_label1.append(X[ite])
            y_label1.append(y[ite])


    X_label0=np.array(X_label0)
    y_label0=np.array(y_label0)
    X_label1=np.array(X_label1)
    y_label1=np.array(y_label1)

    # DEBUG UTILITY STATEMENTS
    # print("X label0: ", X_label0.shape)
    # print("y label0: ", y_label0.shape)
    # print("X label1: ", X_label1.shape)
    # print("y label1: ", y_label1.shape)
    # print("00000000000000000000000 : \n", y_label0)
    # print("\n\n111111111111111111111111 : \n", y_label1)

    # CHOOSING 100 IMAGES FROM THE LABEL 0 RANDOMLY WITHOUT REPLACEMENT
    X_label0 = X_label0[np.random.choice(X_label0.shape[0], 100, replace=False), :]
    y_label0 = y_label0[0:100]

    # DEBUG UTILITY STATEMENTS
    # print("X label0: ", X_label0.shape)
    # print("y label0: ", y_label0.shape)
    # print("X label1: ", X_label1.shape)
    # print("y label1: ", y_label1.shape)
    # plt.imshow(X_label0[5])
    # plt.show()

    X = np.concatenate((X_label0, X_label1))
    y = np.concatenate((y_label0, y_label1))


if set_var_by_rescaling_target_classes == 1:
    # for ite in range(len(y)):
    #     if y[ite] == 0:
    #         y[ite] = label_for_class_0
    #     else:
    #         y[ite] = label_for_class_1

    # y.flags.writeable = True    
    # y[y==0] = 0.4
    # y[y==1] = 0.6  

    # np.place(y, y==0, 0.4)
    # np.place(y, y==1, 0.6)  

    temp_y = []
    for ite in range(0,100):
        temp_y.append(label_for_class_0)
    for ite in range(0,100):
        temp_y.append(label_for_class_1)
    
    y = np.array(temp_y)
    print("Shape of Y data: ", y.shape)
    print("Y data: \n", y)

##################################################################################################################
# TRAIN TEST SPLIT
##################################################################################################################

if do_random == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# DEBUG UTILITY STATEMENTS
# print(" X train : ",X_train.shape)
# print(" Y train : ",y_train.shape)
# print(" X test  : ",X_test.shape)
# print(" Y test  : ",y_test.shape)

# a = (y_test == 0).sum()
# b = (y_test == 1).sum()
# print("Number of label 0 : ", a)
# print("Number of label 1 : ", b)