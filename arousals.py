import pyedflib
import numpy as np
import os
import shhsfiles
import edfdata
#import kerasmodels

# Load all the EDF files in a given path
def loadAllEdf(path):
    print ("Loading EDF files from", path)
    edfFiles = []
    for f in os.listdir(path):
        if shhsfiles.isSHHSFile(f):
            edfFiles.append(shhsfiles.loadEdf(f, path))
    return edfFiles

# Slice data in train, test and validation
def sliceData(X, Y):
    total = X.shape[0];
    firstSlice = round(total * .7);
    secondSlice = firstSlice + round(total * .1);

    X_train = X[:firstSlice]
    X_test = X[firstSlice:secondSlice]
    X_validate = X[secondSlice:]

    Y_train = Y[:firstSlice]
    Y_test = Y[firstSlice:secondSlice]
    Y_validate = Y[secondSlice:]

    return X_train, Y_train, X_test, Y_test, X_validate, Y_validate

# def preprocessing(X):
#     for i in range(0, X.shape[2]):
#         val = np.mean(X[:,:,i])
#         X[:,:,i] = X[:,:,i] - val
#     return X

if __name__ == "__main__":
    print("Automatic detection of arousals")
    path = "/Users/dimitrisathanasakis/Work/spartan/apnea/SHHS"
    edfFiles = loadAllEdf(path)
    print("Loaded {} EDF files".format(len(edfFiles)))

    timeWindow=30#time in seconds
    (signals,rate,X) = edfdata.loadData(edfFiles, time = timeWindow)
    Y=edfdata.loadLabels(edfFiles,time = timeWindow)#change toe womdpws here if necessary

    # X=np.array(X)    
    # print("X: ", X.shape)
    # print("Y: ", Y.shape)
    # # X = preprocessing(X)

    # print("X: ", X.shape)
    # print("Y: ", Y.shape)

    # X_train, Y_train, X_test, Y_test, X_validate, Y_validate = sliceData(X, Y)
    # #m=np.mean(X_train)

    # print("X_train: ", X_train.shape)
    # print("Y_train: ", Y_train.shape)
    # print("X_test: ", X_test.shape)
    # print("Y_test: ", Y_test.shape)
    # print("X_validate: ", X_validate.shape)
    # print("Y_validate: ", Y_validate.shape)

    # model = kerasmodels.createMLPModel(X_train[:,:,0], Y_train)
    # kerasmodels.runModel(X_train, Y_train, X_test, Y_test, model)
