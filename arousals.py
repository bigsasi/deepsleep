import pyedflib
import numpy as np
import os
import shhsfiles
import edfdata
from kerasmodels import createDimmuMLPModel, convModel

# Load all the EDF files in a given path
def loadAllEdf(path):
    print ("Loading EDF files from", path)
    edfFiles = []
    for f in os.listdir(path):
        if shhsfiles.isSHHSFile(f):
            edfFiles.append(shhsfiles.loadEdf(f, path))
    return edfFiles

# # Slice data in train, test and validation
# def sliceData(X, Y):
#     total = X.shape[0];
#     firstSlice = round(total * .7);
#     secondSlice = firstSlice + round(total * .1);

#     X_train = X[:firstSlice]
#     X_test = X[firstSlice:secondSlice]
#     X_validate = X[secondSlice:]

#     Y_train = Y[:firstSlice]
#     Y_test = Y[firstSlice:secondSlice]
#     Y_validate = Y[secondSlice:]

#     return X_train, Y_train, X_test, Y_test, X_validate, Y_validate


if __name__ == "__main__":
    print("Automatic detection of arousals")
    path = "./SHHS"
    edfFiles = loadAllEdf(path)
    print("Loaded {} EDF files".format(len(edfFiles)))

    timeWindow=30#time in seconds
    (signals,rate,X) = edfdata.loadData(edfFiles, time = timeWindow)
    Y=edfdata.loadLabels(edfFiles,time = timeWindow)#change toe womdpws here if necessary
    # reshape to normalize everything, keep 2d for mlp/linear models
    # 3d for convnets and lstms and shit
    tmp=[]  
    for i,thisX in enumerate(X):
        tmpX=np.reshape(thisX,(thisX.shape[0],thisX.shape[1]) )
        meanX=np.mean(tmpX,1)
        meanX=np.reshape(meanX,(meanX.shape[0],1))
        tmpX=tmpX-meanX
        stdX=np.std(tmpX,1)
        stdX=np.reshape(stdX,(stdX.shape[0],1))
        tmpX=tmpX/stdX
        tmp.append(tmpX)
        X[i]=np.reshape(tmpX, (tmpX.shape[0],tmpX.shape[1],1))
    
    valSize = 5000
    predsMLP = []
    predsConv = []
    mlpModels = []
    convModels = []
    for i,thisRep in enumerate(tmp):
        thisX = thisRep[valSize:]
        thisY = Y[valSize:]
        model = createDimmuMLPModel(thisX,thisY)
        model.fit(thisX,thisY, batch_size=256)
        mlpModels.append(model)
        preds=model.predict(thisRep[:valSize])
        for j,pred in enumerate(preds):
            if pred>0.5:
                preds[j]=1
            else:
                preds[j]=0
        predsMLP.append(preds)
    predsMLP=np.reshape(np.array(predsMLP),(len(tmp),valSize))
    
    for i,thisRep in enumerate(X):
        thisX = thisRep[valSize:]
        thisY = Y[valSize:]
        model = convModel(thisX,thisY,(thisX.shape[1],1),kernel_size=rate[i])
        model.fit(thisX,thisY,batch_size=256)
        preds=model.predict(thisRep[:valSize])
        for j,pred in enumerate(preds):
            if pred>0.5:
                preds[j]=1
            else:
                preds[j]=0
        predsConv.append(preds)

