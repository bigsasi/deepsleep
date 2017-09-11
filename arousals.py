import pyedflib
import numpy as np
import os
import shhsfiles
import edfdata
from kerasmodels import createDimmuMLPModel, convModel

if __name__ == "__main__":
    print("Automatic detection of arousals")
    path = "./SHHS"
    edfFiles = edfdata.loadPath(path)
    print("Loaded {} EDF files".format(len(edfFiles)))

    timeWindow=30#time in seconds
    signals = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']
    (rate, X, Y) = edfdata.readFiles(edfFiles, signals, time = timeWindow)
    # reshape to normalize everything, keep 2d for mlp/linear models
    # 3d for convnets and lstms and shit
    
    # print("Len(Y):", len(Y))
    # print("Sum(Y):", sum(Y))
    # for s in signals:
    #     print("Shape[{}]: {}".format(s, X[s].shape))
    
    tmp=[]  
    for s in signals:
        tmpX=np.reshape(X[s],(X[s].shape[0],X[s].shape[1]) )
        meanX=np.mean(tmpX,1)
        meanX=np.reshape(meanX,(meanX.shape[0],1))
        tmpX=tmpX-meanX
        stdX=np.std(tmpX,1)
        stdX=np.reshape(stdX,(stdX.shape[0],1))
        tmpX=tmpX/stdX
        tmp.append(tmpX)
        X[s]=np.reshape(tmpX, (tmpX.shape[0],tmpX.shape[1],1))
    
    # for i,thisX in enumerate(X):
    #     tmpX=np.reshape(thisX,(thisX.shape[0],thisX.shape[1]) )
    #     meanX=np.mean(tmpX,1)
    #     meanX=np.reshape(meanX,(meanX.shape[0],1))
    #     tmpX=tmpX-meanX
    #     stdX=np.std(tmpX,1)
    #     stdX=np.reshape(stdX,(stdX.shape[0],1))
    #     tmpX=tmpX/stdX
    #     tmp.append(tmpX)
    #     X[i]=np.reshape(tmpX, (tmpX.shape[0],tmpX.shape[1],1))
    
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

    for s in signals:
        thisX = X[s][valSize:]
        thisY = Y[valSize:]
        model = convModel(thisX,thisY,(thisX.shape[1],1),kernel_size=rate[s])
        model.fit(thisX,thisY,batch_size=256)
        preds=model.predict(X[s][:valSize])
        for j,pred in enumerate(preds):
            if pred>0.5:
                preds[j]=1
            else:
                preds[j]=0
        predsConv.append(preds)

    # for i,thisRep in enumerate(X):
    #     thisX = thisRep[valSize:]
    #     thisY = Y[valSize:]
    #     model = convModel(thisX,thisY,(thisX.shape[1],1),kernel_size=rate[i])
    #     model.fit(thisX,thisY,batch_size=256)
    #     preds=model.predict(thisRep[:valSize])
    #     for j,pred in enumerate(preds):
    #         if pred>0.5:
    #             preds[j]=1
    #         else:
    #             preds[j]=0
    #     predsConv.append(preds)

