import pyedflib
import numpy as np
import os
import shhsfiles
import edfdata
from hypnomodel import createHypnoMLPModel, convHypnoModel, properHypnoConv, lstmHypnoModel, doubleHypnoConv
from keras.utils.np_utils import to_categorical

def checkResults(y_true, y_pred):
    for i in range(5):
        print("Sleep Stage {}:".format(i))
        true_0 = y_true == i
        false_0 = y_true != i
        tp = sum(y_pred[true_0] == i)
        fp = sum(y_pred[false_0] == i)
        tn = sum(y_pred[false_0] != i)
        fn = sum(y_pred[true_0] != i)
        print("tp, fp, tn, fn: {}, {}, {}, {}".format(tp, fp, tn, fn))
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        print("acc, sens, spec: {}, {}, {}".format(acc, sens, spec))

if __name__ == "__main__":
    print("Automatic detection of arousals")
    path = "./SHHS"
    edfFiles = edfdata.loadPath(path)
    print("Loaded {} EDF files".format(len(edfFiles)))

    timeWindow=30#time in seconds
    signals = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']
    (rate, X, Y_ar) = edfdata.readFiles(edfFiles, signals, time = timeWindow)
    Y = edfdata.readHypnograms(edfFiles)
    Y[Y >= 5] = 4
    categorical_Y = to_categorical(Y)
    # reshape to normalize everything, keep 2d for mlp/linear models
    # 3d for convnets and lstms and shit
    
    
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
    
    valSize = 5000
    predsMLP = []
    predsConv = []
    mlpModels = []
    convModels = []

    # Convulational model
    # for s in signals:
    #     thisX = X[s][valSize:]
    #     thisY = categorical_Y[valSize:]
    #     model = convHypnoModel(thisX,thisY,(thisX.shape[1],1),kernel_size=rate[s])
    #     model.fit(thisX,thisY,batch_size=256)
    #     preds=model.predict(X[s][:valSize])
    #     preds = np.argmax(preds, axis=1)
    #     checkResults(Y[:valSize].flatten(), preds)
    #     predsConv.append(preds)

    # The proper conv model
    # for s in signals:
    #     thisX = X[s][valSize:]
    #     thisY = categorical_Y[valSize:]
    #     model = properHypnoConv(thisX,thisY,(thisX.shape[1],1))
    #     model.fit(thisX,thisY,batch_size=256)
    #     preds=model.predict(X[s][:valSize])
    #     preds = np.argmax(preds, axis=1)
    #     checkResults(Y[:valSize].flatten(), preds)
    #     predsConv.append(preds)

    # Convulational model with double output
    for s in signals:
        thisX = X[s][valSize:]
        thisY = categorical_Y[valSize:]
        thisY_ar = Y_ar[valSize:]
        model = doubleHypnoConv(thisX,thisY,(thisX.shape[1],1))
        model.fit(thisX, [thisY, thisY_ar], batch_size=256)
        (sleep_preds, ar_preds) = model.predict(X[s][:valSize])
        sleep_preds = np.argmax(sleep_preds, axis=1)
        checkResults(Y[:valSize].flatten(), sleep_preds)
        predsConv.append(sleep_preds)

    # Convulational model with 2 signals at the same time (eeg1 + eeg2)
    # allX = X['eeg1']
    # allX = np.concatenate((allX, X['eeg2']),axis = 2)
    # print("allX.shape: ", allX.shape)
    # # for s in signals:
    # thisX = allX[valSize:]
    # thisY = categorical_Y[valSize:]
    # model = properHypnoConv(thisX,thisY,(thisX.shape[1],thisX.shape[2]))
    # model.fit(thisX,thisY,batch_size=256)
    # preds=model.predict(allX[:valSize])
    # preds = np.argmax(preds, axis=1)
    # checkResults(Y[:valSize].flatten(), preds)
    # predsConv.append(preds)

    # Convulational model with 2 signals at the same time (eogr + eogl)    
    # allX = X['eogr']
    # allX = np.concatenate((allX, X['eogl']),axis = 2)
    # thisX = allX[valSize:]
    # thisY = categorical_Y[valSize:]
    # model = properHypnoConv(thisX,thisY,(thisX.shape[1],thisX.shape[2]))
    # model.fit(thisX,thisY,batch_size=256)
    # preds=model.predict(allX[:valSize])
    # preds = np.argmax(preds, axis=1)
    # checkResults(Y[:valSize].flatten(), preds)
    # predsConv.append(preds)

    # Lstm model
    # for s in signals:
    #     thisX = X[s][valSize:]
    #     thisY = categorical_Y[valSize:]
    #     model = lstmHypnoModel(thisX,thisY,(thisX.shape[1],1))
    #     model.fit(thisX,thisY,batch_size=256)
    #     preds=model.predict(X[s][:valSize])
    #     preds = np.argmax(preds, axis=1)
    #     checkResults(Y[:valSize].flatten(), preds)
    #     # for j,pred in enumerate(preds):
    #     #     if pred>0.5:
    #     #         preds[j]=1
    #     #     else:
    #     #         preds[j]=0
    #     predsConv.append(preds)