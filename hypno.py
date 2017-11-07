import ConfigParser
import pyedflib
import numpy as np
import os
import sys
import pickle
import shhsfiles
import edfdata
import scipy.signal
import hypnomodel
from dataset import Dataset
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from utils import print_validation
from utils import tensor_resampling
from utils import tensor_padding
from utils import merge_input
from utils import combine_predictions

def define_callbacks(file_name):
    filepath = file_name + "{epoch:02d}-{val_acc:.2f}.hdf5"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    return callbacks

def reshape_3d(x, rate, time_window):
    x_shape = x.shape
    shape1 = rate * time_window
    shape0 = x_shape[0] // shape1
    shape2 = x_shape[1]
    X = np.reshape(x, (shape0, shape1, shape2))
    return X

def prepare_data(X, Y, signals, signals_rate, timeWindow):
    # Basic transformations to hypnogram
    Y[Y == 2] = 1 # merge N1 and N2
    Y[Y == 3] = 2 # merge stage 3 & 4 and move to number 2
    Y[Y == 4] = 2 
    Y[Y >= 5] = 3 # move rem to number 3

    X = reshape_3d(X, 125, timeWindow)
       
    # window normalization:
    for idx in [0, 1, 3, 4]:
        meanX = np.mean(X[:, idx], 1)
        stdX = np.std(X[:, idx], 1)
        meanX = np.reshape(meanX, (meanX.shape[0], 1))
        stdX = np.reshape(stdX, (stdX.shape[0], 1))
        X[:, idx] -= meanX
        X[:, idx] /= stdX

    # signal normalization:
    for idx in [2]:
        meanX = np.mean(X[:, idx])
        stdX = np.std(X[:, idx])
        X[:, idx] -= meanX
        X[:, idx] /= stdX

    return X, Y

def main():
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    edf_path = config.get('paths', 'edf')
    pickle_path = config.get('paths', 'pickle')

    SIGNALS = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']
    signals_rate = {'eeg1': 125, 'eeg2': 125, 'emg':125, 'eogr':50, 'eogl':50}
    time_window = 30 #time in seconds

    dataset = Dataset(edf_path, pickle_path, SIGNALS)
    dataset.load()
    
    x_test, y_test = dataset.test_set()
    x_test, y_test = prepare_data(x_test, y_test, SIGNALS, signals_rate, time_window)
    
    x_train, y_train = dataset.train_and_validation_set()
    x_train, y_train = prepare_data(x_train, y_train, SIGNALS, signals_rate, time_window)
    
    max_epochs = 50
    y_cat = to_categorical(y_train)
    
    mlp_layers = []
    mlp_layers.append([4])
    # mlp_layers.append([16, 4])
    # mlp_layers.append([32, 4])
    # mlp_layers.append([64, 4])
    # mlp_layers.append([128, 4])

    for layers in mlp_layers:
        preds_list = [] 
        for i in range(len(SIGNALS)):
            signal_train = x_train[:, :, i]
            signal_test = x_test[:, :, i]
            callbacks = define_callbacks(SIGNALS[i] + str(layers))
            model = hypnomodel.mlpModel(input1_shape=(signal_train.shape[1], ), layers=layers)
            model.fit(signal_train, y_cat, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
            preds = model.predict(signal_test)
            preds_list.append(preds)
            preds = np.argmax(preds, axis=1)
            print_validation(y_test.flatten(), preds)

        preds = combine_predictions(preds_list, option='mean')
        preds = np.argmax(preds, axis=1)
        print_validation(y_test.flatten(), preds)

    conv_layers = []
    conv_layers.append([(128,20), (128,20), 20, (256,20)])
    # for k in [64, 128, 256]:
    #     for f in [3, 25, 50, 75, 125, 250][-2:]: # run only with 2 last f
    #         conv_layers.append([(k, f)])

    # for k in [64, 128, 256]:
    #     for f in [5, 7, 11]:
    #         conv_layers.append([(k, f), (k, f)])

    # for k in [128]:
    #     for f in [10, 15, 20, 25, 30]:
    #         conv_layers.append([(k, f), (k, f), f, (k, f)])

    # for f in [3, 5, 7, 11, 15, 20, 25]:
    #     conv_layers.append([(64, f), 2, (128, f), 2, (256, f), 2, (512, f)])

    # for k in [64, 128]:
    #     for f in [3, 5, 7, 11, 15, 20, 25, 30]:
    #         conv_layers.append([(k, f), (k, f), f, (k, f), (k, f)])

    # for f in [30]:
    #     for pool in [30]:
    #         conv_layers.append([(256, f), pool, (512, f), pool])

    for layers in conv_layers:
        callbacks = define_callbacks('conv' + str(layers))
        model = hypnomodel.convModel(input1_shape=(x_train.shape[1], x_train.shape[2]), layers=layers)
        model.fit(x_train, y_cat, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
        preds = model.predict(x_test)
        preds = np.argmax(preds, axis=1)
        print_validation(y_test.flatten(), preds)
        
    lstm_layers = []
    # lstm_layers.append([16])
    # lstm_layers.append([32])
    # lstm_layers.append([64])
    # lstm_layers.append([32, 16])
    # lstm_layers.append([64, 32])
    
    for conv_layer in conv_layers:
        for lstm_layer in lstm_layers:
            callbacks = define_callbacks('convlstm' + str(conv_layer) + str(lstm_layer))
            model = hypnomodel.convLstmModel(input1_shape=(x_train.shape[1], x_train.shape[2]), 
                conv_layers=conv_layer, lstm_layers=lstm_layer)
            model.fit(x_train, y_cat, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
            preds = model.predict(x_test)
            preds = np.argmax(preds, axis=1)
            print_validation(y_test.flatten(), preds)


if __name__ == '__main__':
    main()
