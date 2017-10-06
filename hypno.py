import pyedflib
import numpy as np
import os
import pickle
import shhsfiles
import edfdata
import scipy.signal
import hypnomodel
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

def load_edf_save_pickle(edf, signals, pickle_file):
    edf.open(edf.file_name)
    raw_signals = edfdata.readSignals(edf, signals)
    signals_rate = edfdata.readRates(edf, signals)
    hypnogram = edfdata.readHypnogram(edf)
    arousals = edfdata.readLabels(edf)
    edf._close()
    f = open(pickle_file, 'wb')
    pickle.dump([raw_signals, signals_rate, hypnogram, arousals], f, protocol=4)
    return [raw_signals, signals_rate, hypnogram, arousals]

if __name__ == "__main__":
    EDF_PATH = "./SHHS/"
    PICKLE_PATH = "/media/Data/home/sasi/pickle_edf"
    edfFiles = edfdata.loadEdfs(EDF_PATH)
    
    print("Loading {} EDF files".format(len(edfFiles)))

    timeWindow=30 #time in seconds
    signals = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']

    edf_duration = np.zeros(len(edfFiles), dtype=np.int32)

    for (i, edf) in enumerate(edfFiles):
        edf_duration[i] = edf.file_duration

    total_seconds = np.sum(edf_duration)
    total_epochs = total_seconds // 30

    X = {}
    Y = np.zeros((total_epochs, 1))

    for (i, edf) in enumerate(edfFiles):
        edf_file_name = edf.file_name[len(EDF_PATH) + 1:-4]
        pickle_file = PICKLE_PATH + "/" + edf_file_name + ".pckl"
        print("Loading file {}: {}".format(i + 1, edf_file_name))
        try:
            f = open(pickle_file, 'rb')
            [raw_signals, signals_rate, hypnogram, arousals] = pickle.load(f)
            f.close()
        except IOError:
            [raw_signals, signals_rate, hypnogram, arousals] = load_edf_save_pickle(edf, signals, pickle_file)
            
        if not X:
            # Define X and Y
            for signal in signals:
                X[signal] = np.zeros((total_seconds * signals_rate[signal]), dtype=np.float32)


        first_second = np.sum(edf_duration[:i])
        last_second = first_second + edf_duration[i]
        for signal in signals:
            rate = signals_rate[signal]
            X[signal][first_second * rate:last_second * rate] = raw_signals[signal]

        first_epoch = first_second // 30
        last_epoch = last_second // 30
        Y[first_epoch:last_epoch] = hypnogram

    # Basic transformations to hypnogram
    Y[Y == 2] = 1 # merge N1 and Ne
    Y[Y == 3] = 2 # merge stage 3 & 4 and move to number 2
    Y[Y == 4] = 2 
    Y[Y >= 5] = 3 # move rem to number 3
    cat_Y = to_categorical(Y)

    # reshape to 2d
    for signal in signals:
        shape1 = signals_rate[signal] * timeWindow
        shape0 = len(X[signal]) // shape1 
        
        X[signal] = np.reshape(X[signal], (shape0, shape1))
        
    # window normalization:
    for s in ['eeg1', 'eeg2', 'eogr', 'eogl']:
        meanX = np.mean(X[s], 1)
        stdX = np.std(X[s], 1)
        meanX = np.reshape(meanX, (meanX.shape[0], 1))
        stdX = np.reshape(stdX, (stdX.shape[0], 1))
        X[s] -= meanX
        X[s] /= stdX

    # signal normalization:
    for s in ['emg']:
        meanX = np.mean(X[s])
        stdX = np.std(X[s])
        X[s] -= meanX
        X[s] /= stdX
        
    

    validation_files = 10
    max_epochs = 50
    valSize = np.sum(edf_duration[:validation_files]) // 30

    y_train = cat_Y[valSize:]
    y_test = Y[:valSize]

    mlp_layers = []
    # mlp_layers.append([4])
    # mlp_layers.append([16, 4])
    # mlp_layers.append([32, 4])
    # mlp_layers.append([64, 4])
    # mlp_layers.append([128, 4])

    for layers in mlp_layers:
        preds_list = [] 
        for sig in ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']:
            x_train = X[sig][valSize:]
            x_test = X[sig][:valSize]
            
            callbacks = define_callbacks(sig + str(layers))
            model = hypnomodel.mlpModel(input1_shape=(x_train.shape[1], ), layers=layers)
            model.fit(x_train, y_train, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
            preds = model.predict(x_test)
            preds_list.append(preds)
            preds = np.argmax(preds, axis=1)
            print_validation(y_test.flatten(), preds)

        preds = combine_predictions(preds_list, option='mean')
        preds = np.argmax(preds, axis=1)
        print_validation(y_test.flatten(), preds)

    # Convert everything to a 3d tensor
    for signal in ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']:
        old_shape = X[signal].shape
        X[signal] = np.reshape(X[signal], (old_shape[0], old_shape[1], 1))

    for signal in ['eogl', 'eogr']:
        X[signal] = tensor_padding(X[signal], 50, 125)
    allX = merge_input(X, ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl'])

    conv_layers = []
    # for k in [64, 128, 256]:
    #     for f in [3, 25, 50, 75, 125, 250][-2:]: # run only with 2 last f
    #         conv_layers.append([(k, f)])

    # for k in [64, 128, 256]:
    #     for f in [5, 7, 11]:
    #         conv_layers.append([(k, f), (k, f)])

    # for k in [64, 128, 256]:
    #     for f in [3, 5, 7, 11, 15, 20, 25, 30]:
    #         conv_layers.append([(k, f), (k, f), f, (k, f)])

    # for k in [64, 128]:
    #     for f in [3, 5, 7, 11, 15, 20, 25, 30]:
    #         conv_layers.append([(k, f), (k, f), f, (k, f), (k, f)])

    for f in [7, 15, 25]:
        for pool in [2, 7, 15, 25]:
            conv_layers.append([(128, f), (128, f), pool, (256, f), (256, f)])

    x_train = allX[valSize:]
    x_test = allX[:valSize]

    for layers in conv_layers:
        callbacks = define_callbacks('conv' + str(layers))
        model = hypnomodel.convModel(input1_shape=(x_train.shape[1], x_train.shape[2]), layers=layers)
        model.fit(x_train, y_train, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
        preds = model.predict(x_test)
        preds = np.argmax(preds, axis=1)
        print_validation(y_test.flatten(), preds)
        
    lstm_layers = []
    lstm_layers.append([16])
    lstm_layers.append([32])
    lstm_layers.append([64])
    lstm_layers.append([32, 16])
    lstm_layers.append([64, 32])
    
    for conv_layer in conv_layers:
        for lstm_layer in lstm_layers:
            callbacks = define_callbacks('convlstm' + str(conv_layer) + str(lstm_layer))
            model = hypnomodel.convLstmModel(input1_shape=(x_train.shape[1], x_train.shape[2]), 
                conv_layers=conv_layer, lstm_layers=lstm_layer)
            model.fit(x_train, y_train, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
            preds = model.predict(x_test)
            preds = np.argmax(preds, axis=1)
            print_validation(y_test.flatten(), preds)
