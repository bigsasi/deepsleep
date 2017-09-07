import numpy as np
import shhsfiles
import os

# Loads data from a list of edfFiles using windows of time second
# Returns the list with the signals being loaded, a list with their
# sample_rate and a list with the raw data
def loadData(edfFiles, time = 30):
    signals = ['eeg1', 'eeg2', 'emg', 'ecg', 'eogr', 'eogl']
    rate = list()
    X = list()
    for i in range(0, len(signals)):
        rate.append(int(edfFiles[0][signals[i] + 'Header']['sample_rate']))
        X.append(extractAllEdfSignals(edfFiles, time, signals[i], rate[i]))
    return signals, rate, X

# Loads labels from a list of edfFiles using windows of time seconds
def loadLabels(edfFiles, time = 30):
    return extractAllEdfLabels(edfFiles, time)

# Loads all the EDF files in a given path
def loadPath(path):
    print ("Loading EDF files from", path)
    edfFiles = []
    for f in os.listdir(path):
        if shhsfiles.isSHHSFile(f):
            edfFiles.append(shhsfiles.loadEdf(f, path))
    return edfFiles

def formatData(time):
    pass

# Read signal from edf file into 3-d matrix where 1d: windows, 2d: time (single window), 3d: signal
def extractEdfSignal(edf, time, signal_name, sample_rate):
    n_signals = 1
    window_len = time * sample_rate

    if sample_rate != edf[signal_name + 'Header']['sample_rate']:
        print('Invalid rate in file {}!'.format(edf['name']))

    signal = edf[signal_name]
    num_windows = len(signal) // window_len


    X = signal.reshape((num_windows, window_len, 1));

    return X


def extractEdfLabels(edf, time = 30):
    """ loadEdfLabels(edf, time = 30),
    edf: dictionary of edf file, time: time for window 
    Returns the presense of event in each window as (0,1) """
    #just load any signal to get number of seconds in the file
    signal = edf['eeg1']
    sample_rate = int(edf['eeg1Header']['sample_rate'])
    window_len = time * sample_rate

    Y = np.zeros(len(signal) // window_len)
    for arousal in edf['arousals']:
        start = arousal['start']
        duration = arousal['duration']
        first = int(round(start / time))
        last = first + int(duration / time)
        Y[first:last + 1] = 1

    return Y

def extractAllEdfSignals(edfFiles, time, signal_name, rate):
    X = extractEdfSignal(edfFiles[0], time, signal_name, rate)
    for i in range(1, len(edfFiles)):
        # print("Loading file", i)
        Xi = extractEdfSignal(edfFiles[i], time, signal_name, rate)
        # print("With arousal:", sum(Yi_train))
        X = np.concatenate((X, Xi), axis = 0)

    return X

def extractAllEdfLabels(edfFiles, time):
    # Load first file alone, in order to shape Y_train
    Y = extractEdfLabels(edfFiles[0], time)
    for i in range(1, len(edfFiles)):
        Yi = extractEdfLabels(edfFiles[i], time)
        # print("With arousal:", sum(Yi_train))
        Y = np.concatenate((Y, Yi), axis = 0)

    return Y
