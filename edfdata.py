"""Module for reading edf files"""
import os
import re
import numpy as np
import shhsfiles
import pyedflib

# Loads all the EDF files in a given path
def loadEdfs(path):
    """ Returns the list of edf files in path"""
    print ("Loading EDF files from", path)
    edf_files = []
    listfiles = os.listdir(path)
    listfiles.sort()
    for file_name in listfiles:
        if isEdfFile(file_name):
            edf = pyedflib.EdfReader(path + "/" + file_name)
            edf._close()
            edf_files.append(edf)
    return edf_files

def getNumEpochs(edfFile):
    return edfFile.file_duration // 30

def readHypnograms(edfFiles):
    edf = edfFiles[0]
    edf.open(edf.file_name)
    hypnograms = shhsfiles.readHypnogram(edf)
    edf._close()
    for edf in edfFiles[1:]:
        edf.open(edf.file_name)
        edf_hypno = shhsfiles.readHypnogram(edf)
        print("{} epochs".format(len(edf_hypno)))
        hypnograms = np.concatenate((hypnograms, edf_hypno), axis = 0)
        edf._close()
    return hypnograms

def formatLabels(labels, total_time, time):
    """Format labels into vector where each value represents a window of
    time seconds"""
    time_threshold = 1
    num_windows = total_time // time
    Y = np.zeros(num_windows)
    for label in labels:
        start = label['start']
        duration = label['duration']
        end = start + duration
        start_window = int(round(start / time))
        end_window = int(round(end / time))
        if end_window > start_window:
            window_limit = (start_window + 1) * 30
            if window_limit - start <= time_threshold:
                start_window += 1
            if end - window_limit <= time_threshold:
                end_window -= 1
        Y[start_window:end_window + 1] = 1
    print("{} arousals".format(len(labels)))
    return Y

def formatSignal(raw_signal, time, signal_rate):
    window_len = time * signal_rate
    num_windows = len(raw_signal) // window_len
    return raw_signal.reshape((num_windows, window_len, 1))

def readRawFiles(edfFiles, signals, time):
    """ Read a list of edfFiles returning a map with the signal rates, 
    a map with the signals in vector format and the vector with the 
    labels"""
    data = {}
    edf = edfFiles[0]
    edf.open(edf.file_name)
    rates, data_edf, labels_edf = readFile(edf, signals)
    for signal in signals:
        data[signal] = data_edf[signal]        
    labels = formatLabels(labels_edf, edf.file_duration, time)
    edf._close()
    
    for edf in edfFiles[1:]:
        edf.open(edf.file_name)
        rates_edf, data_edf, labels_edf = readFile(edf, signals)
        if rates != rates_edf:
            print("File {} does not match the reference signal rates".format(edf.file_name))
            edf._close()
            continue
        for signal in signals:
            data[signal] = np.concatenate((data[signal], data_edf[signal]), axis = 0)
        formated_labels = formatLabels(labels_edf, edf.file_duration, time)
        labels = np.concatenate((labels, formated_labels), axis = 0)

    return rates, data, labels

def readFiles(edfFiles, signals, time):
    """ Read a list of edfFiles returning a map with the signal rates, 
    a map with the signals in 3d format (using time seconds windows) 
    and the vector with the labels"""
    data = {}
    edf = edfFiles[0]
    edf.open(edf.file_name)
    rates, data_edf, labels_edf = readFile(edf, signals)
    for signal in signals:
        data[signal] = formatSignal(data_edf[signal], time, rates[signal])        
    labels = formatLabels(labels_edf, edf.file_duration, time)
    edf._close()

    for edf in edfFiles[1:]:
        edf.open(edf.file_name)
        rates_edf, data_edf, labels_edf = readFile(edf, signals)
        if rates != rates_edf:
            print("File {} does not match the reference signal rates".format(edf.file_name))
            edf._close()
            continue
        for signal in signals:
            formated_signal = formatSignal(data_edf[signal], time, rates_edf[signal])
            data[signal] = np.concatenate((data[signal], formated_signal), axis = 0)
        formated_labels = formatLabels(labels_edf, edf.file_duration, time)
        labels = np.concatenate((labels, formated_labels), axis = 0)
        edf._close()

    return rates, data, labels

def readFile(edf, signals):
    rates = readRates(edf, signals)
    data = readSignals(edf, signals)
    labels = readLabels(edf)

    return rates, data, labels

def readLabels(edf):
    if isSHHSedf(edf):
        return shhsfiles.readLabels(edf)

def readSignals(edf, signals):
    if isSHHSedf(edf):
        return shhsfiles.readSignals(edf, signals)

def readHypnogram(edf):
    if isSHHSedf(edf):
        return shhsfiles.readHypnogram(edf)

def readRates(edf, signals):
    if isSHHSedf(edf):
        return shhsfiles.readRates(edf, signals)

def isSHHSedf(edf):
    """ isSHHSFile(edf)
    Returns true if edf is from SHHS dataset"""
    # TODO: change the regex to math all SHHS files
    file_name = edf.file_name
    return re.match(".*SHHS.*", file_name)
    # return re.match("[0-9]*\.EDF$", file_name)

def isEdfFile(file_name):
    return re.match(".*[0-9]\.(EDF|edf)$", file_name)