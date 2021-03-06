import xml.etree.ElementTree
import numpy as np

def getSignalIndex(signal):
    return {'eeg1': 7, 
            'eeg2': 2,
            'emg': 4,
            'eogl': 5, 
            'eogr': 6,
            'ecg': 3}.get(signal, 0)

def readSignals(edf, signals):
    x = {}
    for signal in signals:
        x[signal] = edf.readSignal(getSignalIndex(signal))
    return x

def readRates(edf, signals):
    rates = {}
    for signal in signals:
        rates[signal] = int(edf.getSignalHeader(getSignalIndex(signal))['sample_rate'])
    return rates

def readLabels(edf):
    name = edf.file_name + ".XML"
    root = xml.etree.ElementTree.parse(name).getroot()
    arousals = []
    for event in root.iter('ScoredEvent'):
        if event[0].text == "Arousal ()":
            arousals.append({"start": float(event[1].text), "duration": float(event[2].text)})
    return arousals

def readHypnogram(edf):
    name = edf.file_name + ".XML"
    root = xml.etree.ElementTree.parse(name).getroot()
    hypnogram = np.zeros((edf.file_duration // 30, 1), dtype=np.int)
    for i, event in enumerate(root.iter('SleepStage')):
        stage = int(event.text)
        hypnogram[i] = stage
        # stage = int(event.text)
        # print("Stage:",stage)
    return hypnogram
