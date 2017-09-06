import re
import pyedflib
import xml.etree.ElementTree

def isSHHSFile(fileName):
    # TODO: change the regex to math all SHHS files
    #return re.match("20[0-1][0-9]*\.EDF$", fileName)
    return re.match("[0-9]*\.EDF$", fileName)

def loadEdf(fileName, path):
    edf = pyedflib.EdfReader(path + "/" + fileName)

    return {"name": fileName,
            "duration": edf.getFileDuration(),
            "eeg2": edf.readSignal(2),
            "eeg2Header": edf.getSignalHeader(2),
            "ecg": edf.readSignal(3),
            "ecgHeader": edf.getSignalHeader(3),
            "emg": edf.readSignal(4),
            "emgHeader": edf.getSignalHeader(4),
            "eogl": edf.readSignal(5),
            "eoglHeader": edf.getSignalHeader(5),
            "eogr": edf.readSignal(6),
            "eogrHeader": edf.getSignalHeader(6),
            "eeg1": edf.readSignal(7),
            "eeg1Header": edf.getSignalHeader(7),
            "arousals": loadArousals(fileName, path)
            }

def loadArousals(fileName, path):
    name = fileName + ".XML"
    root = xml.etree.ElementTree.parse(path + "/" + name).getroot()
    arousals = []
    for event in root.iter('ScoredEvent'):
        if event[0].text == "Arousal ()":
            # print event[0].text
            # print event[1].text
            # print event[2].text
            arousals.append({"start": float(event[1].text), "duration": float(event[2].text)})
    return arousals
