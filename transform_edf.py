"""From edf to pickle file containing
[raw_signals, signals_rate, hypnogram, arousals]"""
import pickle
import edfdata

if __name__ == "__main__":
    EDF_PATH = "/home/sasi/projects/apnea/SHHS"
    PICKLE_PATH = "/media/Data/home/sasi/pickle_edf"
    EDF_FILES = edfdata.loadEdfs(EDF_PATH)
    print("Loaded {} EDF files from {}".format(len(EDF_FILES), EDF_PATH))

    SIGNALS = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']

    i = 0
    for edf in EDF_FILES:
        i += 1
        edf_file_name = edf.file_name[len(EDF_PATH) + 1:-4]
        pickle_file = PICKLE_PATH + "/" + edf_file_name + ".pckl"
        try:
            f = open(pickle_file, 'rb')
            f.close()
        except IOError:
            print("Reading file {}: {}".format(i, edf_file_name))
            edf.open(edf.file_name)
            raw_signals = edfdata.readSignals(edf, SIGNALS)
            signals_rate = edfdata.readRates(edf, SIGNALS)
            hypnogram = edfdata.readHypnogram(edf)
            arousals = edfdata.readLabels(edf)
            edf._close()
            f = open(pickle_file, 'wb')
            pickle.dump([raw_signals, signals_rate, hypnogram, arousals], f, protocol=4)