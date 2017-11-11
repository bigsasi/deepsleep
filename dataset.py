import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import edfdata
from utils import tensor_padding

def load_edf_save_pickle(edf, signals, pickle_file):
    edf.open(edf.file_name)
    raw_signals = edfdata.readSignals(edf, signals)
    for s in signals:
        raw_signals[s] = raw_signals[s].astype(np.float32)
    signals_rate = edfdata.readRates(edf, signals)
    hypnogram = edfdata.readHypnogram(edf)
    arousals = edfdata.readLabels(edf)
    edf._close()
    f = open(pickle_file, 'wb')
    pickle.dump([raw_signals, signals_rate, hypnogram, arousals], f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    return [raw_signals, signals_rate, hypnogram, arousals]

class Dataset:

    def __init__(self, edf_path, pckl_path, signals):
        self.edf_path = edf_path
        self.pckl_path = pckl_path
        self.signals = signals
        self.edf_files = []
        self.edf_duration = []
        self.total_seconds = 0
        self.total_epochs = 0
        self.train_files = 180
        self.test_files = 50
        self.batch_size = 50
        self.validation_files = 20
        self.current_batch = 0
        self.reference_rate = 125

    def load(self):
        self.edf_files = edfdata.loadEdfs(self.edf_path)
        total_files = self.train_files + self.test_files + self.validation_files
        
        if total_files > len(self.edf_files):
            print("WARNING: Dataset is configured to use more files than available")

        self.edf_duration = np.zeros(len(self.edf_files), dtype=np.int32)
        for (i, edf) in enumerate(self.edf_files):
            self.edf_duration[i] = edf.file_duration

        self.total_seconds = np.sum(self.edf_duration)
        self.total_epochs = self.total_seconds // 30
        
    def configure(self, train, test, validation, batch_size=50):
        self.train_files = train
        self.test_files = test
        self.validation_files = validation
        self.batch_size = batch_size

    def signal_index(self, signal):
        for (i, s) in enumerate(self.signals):
            if s == signal:
                return i
        return -1 

    def load_set(self, files_list):

        edf_duration = self.edf_duration[files_list]
        seconds = np.sum(edf_duration)
        epochs = seconds // 30

        X = np.empty((seconds * 125, len(self.signals)), dtype=np.float32)
        Y = np.zeros((int(epochs), 1))

        for (i, file_id) in enumerate(files_list):
            edf = self.edf_files[file_id]
            edf_file_name = edf.file_name[len(self.edf_path) + 1:-4]
            pickle_file = self.pckl_path + "/" + edf_file_name + ".pckl"
            print("Loading file {}: {}".format(file_id, edf_file_name))
            try:
                f = open(pickle_file, 'rb')
                [raw_signals, signals_rate, hypnogram, arousals] = pickle.load(f)
                f.close()
            except IOError:
                [raw_signals, signals_rate, hypnogram, arousals] = load_edf_save_pickle(edf, self.signals, pickle_file)

            first_second = np.sum(edf_duration[:i])
            last_second = first_second + edf_duration[i]

            for signal in self.signals:
                rate = signals_rate[signal]
                mean_signal = np.mean(raw_signals[signal])
                std_signal = np.std(raw_signals[signal])
                raw_signals[signal] -= mean_signal
                raw_signals[signal] /= std_signal
                if rate != self.reference_rate:
                    shape1 = rate * 30
                    shape0 = raw_signals[signal].shape[0] // shape1
                    raw_signals[signal] = raw_signals[signal].reshape((shape0, shape1))
                    raw_signals[signal] = tensor_padding(raw_signals[signal], rate, self.reference_rate).flatten()
                X[first_second * self.reference_rate:last_second * self.reference_rate, self.signal_index(signal)] = raw_signals[signal]

            first_epoch = first_second // 30
            last_epoch = last_second // 30
            Y[first_epoch:last_epoch] = hypnogram

        return X, Y

    def validation_list(self):
        start = self.test_files
        num_files = self.validation_files
        return np.arange(start, start + num_files)

    def test_list(self):
        start = 0
        num_files = self.test_files
        return np.arange(start, start + num_files)

    def train_list(self):
        start = self.validation_files + self.test_files
        num_files = self.train_files
        return np.arange(start, start + num_files)

    def train_and_validation_list(self):
        start = self.test_files
        num_files = self.train_files + self.validation_files
        return np.arange(start, start + num_files)


