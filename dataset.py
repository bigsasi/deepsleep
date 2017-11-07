#import cPickle as pickle
import pickle
import numpy as np
import edfdata
import sys
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
        self.train_files = 10
        self.test_files = 10
        self.batch_size = 50
        self.validation_files = 10
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

    def __load_set(self, files_slice):

        edf_duration = self.edf_duration[files_slice]
        seconds = np.sum(edf_duration)
        epochs = seconds // 30

        X = np.empty((seconds * 125, len(self.signals)), dtype=np.float32)
        Y = np.zeros((int(epochs), 1))

        for (i, edf) in enumerate(self.edf_files[files_slice]):
            edf_file_name = edf.file_name[len(self.edf_path) + 1:-4]
            pickle_file = self.pckl_path + "/" + edf_file_name + ".pckl"
            print("Loading file {}: {}".format(files_slice.start + i + 1, edf_file_name))
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

    def has_next_batch(self, train_and_validation=True):
        if train_and_validation:
            total = self.train_files + self.test_files
        else:
            total = self.train_files

        return self.current_batch + self.batch_size <= total

    def next_batch(self, train_and_validation=True):
        if train_and_validation:
            slice = self.__train_and_validation_slice()
        else:
            slice = self.__train_slice()
        start = self.current_batch
        num_files = self.batch_size
        self.current_batch += num_files
        batch_slice = slice[start:start + num_files]
        return self.__load_set(batch_slice)

    def test_set(self):
        return self.__load_set(self.__test_slice())

    def validation_set(self):
        return self.__load_set(self.__validation_slice())

    def train_set(self):
        return self.__load_set(self.__train_slice())

    def train_and_validation_set(self):
        return self.__load_set(self.__train_and_validation_slice())

    def __validation_slice(self):
        start = self.test_files
        num_files = self.validation_files
        return slice(start, start + num_files)

    def __test_slice(self):
        start = 0
        num_files = self.test_files
        return slice(start, start + num_files)

    def __train_slice(self):
        start = self.validation_files + self.test_files
        num_files = self.train_files
        return slice(start, start + num_files)


    def __train_and_validation_slice(self):
        start = self.test_files
        num_files = self.train_files + self.validation_files
        return slice(start, start + num_files)
