#import cPickle as pickle
import pickle
import numpy as np
import edfdata
import sys
from utils import tensor_padding
from keras.utils.np_utils import to_categorical

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

def reshape_3d(x, rate, time_window):
    x_shape = x.shape
    shape1 = rate * time_window
    shape0 = x_shape[0] // shape1
    shape2 = x_shape[1]
    X = np.reshape(x, (shape0, shape1, shape2))
    return X

def prepare_data(X, Y, timeWindow):
    # Basic transformations to hypnogram
    Y[Y == 2] = 1 # merge N1 and N2
    Y[Y == 3] = 2 # merge stage 3 & 4 and move to number 2
    Y[Y == 4] = 2 
    Y[Y >= 5] = 3 # move rem to number 3

    X = reshape_3d(X, 125, timeWindow)
       
    # window normalization:
    for idx in [0, 1, 3, 4]:
        signal = X[:, :, idx]
        mean_x = np.mean(signal, 1)
        std_x = np.std(signal, 1)
        mean_x = np.reshape(mean_x, (mean_x.shape[0], 1))
        std_x = np.reshape(std_x, (std_x.shape[0], 1))
        signal -= mean_x
        signal /= std_x
        X[:, :, idx] = signal
        
    # signal normalization:
    for idx in [2]:
        signal = X[:, :, idx]
        mean_x = np.mean(signal)
        std_x = np.std(signal)
        signal -= mean_x
        signal /= std_x
        X[:, :, idx] = signal
        
    return X, Y


class Dataset:

    signals = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']
    train_files = 10
    validation_files = 10
    test_files = 1
    reference_rate = 125
    window_length = 30

    @staticmethod
    def sample_shape():
        return (Dataset.reference_rate * Dataset.window_length, len(Dataset.signals))

    def __init__(self, edf_path, pckl_path, batch_size=360, files_in_memory=10):
        self.edf_path = edf_path
        self.pckl_path = pckl_path
        self.edf_files = []
        self.edf_duration = []
        self.total_seconds = 0
        self.total_epochs = 0
        self.batch_size = batch_size
        self.files_in_memory = files_in_memory
        # From all the files in memory just load 6 hours  (6 * 60 * 2 samples)
        self.max_samples = self.files_in_memory * 6 * 60 * 2
        self.shuffle = True
        self.__load()

    def __load(self):
        self.edf_files = edfdata.loadEdfs(self.edf_path)
        total_files = self.train_files + self.test_files + self.validation_files
        
        if total_files > len(self.edf_files):
            print("WARNING: Dataset is configured to use more files than available")

        self.edf_duration = np.zeros(len(self.edf_files), dtype=np.int32)
        for (i, edf) in enumerate(self.edf_files):
            self.edf_duration[i] = edf.file_duration

        self.total_seconds = np.sum(self.edf_duration)
        self.total_epochs = self.total_seconds // 30

    def signal_index(self, signal):
        for (i, s) in enumerate(self.signals):
            if s == signal:
                return i
        return -1 
    
    def __get_exploration_order(self, num_items):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(num_items)
        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def generator(self, files, name="default_generator"):
        while 1:
            indexes = self.__get_exploration_order(len(files))
        
            # Generate batches
            imax = len(indexes) // self.files_in_memory
            for i in range(imax):
                # Load files_in_memory files
                files_selected = [files[k] for k in indexes[i*self.files_in_memory:(i+1)*self.files_in_memory]]
                X, y = self.load_set(files_selected)
                X, y = prepare_data(X, y, 30)
                y_cat = to_categorical(y)

                # Limit number of samples to avoid problems with different duration between files
                batch_indexes = self.__get_exploration_order(len(y))[:self.max_samples]     
                jmax = self.max_samples // self.batch_size
                for j in range(jmax):
                    # Select batch_size samples
                    batches = batch_indexes[j * self.batch_size:(j + 1) * self.batch_size]
                    yield X[batches, :, :], y_cat[batches, :]


    def steps_per_epoch(self, files):
        return (len(files) // self.files_in_memory) * (self.max_samples // self.batch_size)

    def load_set(self, files_list):

        edf_duration = self.edf_duration[files_list]
        seconds = np.sum(edf_duration)
        epochs = seconds // 30

        X = np.empty((seconds * 125, len(self.signals)), dtype=np.float32)
        Y = np.zeros((int(epochs), 1))

        for (i, file_idx) in enumerate(files_list):
            edf = self.edf_files[file_idx]
            edf_file_name = edf.file_name[len(self.edf_path) + 1:-4]
            pickle_file = self.pckl_path + "/" + edf_file_name + ".pckl"
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
