try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import edfdata
import re
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

def prepare_data(X, Y):
    # Basic transformations to hypnogram
    Y[Y == 2] = 1 # merge N1 and N2
    Y[Y == 3] = 2 # merge stage 3 & 4 and move to number 2
    Y[Y == 4] = 2 
    Y[Y >= 5] = 3 # move rem to number 3

    X = reshape_3d(X, Dataset.reference_rate, Dataset.window_length)
        
    return X, Y


class Dataset:

    signals = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']
    train_files = 180
    validation_files = 20
    test_files = 50
    reference_rate = 125
    window_length = 30
    max_samples = 6 * 60 * 2

    @staticmethod
    def sample_shape():
        return (Dataset.reference_rate * Dataset.window_length, len(Dataset.signals))

    def __init__(self, edf_path, pckl_path, batch_size=32):
        self.edf_path = edf_path
        self.pckl_path = pckl_path
        self.edf_files = []
        self.edf_duration = []
        self.total_seconds = 0
        self.total_epochs = 0
        self.batch_size = batch_size
        self.shuffle = True
        self.__load()
        self.means, self.stds = self.mean_and_deviation(self.train_list())

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

    def generator(self, files, num_files_loaded, name):
        while 1:
            indexes = self.__get_exploration_order(len(files))
        
            # Generate batches
            imax = len(indexes) // num_files_loaded
            for i in range(imax):
                # Load files_in_memory files
                files_selected = [files[k] for k in indexes[i*num_files_loaded:(i+1)*num_files_loaded]]
                X, y = self.load_set(files_selected)
                X, y = prepare_data(X, y)
                y_cat = to_categorical(y)

                # Limit number of samples to avoid problems with different duration between files
                batch_indexes = self.__get_exploration_order(len(y))[:Dataset.max_samples * num_files_loaded]     
                jmax = (Dataset.max_samples * num_files_loaded) // self.batch_size
                for j in range(jmax):
                    # Select batch_size samples
                    batches = batch_indexes[j * self.batch_size:(j + 1) * self.batch_size]
                    yield X[batches, :, :], y_cat[batches, :]


    def steps_per_epoch(self, files, num_files_loaded):
        return (len(files) // num_files_loaded) * ((Dataset.max_samples * num_files_loaded) // self.batch_size)

    def load_set(self, files_list):

        edf_duration = self.edf_duration[files_list]
        seconds = np.sum(edf_duration)
        epochs = seconds // 30

        X = np.empty((seconds * Dataset.reference_rate, len(self.signals)), dtype=np.float32)
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
                raw_signals[signal] -= self.means[signal]
                raw_signals[signal] /= self.stds[signal]
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

    def mean_and_deviation(self, files_list):
        sums = {}
        lens = {}
        means = {}
        diffs = {}
        stds = {}
        for signal in self.signals:
            sums[signal] = 0
            lens[signal] = 0
            means[signal] = 0
            diffs[signal] = 0
            stds[signal] = 0

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
            for signal in self.signals:
                sums[signal] += np.sum(raw_signals[signal])
                lens[signal] += np.sum(len(raw_signals[signal]))
        
        for signal in self.signals:
            means[signal] = sums[signal] / lens[signal]

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
            for signal in self.signals:
                diffs[signal] += np.sum(np.square(raw_signals[signal] - means[signal]))

        for signal in self.signals:
            stds[signal] = np.sqrt(diffs[signal] / (lens[signal] - 1))

        return means, stds

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

    def name_to_id(self, files_name):
        files_list = []
        for file_name in files_name:
            for (file_id, edf) in enumerate(self.edf_files):
                if re.match(".*" + file_name + ".*", edf.file_name):
                    files_list.append(file_id)

        return files_list

    def print_set(self, files_list, name):
        print("Dataset: {}".format(name))
        for file_id in files_list:
            edf_file_name = self.edf_files[file_id].file_name[len(self.edf_path) + 1:-4]
            print(edf_file_name)
