try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import numpy as np
import hypnomodel
from dataset import Dataset
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from utils import print_validation
from utils import combine_predictions

def define_callbacks(file_name):
    filepath = file_name + "{epoch:02d}-{val_acc:.2f}.hdf5"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=0),
    ]
    return callbacks

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

    return X, Y

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    edf_path = config.get('paths', 'edf')
    pickle_path = config.get('paths', 'pickle')

    SIGNALS = ['eeg1', 'eeg2', 'emg', 'eogr', 'eogl']
    signals_rate = {'eeg1': 125, 'eeg2': 125, 'emg':125, 'eogr':50, 'eogl':50}
    time_window = 30 #time in seconds

    dataset = Dataset(edf_path, pickle_path)
    
    test_list = dataset.test_list()
    train_list = dataset.train_list()
    validation_list = dataset.validation_list()
   
    max_epochs = 50

    conv_layers = []
    conv_layers.append([(128,20), (128,20), 20, (256,20)])
    # for k in [64, 128, 256]:
    #     for f in [3, 25, 50, 75, 125, 250][-2:]: # run only with 2 last f
    #         conv_layers.append([(k, f)])

    # for k in [64, 128, 256]:
    #     for f in [5, 7, 11]:
    #         conv_layers.append([(k, f), (k, f)])

    # for k in [128]:
    #     for f in [10, 15, 20, 25, 30]:
    #         conv_layers.append([(k, f), (k, f), f, (k, f)])

    # for f in [3, 5, 7, 11, 15, 20, 25]:
    #     conv_layers.append([(64, f), 2, (128, f), 2, (256, f), 2, (512, f)])

    # for k in [64, 128]:
    #     for f in [3, 5, 7, 11, 15, 20, 25, 30]:
    #         conv_layers.append([(k, f), (k, f), f, (k, f), (k, f)])

    # for f in [30]:
    #     for pool in [30]:
    #         conv_layers.append([(256, f), pool, (512, f), pool])

    y_test = []
    preds = []
    input1_shape = Dataset.sample_shape()
    for layers in conv_layers:
        callbacks = define_callbacks('conv' + str(layers))
        model = hypnomodel.convModel(input1_shape=input1_shape, layers=layers)
        model.fit_generator(generator=dataset.generator(train_list, 10, "train"),
                    steps_per_epoch=dataset.steps_per_epoch(train_list, 10),
                    validation_data=dataset.generator(validation_list, 1, "validation"),
                    validation_steps=dataset.steps_per_epoch(validation_list, 1),
                    callbacks=callbacks,
                    epochs=max_epochs)
        for file_id in test_list:
            x_file, y_file = dataset.load_set([file_id])
            x_file, y_file = prepare_data(x_file, y_file, 30)
            preds_file = model.predict(x_file)
            preds_file = np.argmax(preds_file, axis=1)
            print("File ", file_id)
            print_validation(y_file.flatten(), preds_file)
            if not len(y_test):
                y_test = y_file
                preds = preds_file
            else:
                y_test = np.concatenate((y_test, y_file))
                preds = np.concatenate((preds, preds_file))
        print("All files:")
        print_validation(y_test.flatten(), preds)


if __name__ == '__main__':
    main()
