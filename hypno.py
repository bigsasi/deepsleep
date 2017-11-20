try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import numpy as np
import hypnomodel
from dataset import Dataset
from dataset import prepare_data
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


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    edf_path = config.get('paths', 'edf')
    pickle_path = config.get('paths', 'pickle')

    dataset = Dataset(edf_path, pickle_path)
    
    test_list = dataset.test_list()
    train_list = dataset.train_list()
    validation_list = dataset.validation_list()
   
    dataset.print_set(train_list, "train")
    dataset.print_set(validation_list, "validation")
    dataset.print_set(test_list, "test")

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

<<<<<<< HEAD
    test_durations = dataset.edf_duration[test_list]

=======
    y_test = []
    preds = []
    input1_shape = Dataset.sample_shape()
>>>>>>> origin/data-generator
    for layers in conv_layers:
        callbacks = define_callbacks('conv' + str(layers))
        model = hypnomodel.convModel(input1_shape=input1_shape, layers=layers)
        model.fit_generator(generator=dataset.generator(train_list, 30, "train"),
                    steps_per_epoch=dataset.steps_per_epoch(train_list, 30),
                    validation_data=dataset.generator(validation_list, 1, "validation"),
                    validation_steps=dataset.steps_per_epoch(validation_list, 1),
                    callbacks=callbacks,
                    epochs=max_epochs)
        for file_id in test_list:
            x_file, y_file = dataset.load_set([file_id])
            x_file, y_file = prepare_data(x_file, y_file)
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
<<<<<<< HEAD

        for (i, file_id) in enumerate(test_list):
            first_second = np.sum(test_durations[:i])
            last_second = first_second + test_durations[i]
            first_epoch = first_second // 30
            last_epoch = last_second // 30
            file_preds = preds[first_epoch:last_epoch] 
            file_y_test = y_test[first_epoch:last_epoch]
            print_validation(file_y_test.flatten(), file_preds)
        
    lstm_layers = []
    # lstm_layers.append([16])
    # lstm_layers.append([32])
    # lstm_layers.append([64])
    # lstm_layers.append([32, 16])
    # lstm_layers.append([64, 32])
    
    for conv_layer in conv_layers:
        for lstm_layer in lstm_layers:
            callbacks = define_callbacks('convlstm' + str(conv_layer) + str(lstm_layer))
            model = hypnomodel.convLstmModel(input1_shape=(x_train.shape[1], x_train.shape[2]), 
                conv_layers=conv_layer, lstm_layers=lstm_layer)
            model.fit(x_train, y_cat, epochs=max_epochs, validation_split=0.1, callbacks=callbacks)
            preds = model.predict(x_test)
            preds = np.argmax(preds, axis=1)
            print_validation(y_test.flatten(), preds)
=======
>>>>>>> origin/data-generator


if __name__ == '__main__':
    main()
