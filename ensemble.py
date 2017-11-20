import os
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import numpy as np
from dataset import prepare_data
from dataset import Dataset
from hypnomodel import precision, recall
from keras.models import load_model
from utils import print_validation

test_names = ['shhs2-200089', 'shhs2-200133',
                'shhs2-200136', 'shhs2-200145',
                'shhs2-200168', 'shhs2-200175',
                'shhs2-200182', 'shhs2-200191',
                'shhs2-200192', 'shhs2-200194',
                'shhs2-203637', 'shhs2-203650',
                'shhs2-203652', 'shhs2-203657',
                'shhs2-203704',
                'shhs2-203708', 'shhs2-203711',
                'shhs2-203766', 'shhs2-203773',
                'shhs2-203795', 
                'shhs2-201897', 
                'shhs2-201933', 'shhs2-201988',
                'shhs2-202015',
                'shhs2-202442', 'shhs2-202463',
                'shhs2-202465', 'shhs2-202483',
                'shhs2-202512', 'shhs2-204386',
                'shhs2-204387', 'shhs2-204391',
                'shhs2-204417', 'shhs2-204421',
                'shhs2-204430', 'shhs2-200313',
                'shhs2-200700', 'shhs2-200939',
                'shhs2-201118', 'shhs2-201326'
]

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    edf_path = config.get('paths', 'edf')
    pickle_path = config.get('paths', 'pickle')
    models_path = 'models'

    dataset = Dataset(edf_path, pickle_path)
    
    # test_list = dataset.test_list()
    test_list = dataset.name_to_id(test_names)
    train_list = dataset.train_list()
    validation_list = dataset.validation_list()

    dataset.print_set(train_list, "train")
    dataset.print_set(validation_list, "validation")
    dataset.print_set(test_list, "test")

    custom_objects = {'precision': precision,
                      'recall': recall}
    
    model_list = []
    for model_name in os.listdir(models_path):
        model_path = models_path + "/" + model_name
        model_list.append(load_model(model_path, custom_objects=custom_objects))
    
    num_models = len(model_list)
    print("Using {} models".format(num_models))
    
    y_test = []
    preds = []
    for (pos, file_id) in enumerate(train_list):
        x_file, y_file = dataset.load_set([file_id])
        x_file, y_file = prepare_data(x_file, y_file)
        preds_file = np.zeros((len(y_file), 4, num_models))
        for (i, model) in enumerate(model_list):
            preds_file[:, :, i] = model.predict(x_file)
        preds_file = np.mean(preds_file, axis=2)
        preds_file = np.argmax(preds_file, axis=1)
        # print("File {}: {} [{}]".format(pos, test_names[pos], file_id))
        print("File {}: [{}]".format(pos, file_id))
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

