import numpy as np
import scipy.signal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def print_validation(y_true, y_pred):
    classes = ['W', 'N1-2', 'N3', 'REM']
    conf_matrix = confusion_matrix(y_true, y_pred)
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    report = classification_report(y_true, y_pred, target_names=classes)
    print(conf_matrix)
    print(cm_normalized)
    print(report)

def tensor_padding(tensor, original_rate, expected_rate):
    """Pad the tensor sampled at original_rate with 0 to equal the
    size of a tensor sampled at expected_rate"""
    original_shape = tensor.shape
    new_timesteps = int(original_shape[1] / original_rate * expected_rate)
    if len(original_shape) == 2:
        new_shape = (original_shape[0], new_timesteps)
    else:
        new_shape = (original_shape[0], new_timesteps, original_shape[2])

    tmp = np.zeros(new_shape)

    if len(original_shape) == 2:
        tmp[:, -original_shape[1]:] = tensor
    else:
        tmp[:, -original_shape[1]:, :] = tensor

    return tmp

def tensor_resampling(tensor, original_rate, expected_rate):
    """Resample a tensor from original_rate to expected_rate"""
    original_shape = tensor.shape
    new_timesteps = int(original_shape[1] / original_rate * expected_rate)
    if len(original_shape) == 2:
        new_shape = (original_shape[0], new_timesteps)
    else:
        new_shape = (original_shape[0], new_timesteps, original_shape[2])

    flatten_tensor = tensor.flatten()
    tmp = scipy.signal.resample_poly(flatten_tensor, expected_rate, original_rate)
    return tmp.reshape(new_shape)

def merge_input(x_map, signals):
    """Concatenate inputs"""
    all_x = x_map[signals[0]]
    for signal in signals[1:]:
        all_x = np.concatenate((all_x, x_map[signal]), axis=2)

    return all_x

def combine_predictions(preds, option='mean'):
    """Combine prediction vectors using option"""
    new_pred = np.zeros(preds[0].shape)

    for p in preds:
        if option == 'mean':
            new_pred += p
        if option == 'max':
            new_pred = np.maximum(new_pred, p)
        if option == 'vote':
            votes_p = np.argmax(p, axis = 1)
            for (i, v) in enumerate(votes_p):
                new_pred[i, v] += 1
    
    if option == 'mean':
        return new_pred / len(preds)

    return new_pred