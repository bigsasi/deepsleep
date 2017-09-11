from keras.models import Sequential
from keras.layers import GlobalAveragePooling1D, MaxPooling1D, Dense,Dropout,MaxPool1D, Conv1D, GlobalMaxPool1D, Activation, Bidirectional, LSTM
from keras import backend as K
from keras.optimizers import Adam
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def createLinearModel(X, Y):
    model = Sequential()
    model.add(Dense(input_shape = (X.shape[1],), units = 1, activation='softmax'))
    model.compile(optimizer = 'adam', loss='hinge',metrics=['acc'])
    print(model.summary())
    return model

def createMLPModel(X, Y):
        model = Sequential()
        model.add(Dense(input_shape = (X.shape[1],), units = 256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer = 'adam', loss='categorical_crossentropy',metrics=['acc'])
        print(model.summary())
        return model

def runModel(X_train, Y_train, X_test, Y_test, model):
    model.fit(X_train, Y_train, epochs = 100, validation_data=(X_test, Y_test))
    #print("Evaluating...", model.metrics)
    #results = model.evaluate(X_test, Y_test)
    #print("Results:", results)

def createDimmuLinearModel(X, Y):
    model = Sequential()
    model.add(Dense(input_shape = (X.shape[1],), units = 1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss='binary_crossentropy',metrics=['acc'])
    print(model.summary())
    return model

def createDimmuMLPModel(X, Y):
    model = Sequential()
    model.add(Dense(input_shape = (X.shape[1],), units = 128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense( units = 1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss='binary_crossentropy',metrics=['acc',precision,recall])
    print(model.summary())
    return model

def lstmModel(x_train, y_train,input1_shape,batch_size=64 ):
    
    model = Sequential()
    model.add(LSTM(64,input_shape=input1_shape,return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                metrics=['accuracy',precision,recall], 
              optimizer=Adam(lr=3e-4))
    print(model.inputs)
    print(model.summary())
    return model

def convModel(x_train, y_train, input_shape, max_features=20000, n_epochs=3, class_weight={0: 1.0, 1:1.0}, batch_size=256, kernel_size=125, pool_size=124, filters=256 ):
    model = Sequential()
    model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=input_shape))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    model.compile(loss='binary_crossentropy',
                metrics=['accuracy',precision,recall], 
              optimizer=Adam(lr=3e-4))
    print(model.inputs)
    print(model.summary())
    # print("x1_train shape:",x1_train.shape,"x2_train shape:",x2_train.shape, "y_train shape:", y_train.shape )

    # model.fit([x1_train,x2_train], y_train,
    #       batch_size=128,#keras should split those in 2 minibatches of 64 per gpu 
    #       epochs=n_epochs,
    #       #validation_split=0.2,
    #       #validation_data=(x1_val,x2_val, y_val))#,
    #       class_weight=class_weight)
    return model

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

def properConv(X, Y,input_shape):
    model = Sequential()
    model.add(Conv1D(64, 30, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 10, activation='relu'))
    model.add(MaxPooling1D(30))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                metrics=['accuracy',precision,recall], 
              optimizer=Adam(lr=3e-4))
    print(model.inputs)
    print(model.summary())
    return model