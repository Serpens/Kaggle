#!/usr/bin/env python
import numpy as np
import pandas
import sklearn
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Convolution2D, Convolution1D, AveragePooling2D, MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier


def create_convoluted_keras_model():
    model = Sequential()
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
    return model


if __name__ == '__main__':
    data = pandas.read_csv('train.csv')
    labels = data['label'].values
    labels = np_utils.to_categorical(labels)
    features = data.iloc[:,1:].values
    features = features.reshape((features.shape[0], 1, 28, 28))

    nn_model = create_convoluted_keras_model()
    nn_model.fit(features, labels, nb_epoch=400, batch_size=512)
    nn_model.evaluate(features, labels)

    test_data = pandas.read_csv('test.csv').values
    test_data = test_data.reshape((test_data.shape[0], 1, 28, 28))
    predictions = nn_model.predict_classes(test_data)

    print('ImageId,Label')
    for i in enumerate(predictions):
        print(str(i[0]+1) + ',' + str(i[1]))
