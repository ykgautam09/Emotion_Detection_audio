import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = './cleaned_data.csv'
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 40
SAVE_MODEL_PATH = './er_model.h5'
OUTPUT_KEYWORD = 6
MFCC_PATH='./cleaned_mfcc_feature.npy'


def load_dataset(data_path):
    data = pd.read_csv(data_path)
    x = np.load(MFCC_PATH,allow_pickle=True)
    y = np.array(data['emotion'])
    print(x[0])
    print(len(x), len(y))
    return x, y


def get_split_data(data_path):
    x, y = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1)

    # convert 2D to 3D
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    print(x_test.shape, x_validation.shape, x_train.shape, y_test.shape, y_validation.shape, y_train.shape)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, loss='sparse_categorical_crossentropy'):
    model = keras.Sequential()

    # layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

    # layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

    # layer 3
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

    # # layer 4
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

    # flatten layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(OUTPUT_KEYWORD, activation='softmax'))

    # compile
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=loss, metrics='accuracy')

    # print model details
    model.summary()

    return model


def train(data_path, learning_rate):
    x_train, x_validation, x_test, y_train, y_validation, y_test = get_split_data(data_path)

    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape, learning_rate)

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_validation, y_validation))

    test_error, test_accuracy = model.evaluate(x_test, y_test)
    print(f'test_error:{test_error} ,test_accuracy:{test_accuracy}')

    model.save(SAVE_MODEL_PATH)


if __name__ == '__main__':
    train(DATA_PATH, LEARNING_RATE)
