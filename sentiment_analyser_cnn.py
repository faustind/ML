import numpy as np
from tensorflow import keras
from keras import layers
from keras import models
from keras.utils import np_utils
from keras.optimizers import RMSprop

MAX_DS_SIZE = 10_000
DATA_PATH = './data/data'
MAX_TEXT_LENGTH = 1014 # max text length
ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\|_@#$%ˆ&*˜`+-=<>()[]{}\n'

def sentiment_classifier():
    """Build the large model for sentiment classification."""
    classifier = models.Sequential()
# layer 1
    classifier.add(layers.Conv1D(1024, 7, activation='relu',
                   input_shape=(1014,70)))
    classifier.add(layers.MaxPooling1D((3,)))

# layer 2
    classifier.add(layers.Conv1D(1024, 7, activation='relu'))
    classifier.add(layers.MaxPooling1D((3,)))

# layer 3, 4, 5, 6
    classifier.add(layers.Conv1D(1024, 7, activation='relu'))
    classifier.add(layers.Conv1D(1024, 7, activation='relu'))
    classifier.add(layers.Conv1D(1024, 7, activation='relu'))

    classifier.add(layers.Conv1D(1024, 7, activation='relu'))
    classifier.add(layers.MaxPooling1D((3,)))

# flattening layer
    classifier.add(layers.Flatten())

    classifier.add(layers.Dense(2048, activation='relu'))

    classifier.add(layers.Dropout(rate=0.5))

    classifier.add(layers.Dense(2048, activation='relu'))

    classifier.add(layers.Dropout(rate=0.5))

# output layer
    classifier.add(layers.Dense(2, activation='sigmoid'))

    classifier.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])

    return classifier

def preprocess_data(file_path):
    """Return a nelem*70*1014 tensor of reviews in file_path."""

    # length of the label at the beginning of each line
    label_length = len('__label__i') # i in {1,2}

    labels = []

    # shape = n_data * nfeature * text_lenght
    dataset_tensor = np.zeros((MAX_DS_SIZE, len(ALPHABET), MAX_TEXT_LENGTH), dtype=np.short)

    with open(file_path) as f:
        for i in range(MAX_DS_SIZE):
            r = f.readline()

            labels.append(0 if r.startswith('__label__1') else 1)

            # take at most L_0 characters from the revesed review to lowercase
            r = reversed(r[label_length:].lower())
            review = list(r)[:MAX_TEXT_LENGTH]

            for j, char in enumerate(review):
                feature_pos = ALPHABET.find(char)
                if feature_pos > -1: # char in considered alphabet
                    dataset_tensor[i,feature_pos,j] = 1


    dataset_tensor = dataset_tensor.reshape((MAX_DS_SIZE, 1014, len(ALPHABET)))
    c_labels = np_utils.to_categorical(labels, 2)

    return (dataset_tensor, c_labels)

def train_classifier():

    classifier = sentiment_classifier()
    classifier.fit(
            trainning_samples, training_labels,
            validation_split=0.4,
            batch_size=32, nb_epoch=10,
            verbose=1)
    return classifier

if __name__ == '__main__':

    ds, labels = preprocess_data(DATA_PATH)

    trainning_samples = ds[:7500]
    training_labels = labels[:7500]

    test_samples = ds[7500:]
    test_labels = labels[7500:]

    classifier = train_classifier()

