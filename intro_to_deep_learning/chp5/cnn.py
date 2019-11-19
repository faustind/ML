# CNN example with MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


(train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()

# preprocess images

# add one dimension for futures feature maps
train_samples = train_samples.reshape(train_samples.shape[0], 28, 28, 1)
train_samples = train_samples.astype("float32")
train_samples = train_samples / 255 # scaling between 0 and 1

# add one dimension for futures feature maps
test_samples = test_samples.reshape(test_samples.shape[0], 28, 28, 1)
test_samples = test_samples.astype("float32")
test_samples = test_samples / 255 # scaling between 0 and 1

# one-hot encoding the labels
c_train_labels = np_utils.to_categorical(train_labels, 10)
c_test_labels = np_utils.to_categorical(test_labels, 10)

# build the model and save it to disk
def build_convnet():
    convnet = Sequential()
    convnet.add(Convolution2D(32, 4, 4, activation='relu', input_shape=(28, 28, 1)))
    convnet.add(MaxPooling2D(pool_size=(2,2)))
    convnet.add(Convolution2D(32, 3, 3, activation='relu'))
    convnet.add(MaxPooling2D(pool_size=(2,2)))
    convnet.add(Dropout(0.3))
    convnet.add(Flatten())
    convnet.add(Dense(10, activation='softmax'))

    convnet.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    convnet.fit(train_samples, c_train_labels, batch_size=32, nb_epoch=20, verbose=1)
    convnet.save('mnist_recognition.h5')

# uncomment to see evaluation metrics on test data
# print("Evaluate on test data")
# test_loss, test_acc = convnet.evaluate(test_samples, c_test_labels, verbose=1)
# print("test accuracy: {:.2f}%".format(test_acc*100))

def show_feature_maps():
    """Displays the feature maps from a random image from the test data."""
    # would like to visualize features map up to the 4th layers
    try:
        convnet = load_model('mnist_recognition.h5')
    except Exception:
        print("Couldn't load model from file mmnist_recognition.h5")
        return
    layer_outputs = [layer.output for layer in convnet.layers[:4]]
    activation_model = models.Model(inputs=convnet.input, outputs=layer_outputs)

    # run the model on a random image
    img_tensor = test_samples[np.random.randint(0, len(test_samples))]
    img_tensor = img_tensor.reshape(1, 28, 28, 1)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in convnet.layers[:4]:
        layer_names.append(layer.name)

    images_per_row = 8

    for layer_name, layer_activation in zip(layer_names, activations):
        # feature maps have shape (_, size, size, n_features)
        n_features = layer_activation.shape[-1] # number of feature maps
        size = layer_activation.shape[1] # of one feature map

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                display_grid[col  * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap=plt.cm.binary)

