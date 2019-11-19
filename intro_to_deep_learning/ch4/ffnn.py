import pandas as pd
import numpy as np
import tensorflow as tf

TARGET_VARIABLE = "user_action"
TRAIN_TEST_SPLIT = 0.5
HIDDEN_LAYER_SIZE = 0.5

raw_data = pd.read_csv("data.csv")

mask = np.random.rand(len(raw_data)) < TRAIN_TEST_SPLIT
training_dataset = raw_data[mask]
test_dataset = raw_data[~mask]

training_data = np.array(training_dataset.drop(TARGET_VARIABLE, axis=1))
training_labels = np.array(training_dataset[[TARGET_VARIABLE]])

test_data = np.array(test_dataset.drop(TARGET_VARIABLE, axis=1))
test_labels = np.array(test_dataset[[TARGET_VARIABLE]])

# building the NN
ffnn = tf.keras.models.Sequential()
ffnn.add(tf.keras.layers.Flatten())
ffnn.add(tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="sigmoid")) # hidden layer and input layer
ffnn.add(tf.keras.layers.Dense(1, activation="sigmoid")) # output layer
ffnn.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])

# training
ffnn.fit(training_data, training_labels, epochs=150, batch_size=2, verbose=1)

# testing model
metrics = ffnn.evaluate(test_data, test_labels, verbose=1)
print("%s: %.2f%%" % (ffnn.metrics_names[1], metrics[1]*100))

# predicting on unseen data
new_data = np.array(pd.read_csv("new_data.csv"))
results = ffnn.predict(new_data)
print(results)
