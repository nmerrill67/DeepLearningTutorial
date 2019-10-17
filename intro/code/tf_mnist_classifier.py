#!/usr/bin/env python3

import tensorflow as tf
mnist = tf.keras.datasets.mnist

# Load the mnist dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a model with only fully-connected layers.
# One hidden layer with 512 neurons and the ReLU nonlinear activation
# One last layer to map the data to 10 dimensions (for ten digits)
# and softmax to transform the distribution to the estimated categorical 
# distribution.
# Use dropout to randomly drop connections to zero and improve testing accuracy
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Create optimizer and loss
model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

# Train the  model for 5 epochs
model.fit(x_train, y_train, epochs=5)
# Evaluate on the test dataset
model.evaluate(x_test, y_test)
