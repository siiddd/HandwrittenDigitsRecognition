#Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

#Import MNIST Digits Dataset
df = keras.datasets.mnist.load_data()

x_train = df[0][0]
y_train = df[0][1]

x_test = df[1][0]
y_test = df[1][1]

#Checking the Shape of the Datasets
x_train.shape

#Visualize the Data
import matplotlib.pyplot as plt
plt.matshow(x_train[1])
y_train[1]

#Flatten the Data from (60000, 28, 28) to (60000, 784)
x_train_flat = x_train.reshape(60000, 28*28)
x_test_flat = x_test.reshape(10000, 28*28)

#Scale the Data and check if there is an improvement in performance
from sklearn.preprocessing import normalize
x_train_normalized = normalize(x_train_flat)
x_test_normalized = normalize(x_test_flat)

#Build a Model with HIDDEN LAYERS
model = keras.Sequential([
                            keras.layers.InputLayer(input_shape = (784,)),
                            keras.layers.Dense(units = 500, activation = 'relu'),
                            keras.layers.Dense(units = 10, activation = 'sigmoid')
                        ])

#Compile the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#Fit our data into the model
data = model.fit(x_train_normalized, y_train, epochs = 20)

#Check for Perfromance
model.evaluate(x_test_normalized, y_test)

#Create a Confusion Matrix to Evaluate Perfromance
from sklearn.metrics import confusion_matrix
y_predicted_normalized = model.predict(x_test_normalized) #Array
y_predicted_list_normalized = []

for x in y_predicted_normalized:
    y_predicted_list_normalized.append(np.argmax(x))
    
cm_normalized = confusion_matrix(y_test,y_predicted_list_normalized)

#Visualize the Epochs vs Accuracy
plt.plot(data.history['accuracy'])
plt.plot(data.history['val_accuracy'])
plt.show()
























