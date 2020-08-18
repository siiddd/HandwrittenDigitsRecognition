#Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

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

#Build a Simple Neural Network
model = keras.Sequential([
                            keras.layers.InputLayer(input_shape = (784,)),
                            keras.layers.Dense(units = 10, activation = 'sigmoid')
                        ])

#Compile the Model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

#Fit our Data into the Model
model.fit(x_train_flat, y_train, epochs = 10)

#Check the Performance on the Test Data
model.evaluate(x_test_flat, y_test)

#Compare the Predictions with Actual Data
plt.matshow(x_test[10])                      #Visual Representation
y_predicted = model.predict(x_test_flat)    #Array of all the Predictions


#Create a Confusion Matrix
from sklearn.metrics import confusion_matrix

y_predicted_list = [] #Create an Empty Array

for x in y_predicted:
    y_predicted_list.append(np.argmax(x)) #Select the Output with the highest Probability Value
    
cm = confusion_matrix(y_test, y_predicted_list) #Confusion Matrix

#Visualize the Confusion Matrix on a HeatMap
sns.heatmap(cm, annot = True)
fig = plt.gcf()
fig.set_size_inches(15,15)
plt.savefig(r'C:\Users\nsid4\Desktop\Confusion_Matrix.png')





























