import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time

import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("../datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")

# Remove noise
data = data.loc[:, (data != 0).any(axis=0)]

# Print general information of the data
print('Data information: ', data.info())
print('Data description\n', data.describe())
print('Size of data: ', data.shape)

# Split in train and test datasets
X = data.drop('ID', axis=1)
y = data['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Train model
t1 = time.time()
neuralNetworkModel = MLPClassifier(max_iter=1000)
neuralNetworkModel.fit(X_train, y_train)
t2 = time.time()
t12 = round(t2-t1, 5)
print('Time used for training with a size of', len(X_train), 'is', t12)

# test prediction
t1 = time.time()
y_pred = neuralNetworkModel.predict(X_test)
t2 = time.time()
t12 = round(t2-t1, 5)
print('Time for prediction with size', len(X_test), 'is: ', t12)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f%%' % (100.0 * neuralNetworkModel.score(X_test, y_test)))