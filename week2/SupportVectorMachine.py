# Deliverable S2
# Classifiers
# Support Vector Machine

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# test prediction
y_pred = svm_model.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f%%' % (100.0 * svm_model.score(X_test, y_test)))