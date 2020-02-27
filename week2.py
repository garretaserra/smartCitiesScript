# Deliverable S2
# Classifiers

import pandas as pd
import numpy as np
import time
from sklearn import datasets
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.tree.export import export_text
from sklearn.tree import plot_tree
from scripts.util import plot_decision_regions

# Read the data from the CSV file
data = pd.read_csv("datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")

# 2) Tractem les dades com categories.
data['ID'] = data['ID'].astype('category')
data['ID'] = data['ID'].cat.codes

# Print general information of the data
print('Data information: ', data.info())
print('Data description\n', data.describe())
print('Size of data: ', data.shape)

# Split in train and test datasets
X = data.drop('ID', axis=1)
y = data['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Class labels:', np.unique(y))
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))


# KNeighborsClassifier
# Create a k-NN classifier with 10 neighbors (Creamos el clasificador)
knn = KNeighborsClassifier(n_neighbors=10)
# Fit the classifier to the data (Entrenar el clasificador)
knn.fit(X_train, y_train)  # Se pasan los datos de training y las etiquetas correspondientes.



# GaussianNB
# SVC
# MLPClassifier
