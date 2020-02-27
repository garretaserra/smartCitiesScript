# Entregable S2
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




# DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
tree_model.fit(X_train, y_train)
# test prediction
y_pred = tree_model.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f%%' % (100.0 * tree_model.score(X_test, y_test)))

# decision boundary
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
train_len = X_train.shape[0]
combined_len = X_combined.shape[0]

# plt.figure(figsize=(3, 3), dpi=300)
# plot_decision_regions(X=X_combined, y=y_combined, classifier=tree_model, test_idx=range(train_len, combined_len))
# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# # plt.savefig('images/03_01.png', dpi=300)
# plt.show()

# GaussianNB
# SVC
# MLPClassifier
