import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = sio.loadmat(r'..\datasets\svhn-test_32x32.mat')
X = digits['X']
y = digits['y']
X = X.transpose((3,0,1,2))  
y = y[:,0]
# to improve the accuracy, more data for each class. (500-1000) images per class.
# 1797 is the MNIST instance number 
X = X[:1797,:] 
y = y[:1797] 

# Image: 32 x 32 pixels  
plt.imshow(X[0,:,:,:])

#  32 x 32 pixels and 3 colours to 3072 attributes 
n_samples = len(X)
X = X.reshape((n_samples, -1))

# Create Test/Train Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Instantiate a classifier
clf = MLPClassifier(alpha=1, max_iter=1000)

# Fit the classifier to the data
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)


# Accuracy
print("Accuracy: {0:.2f}".format(accuracy_score(y_test, y_pred)))

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


