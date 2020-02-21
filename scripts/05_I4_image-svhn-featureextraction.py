import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load the digits dataset
digits = sio.loadmat(r'..\datasets\svhn-test_32x32.mat')

X = digits['X']
y = digits['y']
X = X.transpose((3,0,1,2))  
y = y[:,0]
# to improve the accuracy, more data for each class. (500-1000) images per class.
# 1797 is the MNIST instance number 
img_number = 10000   
X, y = resample(X, y, n_samples=img_number, stratify=y, random_state=0)


# Image: 32 x 32 pixels  
#plt.imshow(X[0,:,:,:])


#
# Intances of pixels 
#

#  32 x 32 pixels and 3 colours to 3072 attributes 
n_samples = len(X)
X_pixels = X.reshape((n_samples, -1))

# Create Test/Train Data
X_train, X_test, y_train, y_test = train_test_split(X_pixels, y, test_size=0.3, random_state=21, stratify=y)

# Instantiate a classifier
# clf = MLPClassifier(alpha=1, max_iter=1000)
# clf = KNeighborsClassifier(n_neighbors=3)
clf = RandomForestClassifier(n_estimators=10, random_state=1)

# Fit the classifier to the data
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)


print("Intances of pixels")

# Accuracy
print("Accuracy: {0:.2f}".format(accuracy_score(y_test, y_pred)))

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


#
# Intances of extracted features 
#


from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

X_features = pd.DataFrame(columns=list(range(0, 144))) # number of features

for i in range(0, img_number):
    img = X[i,:,:,:]

    grayscale = rgb2gray(img)
    grayscale = img_as_ubyte(grayscale)
    
    # Others - Histogram of Oriented Gradients (HOG)

    hog_features = hog(grayscale, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', feature_vector=True)

    X_features.loc[i] = hog_features


# Preparing the train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.3, random_state=21, stratify=y)


# clf = MLPClassifier(alpha=1, max_iter=1000)
# clf = KNeighborsClassifier(n_neighbors=3)
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("Intances of features")

# Accuracy
print("Accuracy: {0:.2f}".format(accuracy_score(y_test, y_pred)))

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


