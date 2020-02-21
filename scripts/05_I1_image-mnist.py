from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = datasets.load_digits()

# Image: 8 x 8 B&W pixels  
print(digits.images[0])
plt.imshow(digits.images[0], cmap=plt.cm.gray_r)

# 8x8 pixels to 64 attributes 
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Create Test/Train Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Instantiate a classifier
clf =  MLPClassifier(alpha=1, max_iter=1000)

# Fit the classifier to the data
clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy: {0:.2f}".format(accuracy_score(y_test, y_pred)))

# Confusion matrix
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


