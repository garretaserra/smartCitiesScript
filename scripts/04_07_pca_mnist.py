import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Evaluate Naive Bayes classifier with original dataset
clf = clf =  GaussianNB()
clf.fit(X_train, y_train)

print("Number of features: {}".format(X.shape[1]))
print("Accuracy: {0:.2f}".format(clf.score(X_test, y_test)))

### PCA to Speed-up Machine Learning Algorithms

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardizing the features
scaler = StandardScaler()
scaler.fit(X_train) 
sX_train = scaler.transform(X_train)
sX_test = scaler.transform(X_test)

# choose the minimum number of principal components such that 95% of the variance is retained.
pca = PCA(.95)
pca.fit(sX_train)
#print("Number of features: {}".format(pca.n_components_))
#print("Variance ratio: {}".format(pca.explained_variance_ratio_))

#Apply the transform)to both the training set and the test set
pcaX_train = pca.transform(sX_train)
pcaX_test = pca.transform(sX_test)

# Evaluate Naive Bayes classifier with reduced dataset
clf = clf =  GaussianNB()
clf.fit(pcaX_train, y_train)

print("Number of features: {}".format(pcaX_train.shape[1]))
print("Accuracy: {0:.2f}".format(clf.score(pcaX_test, y_test)))
