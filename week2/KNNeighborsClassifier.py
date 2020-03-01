import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

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

# KNeighborsClassifier
t1 = time.time()
knn_model = KNeighborsClassifier(metric='minkowski', n_neighbors=5)
knn_model.fit(X_train, y_train)
t2 = time.time()
t12 = round(t2 - t1, 3)
print("Training time: ", t12)

# test prediction
t3 = time.time()
y_pred = knn_model.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
t4 = time.time()
t34 = round(t4 - t3, 3)
print("Prediction time: ", t34)

# Accuracy
acc = knn_model.score(X_test, y_test)
print("Accuracy: {0:.2f}".format(acc))