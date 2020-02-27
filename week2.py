import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the data from the CSV file
data = pd.read_csv("datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")

# Split in train and test datasets
X = data.drop('ID', axis=1)
y = data['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print('Class labels:', np.unique(y))
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# KNeighborsClassifier
# DecisionTreeClassifier
# GaussianNB
# SVC
# MLPClassifier