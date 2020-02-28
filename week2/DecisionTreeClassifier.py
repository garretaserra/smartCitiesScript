# Deliverable S2
# Classifiers
# DecisionTreeClassifier

import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("../datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")

# Print general information of the data
print('Data information: ', data.info())
print('Data description\n', data.describe())
print('Size of data: ', data.shape)

# Split in train and test datasets
X = data.drop('ID', axis=1)
y = data['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print('Starting Decision Tree Classifier')
t1 = time.time()
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=1)
tree_model.fit(X_train, y_train)
t2 = time.time()
t12 = round(t2-t1, 5)
print('Time used for training with a max depth of', tree_model.max_depth, 'and size of', len(X_train), 'is', t12)

# test prediction
print('Start prediction time')
t1 = time.time()
y_pred = tree_model.predict(X_test)
t2 = time.time()
t12 = round(t2-t1, 5)
print('Time for prediction with size', len(X_test), 'is: ', t12)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f%%' % (100.0 * tree_model.score(X_test, y_test)))

# decision boundary
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
train_len = X_train.shape[0]
combined_len = X_combined.shape[0]

# The decision plot can not be represented because it would have 520 dimensions

# Uncomment next line to print decision tree
# print(export_text(tree_model, feature_names=list(X.columns)))
plt.figure(figsize=(6, 6), dpi=300)
# Uncomment next line to plot the decision tree
# plot_tree(tree_model, filled=True)
plt.show()
