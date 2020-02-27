# Entregable S2
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

t1 = time.time()
# KNeighborsClassifier
# Create a k-NN classifier with 5 neighbors (Creamos el clasificador)
# Fit the classifier to the data (Entrenar el clasificador)
# model train metric='euclidean', n_neighbors=1
knn_model = KNeighborsClassifier(metric='minkowski', n_neighbors=5)
knn_model.fit(X_train, y_train)
# Se pasan los datos de training y las etiquetas correspondientes.
t2 = time.time()
t12 = round(t2 - t1, 3)

print("Training time: ", t12)

t3 = time.time()
# test prediction
y_pred = knn_model.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
t4 = time.time()
t34 = round(t4 - t3, 3)
print("Prediction time: ", t34)


# Accuracy
acc = knn_model.score(X_test, y_test)
print("Accuracy: {0:.2f}".format(acc))

# Predict the labels for the test data
print("Test set good labels: {}".format(y_test))
print("Test set predictions: {}".format(y_pred))
print('Well classified samples: {}'.format((y_test == y_pred).sum()))
print('Misclassified samples: {}'.format((y_test != y_pred).sum()))


# plot_decision_regions(X_combined, y_combined, classifier=knn_model, test_idx=range(train_len, combined_len))
# plt.xlabel('petal length [cm]')
# plt.ylabel('petal width [cm]')
# plt.legend(loc='upper left')
# plt.tight_layout()



# DecisionTreeClassifier
# tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=1)
# tree_model.fit(X_train, y_train)
# # test prediction
# y_pred = tree_model.predict(X_test)
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.2f%%' % (100.0 * tree_model.score(X_test, y_test)))

# decision boundary
# X_combined = np.vstack((X_train, X_test))
# y_combined = np.hstack((y_train, y_test))
# train_len = X_train.shape[0]
# combined_len = X_combined.shape[0]

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
