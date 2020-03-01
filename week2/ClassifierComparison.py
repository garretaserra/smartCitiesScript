from NaiveBayes import naive_bayes
from DecisionTreeClassifier import decision_tree_classifier
from MultiLayerPerceptron import multi_layer_perceptron
from SupportVectorMachine import support_vector_machine
from KNNeighborsClassifier import k_nearest_neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("../datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")

# Remove noise
data = data.loc[:, (data != 0).any(axis=0)]

# Split in train and test data sets
X = data.drop('ID', axis=1)
y = data['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

classifiers = []
accuracy = []
train_times = []
prediction_times = []

result = naive_bayes(X_train, X_test, y_train, y_test)
print(result)
classifiers.append('Naive Bayes')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
result = decision_tree_classifier(X_train, X_test, y_train, y_test, 50)
print(result)
classifiers.append('Decision Tree Classifier')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
result = multi_layer_perceptron(X_train, X_test, y_train, y_test)
print(result)
classifiers.append('Multi-Layer Perceptron')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
result = support_vector_machine(X_train, X_test, y_train, y_test)
print(result)
classifiers.append('Support Vector Machine')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
result = k_nearest_neighbors(X_train, X_test, y_train, y_test, 5)
print(result)
classifiers.append('K-Nearest Neighbors')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])

plt.figure(figsize=(10, 6), dpi=300)
plt.bar(classifiers, accuracy)
plt.suptitle("Prediction accuracy")
axes = plt.gca()
axes.set_ylim([0, 100])
plt.show()
