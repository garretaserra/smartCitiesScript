from NaiveBayes import naive_bayes
from DecisionTreeClassifier import decision_tree_classifier
from MultiLayerPerceptron import multi_layer_perceptron
from SupportVectorMachine import support_vector_machine
from KNNeighborsClassifier import k_nearest_neighbors
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data from the CSV file
data = pd.read_csv("../datasets/UJIIndoorLoc/UJIIndoorLoc_B0-ID-01.csv")

# Remove noise
data = data.loc[:, (data != 0).any(axis=0)]

# Split in train and test data sets
X = data.drop('ID', axis=1)
y = data['ID']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

result = naive_bayes(X_train, X_test, y_train, y_test)
print(result)
result = decision_tree_classifier(X_train, X_test, y_train, y_test, 50)
print(result)
result = multi_layer_perceptron(X_train, X_test, y_train, y_test)
print(result)
result = support_vector_machine(X_train, X_test, y_train, y_test)
print(result)
result = k_nearest_neighbors(X_train, X_test, y_train, y_test, 5)
print(result)
