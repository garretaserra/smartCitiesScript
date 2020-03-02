from NaiveBayes import naive_bayes
from DecisionTreeClassifier import decision_tree_classifier
from MultiLayerPerceptron import multi_layer_perceptron
from SupportVectorMachine import support_vector_machine
from KNNeighborsClassifier import k_nearest_neighbors
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

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

# Accuracy chart
plt.figure(figsize=(10, 6), dpi=500)
plt.bar(classifiers, accuracy)
plt.title("Prediction accuracy")
axes = plt.gca()
# Set y axis to go from 0 to 100
axes.set_ylim([0, 100])
plt.show()


# Timings chart
def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')


fig, ax = plt.subplots(figsize=(12, 6), dpi=500)
x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width/2, train_times, width=width, label='Training times', color='Blue')
rects2 = ax.bar(x + width/2, prediction_times, width=width, label='Prediction time', color='Orange')

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()

# Add labels with values to chart
auto_label(rects1)
auto_label(rects2)

plt.title("Times of operation (s)")
plt.show()

# Decision Tree max depth
accuracy = []
train_times = []
prediction_times = []

for i in range(22):     # Max depth for this case
    result = decision_tree_classifier(X_train, X_test, y_train, y_test, 50)
    accuracy.append(result[0])
    train_times.append(result[1])
    prediction_times.append(result[2])
