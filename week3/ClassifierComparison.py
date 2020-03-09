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
data = data.sample(frac=1)

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
cv_scores = [[], [], [], [], []]

result = naive_bayes(X_train, X_test, y_train, y_test, X, y)
print(result)
classifiers.append('Naive Bayes')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
i = 0
for res in result[3]:
    cv_scores[i].append(res)
    i += 1

result = decision_tree_classifier(X_train, X_test, y_train, y_test, 50, X, y)
print(result)
classifiers.append('Decision Tree Classifier')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
i = 0
for res in result[4]:
    cv_scores[i].append(res)
    i += 1

result = multi_layer_perceptron(X_train, X_test, y_train, y_test, X, y)
print(result)
classifiers.append('Multi-Layer Perceptron')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
i = 0
for res in result[3]:
    cv_scores[i].append(res)
    i += 1

result = support_vector_machine(X_train, X_test, y_train, y_test, X, y)
print(result)
classifiers.append('Support Vector Machine')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
i = 0
for res in result[3]:
    cv_scores[i].append(res)
    i += 1

result = k_nearest_neighbors(X_train, X_test, y_train, y_test, 5, X, y)
print(result)
classifiers.append('K-Nearest Neighbors')
accuracy.append(result[0])
train_times.append(result[1])
prediction_times.append(result[2])
i = 0
for res in result[3]:
    cv_scores[i].append(res)
    i += 1

# Accuracy chart
plt.figure(figsize=(10, 6), dpi=500)
plt.bar(classifiers, accuracy)
plt.title("Prediction accuracy")
axes = plt.gca()
# Set y axis to go from 0 to 100
axes.set_ylim([0, 100])
plt.show()

# Cross Validation Chart
fig, ax = plt.subplots(figsize=(12, 6), dpi=500)
x = np.arange(len(classifiers))

ax.bar(x-0.32, cv_scores[0], width=0.15, color='b')
ax.bar(x-0.16, cv_scores[1], width=0.15, color='b')
ax.bar(x, cv_scores[2], width=0.15, color='b')
ax.bar(x+0.16, cv_scores[3], width=0.15, color='b')
ax.bar(x+0.32, cv_scores[4], width=0.15, color='b')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
plt.show()

#
# # Timings chart
# def auto_label(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
#
#
# fig, ax = plt.subplots(figsize=(12, 6), dpi=500)
# x = np.arange(len(classifiers))  # the label locations
# width = 0.35  # the width of the bars
#
# rects1 = ax.bar(x - width/2, train_times, width=width, label='Training times', color='Blue')
# rects2 = ax.bar(x + width/2, prediction_times, width=width, label='Prediction time', color='Orange')
#
# ax.set_yscale('log')
# ax.set_xticks(x)
# ax.set_xticklabels(classifiers)
# ax.legend()
#
# # Add labels with values to chart
# auto_label(rects1)
# auto_label(rects2)
#
# plt.title("Times of operation (s)")
#
# # Decision Tree max depth
# accuracy = []
# train_times = []
# prediction_times = []
#
# for i in range(1, 23):     # Max depth for this case
#     result = decision_tree_classifier(X_train, X_test, y_train, y_test, i)
#     accuracy.append(result[0])
#     train_times.append(result[1])
#     prediction_times.append(result[2])
#
# plt.figure(figsize=(10, 6), dpi=500)
# plt.plot(range(1, 23), accuracy, linewidth=3)
# plt.title("Decision Tree Classifier Accuracy (%)")
# plt.xlabel("Max Depth")
# axes = plt.gca()
# axes.set_ylim([0, 100])     # Set y axis to go from 0 to 100
#
# # KNNeighbors
# accuracy = []
# train_times = []
# prediction_times = []
#
# for i in range(1, 22):     # Max depth for this case
#     result = k_nearest_neighbors(X_train, X_test, y_train, y_test, i)
#     accuracy.append(result[0])
#     train_times.append(result[1])
#     prediction_times.append(result[2])
# plt.figure(figsize=(10, 6), dpi=500)
# plt.plot(range(1, 22), accuracy, linewidth=3)
# plt.title("K-Nearest Neighbor Classifier Accuracy (%)")
# plt.xlabel("K")
# axes = plt.gca()
# axes.set_ylim([0, 100])     # Set y axis to go from 0 to 100
plt.show()
