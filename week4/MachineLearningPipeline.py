# Week 4

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from time import time


# Read the data from the CSV file
data = pd.read_csv("../datasets/UJIIndoorLoc/UJIIndoorLoc_ID.csv")

print('Unique Classes: ', data['ID'].unique())

# Pre-Processing
# Randomize input samples
data = data.sample(frac=1)

no_connection_val = -120

# Remove noise
data.replace(100, no_connection_val, inplace=True)
print('Size before removing noise:', data.shape)
data = data.loc[:, (data != no_connection_val).any(axis=0)]
print('Size after removing noise: ', data.shape)

# Differentiate columns
X = data.drop('ID', axis=1)
y = data['ID'].astype('category')

# Normalise data
x_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(x_scaled)
# print(X.describe())

# Split into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Train data
neural_network_model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(465, 400, 300, 256))
start = time()
neural_network_model.fit(X_train, y_train)
end = time()
print('Time taken: ', round(end-start, 3))

# Test data
y_pred = neural_network_model.predict(X_test)

accuracy = 100.0 * neural_network_model.score(X_test, y_test)
print('Accuracy(%): ', accuracy)
