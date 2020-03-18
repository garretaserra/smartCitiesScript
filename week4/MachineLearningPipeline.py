# Week 4

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from time import time
from imblearn.over_sampling import RandomOverSampler


# Read the data from the CSV file
data = pd.read_csv("../datasets/UJIIndoorLoc/UJIIndoorLoc_ID.csv")

print('Unique Classes: ', len(data['ID'].unique()))

# Pre-Processing
# Randomize input samples
data = data.sample(frac=1)

# Remove noise
print('Size before removing noise:', data.shape)
data = data.loc[:, (data != 100).any(axis=0)]
print('Size after removing noise: ', data.shape)

# Find duplicates
data = data.drop_duplicates()
print('Size after removing duplicates: ', data.shape)

# Differentiate columns
X = data.drop('ID', axis=1)
y = data['ID'].astype('category')

# Oversampling (make sure to have at least 2 instances of each class)
ros = RandomOverSampler(random_state=20, sampling_strategy='minority')
while y.value_counts().le(10).any():
    X, y = ros.fit_resample(X, y)

print('Size after oversampling: ', X.shape)

# Replace 100 by -120
no_connection_val = -120    # Value that will be set when not connected
X.replace(100, no_connection_val, inplace=True)

# Normalise data
x_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(x_scaled)
# print(X.describe())

# Split into train and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Train data
neural_network_model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(465, 735))
start = time()
neural_network_model.fit(X_train, y_train)
end = time()
print('Time taken: ', round(end-start, 3)/60)

# Test data
y_pred = neural_network_model.predict(X_test)

accuracy = 100.0 * neural_network_model.score(X_test, y_test)
print('Accuracy(%): ', accuracy)
