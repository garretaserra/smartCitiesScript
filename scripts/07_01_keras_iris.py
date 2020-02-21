'''
Adapted from: https://medium.com/@andrewdoesdata/deep-learning-with-15-lines-of-python-397e43bcf843
'''

# conda install -c conda-forge tensorflow

from sklearn import datasets
from tensorflow import keras
from tensorflow.keras.layers import Dense

# 1) import data
iris = datasets.load_iris()

# 2) prepare inputs
input_x = iris.data

# 3) prepare outputs: a binary class matrix
output_y = keras.utils.to_categorical(iris.target, 3)

# 4a) Create the model
model = keras.models.Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 4b) Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4c) Fit the model
model.fit(input_x, output_y, epochs=150, batch_size=15, verbose=1)

# 4d) Evaluate the model
score = model.evaluate(input_x, output_y, batch_size=15)
score
