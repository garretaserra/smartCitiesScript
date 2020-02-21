'''
Images adapted from: http://app.visgraf.impa.br/database/faces/

INSTRUCTIONS
1. Añadir una segunda capa oculta de 64 filtros, kernel de 3x3 y poolsize de 2x2 
2. Añadir una tercera capa oculta de 128 filtros, kernel de 3x3 y poolsize de 2x2 
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#import tensorflow as tf  
#import os
#
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
#tf.device('/gpu:0')
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# set data folders
train_data_dir = '..\datasets\FacesDB\Train' 
test_data_dir = '..\datasets\FacesDB\Test'

#Set number of samples
num_categories = 4
nb_train_samples = 27 * num_categories
nb_test_samples = 9 * num_categories

# Set epochs and batch size
num_epochs = 100
batch_size = 5


### PREPARE DATA              

img_width = 128
img_height = 128
num_channels = 3

input_shape = (img_width, img_height, num_channels)

train_datagen = ImageDataGenerator(
        rescale = 1. / 255, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(
        train_data_dir, 
        target_size = (img_width, img_height), 
        batch_size = batch_size, 
        class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1. / 255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir, 
        target_size = (img_width, img_height), 
        batch_size = batch_size, 
        class_mode = 'categorical',
        shuffle = False)


### BUILD THE MODEL

model = Sequential()

#Primera capa oculta de 32 filtros, kernel de 3x3 y poolsize de 2x2 
model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(num_categories)) 
model.add(Activation('sigmoid')) 
model.summary()

model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'rmsprop', 
        metrics = ['accuracy'])


### FIT THE MODEL

history = model.fit_generator(
        train_generator, 
        steps_per_epoch = nb_train_samples // batch_size, 
        epochs = num_epochs,
        validation_data = test_generator,
        validation_steps = nb_test_samples // batch_size)


### PREDICT

Y_pred = model.predict_generator(
        test_generator, 
        nb_test_samples // batch_size+1)


### EVALUATION

y_pred = np.argmax(Y_pred, axis=1) 
print('Matriz de confusión')
print(confusion_matrix(test_generator.classes, y_pred))

plt.figure(figsize=[8, 6]) 
plt.plot(history.history['loss'], 'r', linewidth=3.0) 
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Pérdidas de entrenamiento', 'Pérdidas de validación'], fontsize=24)
plt.xlabel('Epocas ', fontsize=22)
plt.ylabel('Pérdidas', fontsize=22)
plt.ylim(0,7)
plt.title('Curvas de pérdidas', fontsize=22) 
plt.show()

plt.figure(figsize=[8, 6]) 
plt.plot(history.history['accuracy'], 'r', linewidth=3.0) 
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Precisión de entrenamiento', 'Precisión de validación'], fontsize=24)
plt.xlabel('Epocas ', fontsize=22)
plt.ylabel('Precisión', fontsize=22) 
plt.ylim(0,1)
plt.title('Curvas de precisión', fontsize=22)
plt.show()


