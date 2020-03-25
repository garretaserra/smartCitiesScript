from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import os
import matplotlib.pyplot as plt
from math import ceil
from time import time
from tensorflow_core.python.keras.saving.save import load_model
from keras import regularizers


images_dir = './images/dogs'

# Calculate training images
train_beagles_dir = os.path.join(images_dir, 'beagles')  # directory with our training beagle pictures
train_yorkshires_dir = os.path.join(images_dir, 'yorkshires')  # directory with our training yorkshire pictures
num_beagles = len(os.listdir(train_beagles_dir))
num_yorkshires = len(os.listdir(train_yorkshires_dir))
total_train = num_beagles + num_yorkshires
print('total training beagle images:', num_beagles)
print('total training yorkshire images:', num_yorkshires)
print('total training images:', total_train)

# Calculate validation images
validate_beagles_dir = os.path.join(images_dir, 'jakes')  # directory with our validation jakes pictures
validate_yorkshires_dir = os.path.join(images_dir, 'trufis')  # directory with our validation trufis pictures
num_jakes = len(os.listdir(validate_beagles_dir))
num_trufis = len(os.listdir(validate_yorkshires_dir))
total_validation = num_jakes + num_trufis
print('total validation jakes images:', num_jakes)
print('total validation trufis images:', num_trufis)
print('total validation images:', total_validation)

batch_size = 20
epochs = 200  # Iterations
IMG_HEIGHT = 300
IMG_WIDTH = 300

train_image_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=.25,
    height_shift_range=.25,
    horizontal_flip=True,
    zoom_range=0.1,
)
validate_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=images_dir,
                                                           classes=['beagles', 'yorkshires'],
                                                           class_mode='binary',
                                                           shuffle=True,
                                                           seed=10,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH)
                                                           )

val_data_gen = validate_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=images_dir,
                                                            classes=['jakes', 'trufis'],
                                                            class_mode='binary',
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            seed=10,
                                                            )
sample_training_images, _ = next(train_data_gen)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    # Conv2D(64, 3, padding='same', activation='relu'),
    # MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# TODO: Make a function to predict images once the model is trained

# Create directory if it doesn't exist
if not os.path.isdir('./model'):
    os.mkdir('model')
# Load model from file
# if os.path.isfile('./model/model.h5'):
#     model = load_model('./model/model.h5')

model.summary()
t1 = time()
history = model.fit(
    train_data_gen,
    steps_per_epoch=ceil(total_train / batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=ceil(total_validation / batch_size)
)
t2 = time()
print('Finished network.\nTime taken: ', ((t2 - t1) / 60).__round__(1), ' minutes')

# Save model to file
model.save('./model/model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save figure
time = datetime.now()
plt.savefig("./result/%s_%s_%s" % (time.hour, time.minute, time.second))

# Show figure
plt.show()
