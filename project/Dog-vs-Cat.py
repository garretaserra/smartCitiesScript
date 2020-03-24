from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime
import os
import matplotlib.pyplot as plt
from math import ceil

train_dir = './images/dogs-vs-cats/images'

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

total_train = num_cats_tr + num_dogs_tr

print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)
print('total dog images:', total_train)

batch_size = 220
epochs = 40         # Iterations
IMG_HEIGHT = 150
IMG_WIDTH = 150
holdout = 0.2

image_generator = image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.1,
                    validation_split=holdout
                    )

train_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     subset='training',
                                                     seed=100,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

val_data_gen = image_generator.flow_from_directory(batch_size=batch_size,
                                                   directory=train_dir,
                                                   target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                   class_mode='binary',
                                                   subset='validation',
                                                   seed=100,
                                                   )
sample_training_images, _ = next(train_data_gen)

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=ceil(total_train*(1-holdout) / batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=ceil(total_train*holdout / batch_size)
)

print('Finished network')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

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
plt.savefig("./result/%s_%s_%s" % (time.hour, time.minute, time.second) )

# Show figure
plt.show()