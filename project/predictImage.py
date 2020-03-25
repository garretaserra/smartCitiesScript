import numpy as np
from keras.preprocessing import image
from tensorflow_core.python.keras.saving.save import load_model
import os

test = 'jakes'
# test = 'trufis'

loaded_model = load_model('./model/model.h5')
for file in os.listdir('./images/dogs/' + test):
    img = image.load_img('./images/dogs/' + test + '/' + file, target_size=(200, 200))
    img = np.expand_dims(img, axis=0)
    res = loaded_model.predict_classes(img)
    print(res, file)
