
import numpy as np
from keras.preprocessing import image
from tensorflow_core.python.keras.saving.save import load_model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# test = 'jakes'
# test = 'trufis'
# test = 'beagles'
test = 'yorkshires'
loaded_model = load_model('./model/model.h5')
i = 0
total = 0
num = 0
if test == 'trufis' or test == 'yorkshires':
    num = 1

for file in os.listdir('./images/dogs/' + test):
    img = image.load_img('./images/dogs/' + test + '/' + file, target_size=loaded_model.get_layer(index=0).input_shape[1:3])
    img = np.expand_dims(img, axis=0)
    res = loaded_model.predict_classes(img)

    if res[0] == num:
        i = i + 1
    total = total + 1
    print(res, file)
print(i/total*100, '%')
