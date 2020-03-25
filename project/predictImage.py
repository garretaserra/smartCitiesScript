import numpy as np
from keras.preprocessing import image
from tensorflow_core.python.keras.saving.save import load_model


loaded_model = load_model('./model/model.h5')
img = image.load_img('./images/dogs/trufis/image0010.jpg', target_size=(200, 200))
img = np.expand_dims(img, axis=0)
res = loaded_model.predict_classes(img)
print(res)

# def predict(picture):
#     # Load model
#     return
#
#
# print(predict(image.load_img('./images/dogs/yorkshires/image000.jpg', target_size=(200, 200))))
