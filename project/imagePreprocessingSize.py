import os
from PIL import Image

directory = './images/dogs/yorkshires/'
min_image_length = 500

i = 0
for fileName in os.listdir(directory):
    if fileName.endswith(".jpg"):
        print(fileName)
        image = Image.open(directory + fileName)
        short_side = min(image.size)
        if short_side > min_image_length:
            resize_factor = min_image_length / short_side
            target_size = tuple(int(resize_factor * x) for x in image.size)
            resized_image = image.resize(target_size, Image.ANTIALIAS)
        else:
            resized_image = image

        resized_image.save(directory + "image" + str(i).zfill(4) + ".jpg")
        # Delete full size images
        os.remove(directory+fileName)
        i += 1
    else:
        # Delete non jpg files
        os.remove(directory+fileName)
