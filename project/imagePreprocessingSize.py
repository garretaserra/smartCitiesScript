import math
import os
from PIL import Image

subdirectories = ['trufis']
main_directory = 'C:\\Users\\Sergi Garreta\\Desktop\yorkshires'
target_size = 500

for breed in subdirectories:
    directory = os.path.join(main_directory, breed) + '/'
    i = 0
    for fileName in os.listdir(directory):
        if fileName.endswith(".jpg"):
            print(fileName)
            im = Image.open(directory + fileName)
            width, height = im.size  # Get dimensions
            min_length = min(width, height)

            if target_size < min(im.size):
                resize_factor = min(im.size) / target_size
                new_size = tuple(int(resize_factor * x) for x in im.size)
                resized_image = im.resize(new_size, Image.ANTIALIAS)

                left = (width - target_size) / 2
                top = (height - target_size) / 2
                right = (width + target_size) / 2
                bottom = (height + target_size) / 2
                resized_image = resized_image.crop((left, top, right, bottom))

            else:
                if min_length == width:
                    left = 0
                    top = math.floor((height - min_length) / 2)
                    right = width
                    bottom = math.floor((height + min_length) / 2)
                else:
                    left = math.floor((width - min_length) / 2)
                    top = 0
                    right = math.floor((width + min_length) / 2)
                    bottom = height
                resized_image = im.crop((left, top, right, bottom))

            resized_image.save(directory + "image" + str(i).zfill(4) + ".jpg")
            # Delete full size images
            os.remove(directory+fileName)
            i += 1
        else:
            # Delete non jpg files
            os.remove(directory+fileName)
