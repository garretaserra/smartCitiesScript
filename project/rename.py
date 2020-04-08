import os
from PIL import Image

subdirectories = ['jakes', 'trufis']
main_directory = 'C:\\Users\\rocio\\Desktop\\BIG DATA\\SmarCitiesBD\\project\\images\\dogs2'

for breed in subdirectories:
    directory = os.path.join(main_directory, breed) + '/'
    i = 0
    for fileName in os.listdir(directory):
        if fileName.endswith(".jpg"):
            im = Image.open(directory + fileName)
            im.save(directory + breed + "_image" + str(i).zfill(4) + ".jpg")
            os.remove(directory + fileName)
            i += 1