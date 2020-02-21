'''
Global Feature Descriptors
These are the feature descriptors that quantifies an image globally. These don’t have the concept of interest points and thus, takes in the entire image for processing.
* Color - Color Channel Statistics (Mean, Standard Deviation) and Color Histogram
* Shape - Hu Moments, Zernike Moments
* Texture - Haralick Texture, Local Binary Patterns (LBP)
* Others - Histogram of Oriented Gradients (HOG), Threshold Adjancency Statistics (TAS)

Local Feature Descriptors
These are the feature descriptors that quantifies local regions of an image. Interest points are determined in the entire image and image patches/regions surrounding those interest points are considered for analysis. Some of the commonly used local feature descriptors are
* SIFT (Scale Invariant Feature Transform)
* SURF (Speeded Up Robust Features)
* ORB (Oriented Fast and Rotated BRIEF)
* BRIEF (Binary Robust Independed Elementary Features)

Combining Features
There are two popular ways to combine these feature vectors.

For global feature vectors, we just concatenate each feature vector to form a single global feature vector. 

For local feature vectors as well as combination of global and local feature vectors, we need something called as Bag of Visual Words (BOVW). Normally, it uses Vocabulory builder, K-Means clustering, Linear SVM, and Td-Idf vectorization.
'''

import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

img = io.imread(r'..\datasets\image_0404.jpg')
plt.imshow(img)

dataset = pd.DataFrame(columns=["Name"])
dataset.at[0,"Name"] = 'image_0404.jpg'


features = list()
features.append('image_0404.jpg')

#
# Global Feature Descriptors
#

# Color Channel Statistics
ch_average = img.mean(axis=0).mean(axis=0)
ch_std = img.std(axis=0).std(axis=0)

features.extend(ch_average)
features.extend(ch_std)


# Color Histogram
from skimage.exposure import histogram

img_histogram_R, _ = histogram(img[:,:,0], normalize=True)
img_histogram_G, _ = histogram(img[:,:,0], normalize=True)
img_histogram_B, _ = histogram(img[:,:,0], normalize=True)

features.extend(img_histogram_R)
features.extend(img_histogram_G)
features.extend(img_histogram_B)


# Shape - Hu Moments
from skimage.measure import moments_hu
from skimage.measure import moments
from skimage.measure import moments_central
from skimage.measure import moments_normalized
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

grayscale = rgb2gray(img)
grayscale = img_as_ubyte(grayscale)

mu = moments_central(grayscale)
nu = moments_normalized(mu)
hu = moments_hu(nu)

features.extend(hu)


# Texture - Haralick Texture
# conda install -c conda-forge mahotas
from mahotas.features import haralick
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

grayscale = rgb2gray(img)
grayscale = img_as_ubyte(grayscale)

haralick_features = haralick(grayscale)
haralick_features = haralick_features.flatten()

features.extend(haralick_features)

# Texture - Local Binary Patterns (LBP)
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

grayscale = rgb2gray(img)
grayscale = img_as_ubyte(grayscale)

# settings for LBP
radius = 3
n_points = 8 * radius

lbp = local_binary_pattern(grayscale, n_points, radius, 'uniform')
lbp = lbp.flatten()

features.extend(lbp)

# Others - Histogram of Oriented Gradients (HOG)
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

grayscale = rgb2gray(img)
grayscale = img_as_ubyte(grayscale)

hog_features = hog(grayscale, pixels_per_cell=(32, 32), cells_per_block=(1, 1), block_norm='L2-Hys', feature_vector=True)

features.extend(hog_features)