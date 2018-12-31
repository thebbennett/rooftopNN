#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 20:33:15 2018

@author: thebbennett
"""
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
import os
import glob

### Predicting on all images

folder_path = '/home/thebbennett/rooftopNN/data/predict/*.jpg'
# path to model
model_path = '/home/thebbennett/rooftopNN/model.h5'
# dimensions of images
img_width, img_height = 256,256

# load the trained model
model = load_model(model_path)
model.compile(loss='binary_crossentropy',
              optimizer= 'Adam',
              metrics=['accuracy', 'mae'])

# load all images into a list

img_name_list=glob.glob(folder_path)

images = []
for img in img_name_list:
    img = image.load_img(img, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
classes = model.predict(images, batch_size=10)

sums = []
for image in classes:
    b = image > 0.5
    sums.append(len(image[b]))
    
pixels = sum(sums)

original_dim = 2640 

original_feet = 0.5 / original_dim

new_dim = 0.5 * img_width / original_dim

pixels * new_dim


