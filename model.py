# -*- coding: utf-8 -*-
"""
Rooftop Image Classification from Satellite Imagery
Denver,CO

Brittany Bennett
December 2018 
"""

# Load the necessary pockages
import os
import glob
import numpy as np
from tensorflow.python.keras.models import Model
from keras.preprocessing import image


os.chdir("/home/thebbennett/rooftopNN/Images/compressed/images/")
path_to_valid = "/home/thebbennett/rooftopNN/Images/compressed/images/valid"
from keras.utils import Sequence
import PIL.Image

class DataSeq(Sequence):
    def __init__(self,batch_size,crop_size,img_name_list):
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.img_name_list = img_name_list
    
    def __len__(self,):
        return len(img_name_list)//self.batch_size
    
    def __getitem__(self,idx):
        
        img_names = np.random.choice(self.img_name_list,self.batch_size,replace=False)
        mask_names = [i.replace('/train/','/valid/') for i in img_names]        
        imgs = [PIL.Image.open(img_name) for img_name in img_names]
        masks = [PIL.Image.open(mask_name) for mask_name in mask_names]
        
        limit_left = 0
        img = imgs[0]
        limit_right = img.size[0] - self.crop_size[0]
        limit_bottom = 0
        limit_top = img.size[1] - self.crop_size[1]
        
        x_array_list = []
        y_array_list = []
        for img,mask in zip(imgs,masks):
            x = np.random.uniform(limit_left,limit_right)
            y = np.random.uniform(limit_bottom,limit_top)
            crop_coordinate_tuple = (x,y,x+self.crop_size[0],y+self.crop_size[1])
            img_crop = img.crop(crop_coordinate_tuple)
            mask_crop = mask.crop(crop_coordinate_tuple)
            x_array_list.append( np.array(img_crop) )
            y_array_list.append( np.array(mask_crop) )

        batch_y = np.array(y_array_list) / 255
        batch_x = np.array(x_array_list) / 255

        
        return batch_x, batch_y[:,:,:,:1]

import matplotlib.image as mpimg

path = 'train/rooftop/*.jpg'   
img_name_list=glob.glob(path)

test = DataSeq(batch_size = 3, crop_size = (256,256),img_name_list = img_name_list)
batch_x, batch_y = test[0]


from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input

inputs = Input(shape=(256,256,3))
x = inputs



#create a convultional kernel. output 4 filters , length of 3, 
#have output the same size as the input
#use relu activation 
x = Conv2D(4,kernel_size=3,strides=2,padding='same',activation='relu')(x)
x = Conv2D(8,kernel_size=3,strides=1,padding='same',activation='relu')(x)
x = Conv2D(12,kernel_size=3,strides=2,padding='same',activation='relu')(x)
x = Conv2D(16,kernel_size=3,strides=1,padding='same',activation='relu')(x)
x = Conv2D(24,kernel_size=3,strides=2,padding='same',activation='relu')(x)
x = Conv2D(32,kernel_size=3,strides=1,padding='same',activation='relu')(x)

x = Conv2DTranspose(32,kernel_size=3,strides=1,padding='same',activation='relu')(x)
x = Conv2DTranspose(24,kernel_size=3,strides=2,padding='same',activation='relu')(x)
x = Conv2DTranspose(16,kernel_size=3,strides=1,padding='same',activation='relu')(x)
x = Conv2DTranspose(12,kernel_size=3,strides=2,padding='same',activation='relu')(x)
x = Conv2DTranspose(8,kernel_size=3,strides=1,padding='same',activation='relu')(x)
x = Conv2DTranspose(4,kernel_size=3,strides=2,padding='same',activation='relu')(x)
x = Conv2DTranspose(1,kernel_size=3,strides=1,padding='same',activation='relu')(x)


model = Model(inputs,x)

model.compile(optimizer='adam',loss='binary_crossentropy')
model.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(batch_x, batch_y[:,:,:,:1], test_size=0.2)

history = model.fit_generator(generator = test,
                   steps_per_epoch = 100,
                   epochs = 50,
                   validation_data=(X_test, y_test), 
                   validation_steps=None)

print(history.history.keys())

import matplotlib.pyplot as plt

# summarize history for loss
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("semilogy.png")


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("modelloss.png")


img =image.load_img('/home/thebbennett/rooftopNN/Images/compressed/images/train/rooftop/036610ne239nw_compressed_resized.jpg',
                    target_size = (256,256))
plt.imshow(img)

test_image = image.img_to_array(img)

test_image = np.expand_dims(test_image, axis=0)
predicted_mask_batch = model.predict(test_image)

predicted_mask = predicted_mask_batch[0]
predicted_mask = predicted_mask.squeeze()



f = plt.figure(figsize=(30, 15))
ax1 = f.add_subplot(1,2, 1)
ax1.set_title("Original Image",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
ax2 = f.add_subplot(1,2, 2)
ax2.set_title("Predicted Mask",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
plt.imshow(predicted_mask, alpha=0.6)
plt.show(block=True)
f.savefig("predict.png")

