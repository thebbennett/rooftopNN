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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL.Image
from keras.utils import Sequence
import random
import matplotlib.image as mpimg
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Input,Flatten, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.preprocessing import image
from tf_unet import unet, util, image_util
from PIL import Image



os.chdir("/home/thebbennett/rooftopNN/Images/compressed/images/")

class DataSeq(Sequence):
    def __init__(self,batch_size,crop_size,img_name_list):
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.img_name_list = img_name_list
    
    def __len__(self,):
        return len(img_name_list)//self.batch_size
    
    def __getitem__(self,idx):
        
        img_names = np.random.choice(self.img_name_list,self.batch_size,replace=False)
        mask_names = [i.replace('train/','valid/') for i in img_names]        
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
            if random.randint(0,100) < 50:
                img_crop = np.flip(img_crop)
                mask_crop = np.flip(mask_crop)
            if random.randint(0,100) < 50:
                img_crop = np.rot90(img_crop)
                mask_crop = np.rot90(mask_crop)
            x_array_list.append( np.array(img_crop) )
            y_array_list.append( np.array(mask_crop) )
        batch_y = np.array(y_array_list) / 255
        batch_x = np.array(x_array_list) / 255
    

        
        return batch_x, batch_y[:,:,:,:1]
    


path = 'train/*.jpg'   
img_name_list=glob.glob(path)

test = DataSeq(batch_size = 20, crop_size = (256,256),img_name_list = img_name_list)
batch_x, batch_y = test[0]

#create a convultional kernel. output 4 filters , length of 3, 
#have output the same size as the input
#use relu activation 

inputs = Input(shape=(256,256,3))
x = inputs

x = Conv2D(4,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2D(8,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2D(12,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2D(16,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2D(24,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2D(32,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2D(40,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2D(28,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2D(30,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2D(32,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2D(38,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2D(46,kernel_size=3,strides=1,padding='same',activation='elu')(x)


x = Conv2DTranspose(46,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2DTranspose(38,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2DTranspose(32,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2DTranspose(30,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2DTranspose(28,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2DTranspose(40,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2DTranspose(32,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2DTranspose(24,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2DTranspose(16,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2DTranspose(12,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2DTranspose(8,kernel_size=3,strides=1,padding='same',activation='elu')(x)
x = Conv2DTranspose(4,kernel_size=3,strides=2,padding='same',activation='elu')(x)
x = Conv2DTranspose(1,kernel_size=3,strides=1,padding='same',activation='sigmoid')(x)

model = Model(inputs,x)



model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy','mae'])
model.summary()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(batch_x, batch_y[:,:,:,:1],test_size=0.2)
                   
history = model.fit_generator(generator = test,
                   steps_per_epoch = 20,
                   epochs = 5,
                   validation_data=(X_test, y_test), 
                   validation_steps=None)

print(history.history.keys())

import matplotlib.pyplot as plt

# summarize history for loss
f1 = plt.figure(figsize=(10, 10))
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
f1.savefig("semilogy.png")

f2 = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
f2.savefig("modelloss.png")

f3 = plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
f3.savefig("accuracy.png")


img =image.load_img('/home/thebbennett/rooftopNN/Images/compressed/images/train/036610ne239nw_compressed_resized.jpg',
                    target_size = (256,256))
plt.imshow(img)

test_image = image.img_to_array(img)

test_image = np.expand_dims(test_image, axis=0)
predicted_mask_batch = model.predict(test_image)

predicted_mask = predicted_mask_batch[0]
predicted_mask = predicted_mask.squeeze()
f4 = plt.figure(figsize=(30, 15))
ax1 = f4.add_subplot(1,2, 1)
ax1.set_title("Original Image",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
ax2 = f4.add_subplot(1,2, 2)
ax2.set_title("Predicted Mask",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
plt.imshow(predicted_mask, alpha=0.6)
plt.show(block=True)
f4.savefig("predict.png")

