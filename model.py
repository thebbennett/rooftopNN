"""
Rooftop Image Classification from Satellite Imagery
Denver,CO

Brittany Bennett
December 2018 
"""

# Load the necessary packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from keras.utils import Sequence
import random
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Input,Flatten, Activation, MaxPooling2D, UpSampling2D
from keras.layers.core import Dropout
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.layers import concatenate 

os.chdir("/home/thebbennett/rooftopNN/data/")

np.random.seed(1234)
class DataSeq(Sequence):
    def __init__(self,batch_size,crop_size,img_name_list):
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.img_name_list = img_name_list
    
    def __len__(self,):
        return len(img_name_list)//self.batch_size
    def __getitem__(self,idx):
            img_names = np.random.choice(self.img_name_list,self.batch_size,replace=False)
            mask_names = [i.replace('images/','labels/') for i in img_names]        
            imgs = [PIL.Image.open(img_name) for img_name in img_names]
            masks = [PIL.Image.open(mask_name) for mask_name in mask_names]
            center = (random.randint(0,(1000-256)),random.randint(0,(1000-256)))
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
                
                params = np.random.uniform(10,80)
                img = img.rotate(params,center=center)
                mask = mask.rotate(params,center=center)

                crop_coordinate_tuple = (x,y,x+self.crop_size[0],y+self.crop_size[1])
                img_crop = img.crop(crop_coordinate_tuple)
                mask_crop = mask.crop(crop_coordinate_tuple)
                img_crop = np.array(img_crop)
                mask_crop = np.array(mask_crop)
                if random.randint(0,100) < 50:
                    img_crop = np.flip(img_crop)
                    mask_crop = np.flip(mask_crop)
                if random.randint(0,100) < 50:
                    img_crop = np.rot90(img_crop)
                    mask_crop = np.rot90(mask_crop)

                x_array_list.append(img_crop) 
                y_array_list.append(mask_crop) 
            batch_y = np.array(y_array_list) / 255
            batch_x = np.array(x_array_list) / 255



            return batch_x, batch_y[:,:,:,:1]


path = 'train/images/*.jpg'   
img_name_list=glob.glob(path)

batch_size = 4
num_samples = len(img_name_list)

train = DataSeq(batch_size = batch_size, crop_size = (256,256),img_name_list = img_name_list)

path = 'test/images/*.jpg'   
img_name_list=glob.glob(path)
test = DataSeq(batch_size = 5, crop_size = (256,256),img_name_list = img_name_list)

def get_model(optimizer, loss_metric, metrics, lr=1e-3):
    inputs = Input((256, 256, 3))

    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(drop1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(drop2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.3)(pool3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.3)(pool4)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=optimizer(lr=lr), loss=loss_metric, metrics=metrics)
    return model

model = get_model(optimizer=Adam, loss_metric= 'binary_crossentropy', metrics=[ 'accuracy','mae'], lr=1e-3)

                   
history = model.fit_generator(generator = train,
                   steps_per_epoch = 5,
                   epochs = 200,
                   #callbacks=callbacks, # Early stopping
                   validation_data = test)           
                   #validation_steps= 10)
                   #verbose = 0)
os.chdir("/home/thebbennett/rooftopNN/")
model.save("model.h5")
acc = (model.history.history['acc'][-1])
val = (model.history.history['val_acc'][-1])
loss = (model.history.history['loss'][-1])


print(history.history.keys())

import matplotlib.pyplot as plt

os.chdir("/home/thebbennett/rooftopNN/plots")


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
f2.savefig("modelloss.png")

f3 = plt.figure(figsize=(10, 10))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
f3.savefig("accuracy.png")


img =image.load_img('/home/thebbennett/rooftopNN/data/train/images/05681004se.jpg',
                    target_size = (256,256))
test_image = image.img_to_array(img)

test_image = np.expand_dims(test_image, axis=0)
predicted_mask_batch = model.predict(test_image)

predicted_mask = predicted_mask_batch[0]
predicted_mask1 = predicted_mask.squeeze()
f4 = plt.figure(figsize=(30, 15))
ax1 = f4.add_subplot(1,3, 1)
ax1.set_title("Original Image",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
ax2 = f4.add_subplot(1,3, 2)
ax2.set_title("Predicted Mask",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
plt.imshow(predicted_mask1 > 0.5, alpha = 0.6)
ax3 = f4.add_subplot(1,3,3)
ax3.set_title("Probabilities",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(predicted_mask1)
f4.savefig("predict.png")


img =image.load_img('/home/thebbennett/rooftopNN/data/train/images/05681403se.jpg',
                    target_size = (256,256))
test_image = image.img_to_array(img)

test_image = np.expand_dims(test_image, axis=0)
predicted_mask_batch = model.predict(test_image)

predicted_mask = predicted_mask_batch[0]
predicted_mask2 = predicted_mask.squeeze()
f5 = plt.figure(figsize=(30, 15))
ax1 = f5.add_subplot(1,3, 1)
ax1.set_title("Original Image",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
ax2 = f5.add_subplot(1,3, 2)
ax2.set_title("Predicted Mask",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(img)
plt.imshow(predicted_mask2 > 0.5, alpha = 0.6)
ax3 = f5.add_subplot(1,3,3)
ax3.set_title("Probabilities",fontdict={'fontsize': 60, 'fontweight': 'medium'})
plt.imshow(predicted_mask2)
f5.savefig("predict2.png")

img = image.load_img(pred_dir, target_size=(28, 28))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

