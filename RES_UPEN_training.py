# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 16:46:17 2022

@author: Abel
"""
###############UPEN################
import os
import datetime
import numpy as np
import tensorflow as tf
#from efficientnet.keras import EfficientNetB7
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, LeakyReLU, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,MaxPool2D, UpSampling2D,Activation, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import MeanIoU
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
#import cv2
import random
import segmentation_models as sm
import pandas as pd
import tensorflow as tf
import math
import gc
from numpy import load

from pe_functions import prog_expen, prog_expen_conv1, prog_expen_conv2, prog_expen_conv3, prog_expen_conv4
from pe_functions import encoder_block, encoder_block_PE, decoder_block, decoder_block_PE, bn_act, conv_block
from pe_functions import conv_block_PE, stem, residual_block, residual_block_PE, upsample_concat_block, dice_coef, dice_coef_loss
                            
kernel_initializer =  'he_uniform'  # also try 'he_normal' but model not converging... 

#%%
seed = 42
np.random.seed = seed
tf.random.set_seed(seed)
#%%
dataset="MRI"

#%%
X_train=load('MRI_images_128_gabor_tr.npy')
#X_train=load('MRI_Images_128_gabor.npy')
y_train=load('MRI_Masks_128.npy')
X_test=load('MRI_images_128_gabor_ts.npy')
#X_test=load('MRI_Images_128_testing_gabor.npy')
y_test=load('MRI_Masks_128_testing.npy')
#%%%% #########################################CHASE DB##########################
#X_train=load('x_train_CHASEDB1_Gabor_tr.npy')
#y_train=load('x_test_CHASEDB1_Gabor_ts.npy')
#X_test=load('MRI_Images_128_gabor_ts.npy')
#y_test=load('MRI_Masks_128_testing.npy')
##%%%% #########################################ISIC2018##########################
#X_train=load('MRI_Images_128_gabor_tr.npy')
#y_train=load('MRI_Masks_128.npy')
#X_test=load('MRI_Images_128_gabor_ts.npy')
#y_test=load('MRI_Masks_128_testing.npy')


##%%
X_train = X_train.astype(np.int32)
X_test = X_test.astype(np.int32)
#%%
X_train.shape
_, h, w, c = X_train.shape
input_shape = (h, w, c)
#%%
X_test.shape
#%%
heigh=X_train.shape[1]
width=X_train.shape[2]
channels=X_train.shape[3]
#%%
test_img_number = random.randint(0, X_test.shape[0]-1)
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title('Image Input')
#frame = cv2.cvtColor(X_test[test_img_number,:,:], cv2.COLOR_BGRA2RGB)
plt.imshow(X_test[test_img_number,:,:,32])
#plt.imshow(frame)
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(y_test[test_img_number,:,:])
plt.show()

#%%
batch_size=16
num_epochs=200
#%%
expand_type=1
num_terms = 2
#funcType = 'ln(1+x)'
#funcType = 'sin(x)'
funcType = 'arctan(x)'

#%%
#dim=6
#
#kernel_size1 = 3
#kernel_size2 = 3
#kernel_size3 = 3
##kernel_size4 = 1  
##kernel_size5 = 1 
#
#conv_shape1 = int(num_terms*((dim - kernel_size1) + 1)/2)
#conv_shape2 = int(2*((conv_shape1 - kernel_size2 + 1)/2 ))
#conv_shape3 = int(3*((conv_shape1 + conv_shape2 - kernel_size3 + 1)/2))+2
#%%
X_train_images = prog_expen(X_train)
X_test_images  = prog_expen(X_test)
#%%
X_train_images = X_train_images.astype(np.int32)
X_test_images = X_test_images.astype(np.int32)


#%%
channels=X_train_images.shape[3]
#filter_s=32
#filter_s=[16]
#input_shape=(heigh,width,channels)
#vertical_depth=[5]
#PE_1_list=[prog_expen_conv4]
#PE_2_list=[prog_expen_conv5]
#max_val_iou=0

#%%
def ResUNet():
    #f = [16, 32, 64, 128, 256]
    #f = [64, 128, 256, 512]
    #f=[64,128,256,512]
    f = [32, 64, 128, 256]

    inputs = tf.keras.layers.Input((heigh, width, channels))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
#    e2 = residual_block(e1, f[1], strides=2)
#    e3 = residual_block(e2, f[2], strides=2)
#    e4 = residual_block(e3, f[3], strides=2)
    e2 = residual_block_PE(prog_expen_conv2, prog_expen_conv2, e1, f[1], strides=2)
    e3 = residual_block_PE(prog_expen_conv2, prog_expen_conv2, e2, f[2], strides=2)
    e4 = residual_block_PE(prog_expen_conv2, prog_expen_conv2, e3, f[3], strides=2)
    #e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
#    b0 = conv_block(e5, f[4], strides=1)
#    b1 = conv_block(b0, f[4], strides=1)

    b0 = conv_block(e4, f[3], strides=1)
    b1 = conv_block(b0, f[3], strides=1)

    
    ## Decoder
    u1 = upsample_concat_block(b1, e3)
    d1 = residual_block(u1, f[3])
    
    u2 = upsample_concat_block(d1, e2)
    d2 = residual_block(u2, f[2])
    
    u3 = upsample_concat_block(d2, e1)
    d3 = residual_block(u3, f[1])
    
#    u4 = upsample_concat_block(d3, e1)
#    d4 = residual_block(u4, f[1])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d3)
    model = tf.keras.models.Model(inputs, outputs, name=("ResUnet_"+dataset+"_dataset"))
    return model

#%%
model=model = ResUNet()
model.summary()
#from keras.utils.vis_utils  import plot_model
#plot_model(model, to_file='testModel_ResUnet_PE_3.png', show_shapes=True)
#%%

filepath = ("models/"+model.name+"_"+dataset+"_"+".hdf5")

checkpoint = ModelCheckpoint(filepath, monitor='val_mean_io_u', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=50, verbose=1, cooldown=10, min_lr=1e-5)
lr_shceduler = LearningRateScheduler(lambda _, lr: lr * np.exp(-0.01), verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard("logs/1_r2a", update_freq=1)

callbacks_list = [reduce_lr, lr_shceduler, tensorboard, checkpoint]
#%%
adam = tf.keras.optimizers.Adam()
optm = SGD(learning_rate=0.001, decay=1e-3, momentum=0.9, nesterov=True)
#model.compile(optimizer=adam, loss=dice_coef_loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coef])

model.compile(optimizer=Adam(learning_rate = 1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coef])
#%%
start = time.time()
history = model.fit(X_train_images, y_train,                # Train the model using the training set...
                    batch_size=batch_size, epochs=num_epochs,
                    verbose=1, 
                    callbacks=callbacks_list,
                    validation_data=(X_test_images, y_test)
                    #validation_split=0.1
                    ) # ...holding out 10% of the data for validation

end = time.time()
print("The total training Time:::: %f" %(end - start))
#versionCode=(''+str(filter_start)+str(vd)+str(PE_1)[1:27]+str(PE_2)[1:27])
#%%
modelFileName=("models/"+model.name+"_"+dataset+"_"+".hdf5")
model.save(modelFileName)
#%%
unet_history_df = pd.DataFrame(history.history) 
with open('Results/'+model.name+dataset+'_'+'_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
                    
score = model.evaluate(X_test_images, y_test, verbose=2)  # Evaluate the trained model on the test set!
print("Overall Accuracy - %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
predictions = model.predict(X_test_images,verbose=1)

predictions_norm = predictions > 0.5
n_classes = 2
IoU_values = []
for img in range(0, X_test.shape[0]):
    temp_img = X_test_images[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)
    
                    
df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)    


