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

kernel_initializer =  'he_uniform'  # also try 'he_normal' but model not converging... 

#%%
seed = 42

np.random.seed = seed
tf.random.set_seed(seed)
#%%
dataset="MRI"

#%%
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
#
#
##dataset="DRIVE_DB"
#dataset = "CHASEDB1"
#Dataset=dataset
#dim = 192
#%%

# -----------------------------Main Path---------------------------------
#main_path = "datasets"  # <------------------------- CHANGE THIS
#
#
#training_images = f"{main_path}/{dataset}/patches_{dim}/images/train/"
#training_labels = f"{main_path}/{dataset}/patches_{dim}/labels/train/"
#
#train_img = next(os.walk(training_images))[2]
#train_lbs = next(os.walk(training_labels))[2]
#
#train_img.sort()
#train_lbs.sort()
#
#X_train = np.concatenate([np.load(training_images + file_id)["arr_0"] for file_id in train_img], axis=0)
#y_train = np.concatenate([np.load(training_labels + file_id)["arr_0"] for file_id in train_lbs], axis=0)
#
#X_train = X_train / 255
#
#if dataset == "CHASEDB1":
#    y_train = y_train.astype("float32")
#else:
#    y_train = y_train / 255
#
#print(np.max(X_train))
#print(np.min(X_train))
#print(np.max(y_train))
#print(np.min(y_train))

X_train.shape

_, h, w, c = X_train.shape

input_shape = (h, w, c)
##%%
#testing_images = f"{main_path}/{dataset}/patches_{dim}/images/test/"
#
#test_img = next(os.walk(testing_images))[2]
#test_img.sort()
#
#X_test = np.concatenate([np.load(testing_images + file_id)["arr_0"] for file_id in test_img], axis=0)
#
#X_test = X_test / 255

X_test.shape


#testing_labels = f"{main_path}/{dataset}/patches_{dim}/labels/test/"

#test_lbl = next(os.walk(testing_labels))[2]
#test_lbl.sort()

#y_test = np.concatenate([np.load(testing_labels + file_id)["arr_0"] for file_id in test_lbl], axis=0)

#if dataset == "CHASEDB1":
#    y_test = y_test.astype("float32")
#else:
#    y_test = y_test / 255
#
#print(np.max(y_test))
#print(np.min(y_test))



#%%
heigh=X_train.shape[1]
width=X_train.shape[2]
channels=X_train.shape[3]
#%%
#%%

test_img_number = random.randint(0, X_test.shape[0]-1)  #Test with 119

#plt.imshow(X_test_images_1[test_img_number], cv2.COLOR_BGR2RGB)

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
num_epochs=25
#%%
#from keras import backend as K
expand_type=1
num_terms = 2
#funcType = 'ln(1+x)'
#funcType = 'sin(x)'
funcType = 'arctan(x)'

#%%
def prog_expen(x):     
    #new_x = np.zeros((x.shape[0], x.shape[1],x.shape[2],x.shape[3]*num_terms))
    print("Progressive Expansion of the input")
    print(x.shape)
    num=x.shape[0]
    dim_x=x.shape[1]
    dim_y=x.shape[2]
    bands=x.shape[3]
    new_x = np.zeros((x.shape[0]*x.shape[1]*x.shape[2],x.shape[3]*num_terms))
    new_x = new_x.astype('float32') 
    x=np.reshape(x,(num*dim_x*dim_y,bands))

    nn = 0
    for i in range(x.shape[1]):
        col_d = x[:,i].ravel()
        new_x[:,nn] = col_d
        if num_terms > 0:
            if expand_type == 1:
                for od in range(1,num_terms):
                    if funcType  == 'linear':
                       new_x[:,nn+od] = col_d
                    elif funcType == 'sin(x)': # sin(x)
                       new_x[:,nn+od] = new_x[:,nn+od-1] + (((-1)**od)/np.math.factorial(2*od+1))*(col_d**(2*od+1))  
                    elif funcType == 'ln(1+x)':  # log(1+x)
                       new_x[:,nn+od] = new_x[:,nn+od-1] + ((-1)**(od+1+1))*(col_d**(od+1))/(od+1)  # require |x|<1. 
                    elif funcType == 'arctan(x)':  # 
                       new_x[:,nn+od] = new_x[:,nn+od-1] + ((-1)**(od))*(col_d**(2*od+1)/(2*od+1))
                nn = nn + num_terms
            else:
                for od in range(1,num_terms):
                    if funcType  == 'linear':
                       new_x[:,nn+od] = col_d
                    elif funcType == 'sin(x)': # sin(x)
                       new_x[:,nn+od] = (((-1)**od)/np.math.factorial(2*od+1))*(col_d**(2*od+1))  
                    elif funcType == 'ln(1+x)':  # log(1+x)
                       new_x[:,nn+od] = ((-1)**(od+1+1))*(col_d**(od+1))/(od+1)  # require |x|<1. 
                    elif funcType == 'arctan(x)':  # 
                       new_x[:,nn+od] = ((-1)**(od))*(col_d**(2*od+1)/(2*od+1))
                nn = nn + num_terms                
    
    x1 = new_x[:,0::3]
    x2 = new_x[:,1::3]
    x3 = new_x[:,2::3]
    cat_x = np.hstack((x1,x2,x3))
    cat_x=np.reshape(cat_x,(num,dim_x,dim_y,bands*num_terms))
    return cat_x

#%%
def prog_expen_conv1(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
    else:
        p1 = x
        p2 = -((x**2)/2)     
    new_x = K.concatenate([p1, p2], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv2(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
    else:
        p1 = x
        p2 = -((x**2)/2)
    new_x = K.concatenate([p1, p2], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv3(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        #p3 = x - ((x**2)/2)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
    new_x = K.concatenate([p1, p2, p3], axis=len(x.shape)-1)
#    print(new_x)
    return new_x




def prog_expen_conv4(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
    new_x = K.concatenate([p1, p2, p3, p4], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv5(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
    new_x = K.concatenate([p1, p2, p3, p4, p5], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv6(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv7(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv8(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
        p8 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) 
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
        p8 = - ((x**8)/8) 
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7, p8], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv9(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
        p8 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) 
        p9 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) + ((x**9)/9)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
        p8 = - ((x**8)/8) 
        p9 = ((x**9)/9)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9], axis=len(x.shape)-1)
#    print(new_x)
    return new_x

def prog_expen_conv10(x):
    if expand_type == 1:         
        p1 = x
        p2 = x - ((x**2)/2)
        p3 = x - ((x**2)/2) + ((x**3)/3)
        p4 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4)  
        p5 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) 
        p6 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6)
        p7 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7)
        p8 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) 
        p9 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) + ((x**9)/9)
        p10 = x - ((x**2)/2) + ((x**3)/3) - ((x**4)/4) + ((x**5)/5) - ((x**6)/6) + ((x**7)/7) - ((x**8)/8) + ((x**9)/9)- ((x**10)/10)
    else:
        p1 = x
        p2 = -((x**2)/2)
        p3 = (x**3)/3
        p4 = - ((x**4)/4)       
        p5 = ((x**5)/5) 
        p6 = - ((x**6)/6)
        p7 = ((x**7)/7)
        p8 = - ((x**8)/8) 
        p9 = ((x**9)/9)
        p10 = - ((x**10)/10)
    new_x = K.concatenate([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10], axis=len(x.shape)-1)
#    print(new_x)
    return new_x









#%%
num_filter1=256
num_filter2=256
#conv_shape1=64
#conv_shape2=32
#%%
dim=6

kernel_size1 = 3
kernel_size2 = 3
kernel_size3 = 3
#kernel_size4 = 1
#kernel_size5 = 1 

conv_shape1 = int(num_terms*((dim - kernel_size1) + 1)/2)
conv_shape2 = int(2*((conv_shape1 - kernel_size2 + 1)/2 ))
conv_shape3 = int(3*((conv_shape1 + conv_shape2 - kernel_size3 + 1)/2))+2
#%%
X_train_images = prog_expen(X_train)
X_test_images  = prog_expen(X_test)
#%%
X_train_images = X_train_images.astype(np.int32)
X_test_images = X_test_images.astype(np.int32)

#X_train_images = X_train
#X_test_images  = X_test
#%%
#def conv_block(input, num_filters):
#    x = Conv2D(num_filters, 3, padding="same")(input)
#    #x = BatchNormalization()(x)   #Not in the original network. 
#    x = Activation("relu")(x)
#    x = Conv2D(num_filters, 3, padding="same")(x)
#    #x = BatchNormalization()(x)  #Not in the original network
#    x = Activation("relu")(x)
#
#    return x
#
#def conv_block_PE(input, num_filters, PE_1, PE_2):
#    x = Conv2D(num_filters, 3, padding="same")(input)
#    skip=Conv2D(num_filters, 1, padding="same")(x)
#    #x = BatchNormalization()(x)   #Not in the original network. 
#    x = Activation("relu")(x)
#    PEN1   = Lambda(PE_1)(x)
#    PEN1 = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN1)
#    PEN2   = Lambda(PE_2)(PEN1)
#    PEN2 = Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN2)
#    merged2=add([PEN2,skip])
#    merged2=BatchNormalization(axis=3)(merged2)
#    x = Activation('tanh')(merged2)
#    #x = Conv2D(num_filters, 3, padding="same")(x)
#    #x = BatchNormalization()(x)  #Not in the original network
#    #x = Activation("relu")(x)
#
#    return x





def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

def encoder_block_PE(input, num_filters, PE_1, PE_2):
    x = conv_block_PE(input, num_filters, PE_1, PE_2)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x





def decoder_block_PE(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x



#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model


#

def build_unet_PEN(input_shape, filtersStart, PE_1, PE_2, vert_depth):
    inputs = Input(input_shape)
    s_elements=[]
    p_elements=[inputs]
    for i in range(0,vert_depth):
        s, p = encoder_block_PE(p_elements[i], filtersStart*(math.pow(2,i)),PE_1, PE_2)
        s   = Lambda(PE_1)(s)
        p_elements.append(p)
        s_elements.append(s)
        
    b1 = conv_block(p_elements[-1], filtersStart*(math.pow(2,vert_depth))) #Bridge
    b1 = Dropout(0.3)(b1)
    d_elements=[b1]
    
    for j in range(1,(vert_depth+1)):
        concat=s_elements[len(s_elements)-j]
        d = decoder_block(d_elements[j-1], concat, filtersStart*(math.pow(2,vert_depth-j)))

        d_elements.append(d)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d_elements[-1])  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="UPEN_net_ultimate_v18")
    model.summary()
    return model

#%%
################################################################
def UPENnet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, filter_start):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(filter_start, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    PEN1   = Lambda(prog_expen_conv3, output_shape = (conv_shape1, num_filter1), name = 'PEN1_1')(c1)
    PEN1 = Conv2D(filter_start, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN1)
    PEN2   = Lambda(prog_expen_conv4, output_shape = (conv_shape3, num_filter2), name = 'PEN2_1')(PEN1)
    PEN2 = Conv2D(filter_start, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN2)
    merged2=add([PEN2,PEN1])
    merged2=BatchNormalization(axis=3)(merged2)
    c1 = Activation('tanh')(merged2)
    p1=MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(filter_start*2, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    PEN1   = Lambda(prog_expen_conv3, output_shape = (conv_shape1, num_filter1), name = 'PEN1_2')(c2)
    PEN1 = Conv2D(filter_start*2, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN1)
    PEN2   = Lambda(prog_expen_conv4, output_shape = (conv_shape3, num_filter2), name = 'PEN2_2')(PEN1)
    PEN2 = Conv2D(filter_start*2, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN2)
    merged2=add([PEN2,PEN1])
    merged2=BatchNormalization(axis=3)(merged2)
    c2 = Activation('tanh')(merged2)
    p2=MaxPooling2D((2,2))(c2)
     
    c3 = Conv2D(filter_start*4, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    PEN1   = Lambda(prog_expen_conv3, output_shape = (conv_shape1, num_filter1), name = 'PEN1_3')(c3)
    PEN1 = Conv2D(filter_start*4, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN1)
    PEN2   = Lambda(prog_expen_conv4, output_shape = (conv_shape3, num_filter2), name = 'PEN2_3')(PEN1)
    PEN2 = Conv2D(filter_start*4, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN2)
    merged2=add([PEN2,PEN1])
    merged2=BatchNormalization(axis=3)(merged2)
    c3 = Activation('tanh')(merged2)
    p3=MaxPooling2D((2,2))(c3)
     
    c4 = Conv2D(filter_start*8, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    PEN1   = Lambda(prog_expen_conv3, output_shape = (conv_shape1, num_filter1), name = 'PEN1_4')(c4)
    PEN1 = Conv2D(filter_start*8, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN1)
    PEN2   = Lambda(prog_expen_conv4, output_shape = (conv_shape3, num_filter2), name = 'PEN2_4')(PEN1)
    PEN2 = Conv2D(filter_start*8, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN2)
    merged2=add([PEN2,PEN1])
    merged2=BatchNormalization(axis=3)(merged2)
    c4 = Activation('tanh')(merged2)
    p4=MaxPooling2D((2,2))(c4)     
    
    c5 = Conv2D(filter_start*16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(filter_start*16, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(filter_start*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    #p6=MaxPooling2D((2,1))(u6)
    c6 = Conv2D(filter_start*8, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(filter_start*8, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv2DTranspose(filter_start*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    #p7=MaxPooling2D((2,1))(u7)
    c7 = Conv2D(filter_start*4, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(filter_start*4, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv2DTranspose(filter_start*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2], axis=3)
    #p8=MaxPooling2D((2,1))(u8)
    c8 = Conv2D(filter_start*2, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(filter_start*2, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv2DTranspose(filter_start, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    #p9=MaxPooling2D((2,1))(u9)
    c9 = Conv2D(filter_start, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(filter_start, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs], name="UPEN_net_v7")
    model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=[MeanIoU(num_classes=2)])
    model.summary()
    
    return model
#%%
def bn_act(x, act=True):
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def conv_block_PE(PE_block, x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    PEN1   = Lambda(PE_block)(conv)
    PEN1 = Conv2D(filters, (1,1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(PEN1)
    return PEN1


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1 ):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=True)
    
    output = tf.keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=True)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output

def residual_block_PE( PE_1, PE_2, x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block_PE(PE_2, x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=True)
    
    output = tf.keras.layers.Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    c = tf.keras.layers.Concatenate()([u, xskip])
    return c


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
smooth = 1.
#tf.keras.layers.Flatten()(input)
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


#X_train_images = X_train_images_1
#X_test_images  = X_test_images_1
#%%
channels=X_train_images.shape[3]
#filter_s=32
filter_s=[16]
input_shape=(heigh,width,channels)
vertical_depth=[5]
PE_1_list=[prog_expen_conv4]
PE_2_list=[prog_expen_conv5]
max_val_iou=0
#%%
#model = ResUNet()
#model.summary()
#model=build_unet_PEN(input_shape, filter_s, PE_1_list, PE_2_list,4)
#%%


model=model = ResUNet()
#                    model=build_unet_PEN(input_shape, filter_start, PE_1, PE_2,vd)
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
model.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coef])
#%%
start = time.time()
history = model.fit(X_train_images, y_train,                # Train the model using the training set...
                    batch_size=batch_size, epochs=num_epochs,
                    verbose=1, 
                    callbacks=callbacks_list,
                    validation_split=0.1) # ...holding out 10% of the data for validation

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

#%%
#%%
test_pred_batch = (model.predict(X_test_images)[:,:,:,0] > 0.5).astype(np.uint8)
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch, y_test)
print("Mean IoU =", IOU_keras.result().numpy())
