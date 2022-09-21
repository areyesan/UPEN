# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:41:28 2022

@author: Abel
"""
#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,MaxPool2D, UpSampling2D,Activation, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, add
kernel_initializer =  'he_uniform'  # also try 'he_normal' but model not converging... 
from tensorflow.keras import backend as K

from PE_utils import *

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

def residual_block_PE( PE_2, x, filters, kernel_size=(3, 3), padding="same", strides=1):
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


def UPEN_Net(heigh, width, channels, dataset):
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
    e2 = residual_block_PE(prog_expen_conv2, e1, f[1], strides=2)
    e3 = residual_block_PE(prog_expen_conv2, e2, f[2], strides=2)
    e4 = residual_block_PE(prog_expen_conv2, e3, f[3], strides=2)
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