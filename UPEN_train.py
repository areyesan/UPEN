# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:36:17 2022

@author: Abel
"""
import os
import datetime
import time
import math
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, LeakyReLU, MaxPooling2D



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,MaxPool2D, UpSampling2D,Activation, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import MeanIoU
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

#import cv2
import segmentation_models as sm

from PE_utils import *
from UPEN_net_model import *


kernel_initializer =  'he_uniform'  # also try 'he_normal' but model not converging... 

#%%
seed = 42

np.random.seed = seed
tf.random.set_seed(seed)

#dataset="DRIVE_DB"
dataset = "CHASEDB1"
Dataset=dataset
dim = 192
#%%
batch_size=4
num_epochs=2


#%%
################ Parallel GPU#############

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

for d in devices:
    t = d.device_type
    name = d.physical_device_desc
    l = [item.split(':',1) for item in name.split(", ")]
    name_attr = dict([x for x in l if len(x)==2])
    dev = name_attr.get('name', 'Unnamed device')
    print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")

BATCH_SIZE = 32
#GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
GPUS = ["GPU:0"]
strategy = tf.distribute.MirroredStrategy( GPUS )
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
import time

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

tf.get_logger().setLevel('ERROR')

################################################


#%%
# -----------------------------Main Path---------------------------------
main_path = "datasets"  # <------------------------- CHANGE THIS


training_images = f"{main_path}/{dataset}/patches_{dim}/images/train/"
training_labels = f"{main_path}/{dataset}/patches_{dim}/labels/train/"

train_img = next(os.walk(training_images))[2]
train_lbs = next(os.walk(training_labels))[2]

train_img.sort()
train_lbs.sort()

X_train = np.concatenate([np.load(training_images + file_id)["arr_0"] for file_id in train_img], axis=0)
y_train = np.concatenate([np.load(training_labels + file_id)["arr_0"] for file_id in train_lbs], axis=0)

X_train = X_train / 255

if dataset == "CHASEDB1":
    y_train = y_train.astype("float32")
else:
    y_train = y_train / 255

print(np.max(X_train))
print(np.min(X_train))
print(np.max(y_train))
print(np.min(y_train))

X_train.shape

_, h, w, c = X_train.shape

input_shape = (h, w, c)
#%%
testing_images = f"{main_path}/{dataset}/patches_{dim}/images/test/"

test_img = next(os.walk(testing_images))[2]
test_img.sort()

X_test = np.concatenate([np.load(testing_images + file_id)["arr_0"] for file_id in test_img], axis=0)

X_test = X_test / 255

X_test.shape


testing_labels = f"{main_path}/{dataset}/patches_{dim}/labels/test/"

test_lbl = next(os.walk(testing_labels))[2]
test_lbl.sort()

y_test = np.concatenate([np.load(testing_labels + file_id)["arr_0"] for file_id in test_lbl], axis=0)

if dataset == "CHASEDB1":
    y_test = y_test.astype("float32")
else:
    y_test = y_test / 255

print(np.max(y_test))
print(np.min(y_test))



#%%

test_img_number = random.randint(0, X_test.shape[0]-1)  #Test with 119

#plt.imshow(X_test_images_1[test_img_number], cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.title('Image Input')
#frame = cv2.cvtColor(X_test[test_img_number,:,:], cv2.COLOR_BGRA2RGB)
plt.imshow(X_test[test_img_number,:,:])
#plt.imshow(frame)
plt.subplot(122)
plt.title('Ground Truth')
plt.imshow(y_test[test_img_number,:,:])
plt.show()




#%%
X_train_images = prog_expen(X_train)
X_test_images  = prog_expen(X_test)

#%%
_, h, w, c = X_train_images.shape

#%%
with strategy.scope():
    model=model = UPEN_Net( h, w, c, dataset)


model.summary()

#%%
filepath = ("Models_sep_22/"+model.name+"_"+Dataset+"_"+".hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_dice_coef', verbose=2, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.3, patience=50, verbose=1, cooldown=10, min_lr=1e-5)
lr_shceduler = LearningRateScheduler(lambda _, lr: lr * np.exp(-0.01), verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard("logs/1_r2a", update_freq=1)
callbacks_list = [reduce_lr, lr_shceduler, tensorboard, checkpoint]
adam = tf.keras.optimizers.Adam()
    #optm = SGD(learning_rate=0.001, decay=1e-3, momentum=0.9, nesterov=True)
model.compile(optimizer=adam, loss=dice_coef_loss, metrics=['accuracy', dice_coef])

#%%
start = time.time()
history = model.fit(X_train_images, y_train,                # Train the model using the training set...
                    batch_size=batch_size, 
                    epochs=num_epochs,
                    verbose=1, 
                    callbacks=callbacks_list,
                    validation_split=0.1) # ...holding out 10% of the data for validation
                    
end = time.time()
print("The total training Time:::: %f" %(end - start))
#%%
#model.save(filepath)
model.load_weights(filepath)
#%%

unet_history_df = pd.DataFrame(history.history) 
with open('Results_sep_22/'+model.name+Dataset+'_'+'_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)

                    
score = model.evaluate(X_test_images, y_test, verbose=2, batch_size=4)  # Evaluate the trained model on the test set!
print("Overall Accuracy - %s: %.2f%%" % (model.metrics_names[1], score[1]*100))
predictions = model.predict(X_test_images,verbose=2, batch_size=4)

predictions_norm = predictions > 0.5
n_classes = 2
IoU_values = []
for img in range(0, X_test.shape[0]):
    temp_img = X_test_images[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input,verbose=0)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)
    
                    
df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)  