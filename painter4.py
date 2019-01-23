'''
用ResNet50从头开始训练
由于内存的限制
现在采用迭代器生成训练集和验证集
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import h5py
import os
 
from keras.utils import np_utils, conv_utils
from keras.utils.generic_utils import unpack_singleton
from keras.models import Model
#from keras.layers import Flatten, Dense, Input 
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras import backend as K
#from past.builtins import xrange
from keras.layers import AveragePooling2D,GlobalAveragePooling2D

from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Flatten
from keras.layers import add 
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

Width = 224
Height = 224
num_classes = 71  #102       #Caltech101为102  cifar10为10
pic_dir_out = './pic_out4/'#'./pic_out2/'  
pic_dir_data = './pic_dataset/'#'./pic_dataset2/'
sub_dir = '224_resnet50/'
batch_size=32
pic_dir_mine = os.path.join(pic_dir_out,sub_dir)

train_datagen = ImageDataGenerator(
    data_format = 'channels_last',
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=0.3
)
train_generator = train_datagen.flow_from_directory(
    pic_dir_data,#+'train/',
    target_size=(Width, Height),
    batch_size=batch_size,
    shuffle=True,
    seed=208,
    subset="training"
)

# valid_datagen = ImageDataGenerator( 
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     rescale=1./255
# )
vaild_generator = train_datagen.flow_from_directory(
    pic_dir_data,#+'val/',
    target_size=(Width, Height),
    batch_size=batch_size,
    shuffle=True,
    seed=208,
    subset="validation"
)

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet_50(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    #x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show() 
    
def main():

    es = EarlyStopping(monitor='val_acc', patience=20)
    model = resnet_50(Width,Height,3,num_classes)
    #model.summary()
    
    # Save a PNG of the Model Build
    #plot_model(model, to_file='./resnet.png')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    h=model.fit_generator(train_generator,validation_data=vaild_generator,epochs=50,steps_per_epoch=train_generator.n//batch_size
                        ,validation_steps=vaild_generator.n//batch_size,verbose=1,callbacks=[es])
    
    #model.save_weights(os.path.join(pic_dir_mine,'cnn_model_Kaggle_resnet50.h5'))
    accuracy_curve(h)
      
if __name__ == '__main__':
    main()
