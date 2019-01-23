'''
最终代码
使用预训练的各种网络进行训练
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
from keras.callbacks import ModelCheckpoint
# from keras.layers import Flatten, Dense, Input 
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet201
from keras import backend as K
# from past.builtins import xrange
# from keras.layers import AveragePooling2D,GlobalAveragePooling2D

from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D, Flatten
# from keras.layers import add 
from keras.utils import plot_model
# from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

Width = 224
Height = 224
num_classes = 31 #71  #102       #Caltech101为102  cifar10为10
pic_dir_data = './pic_dataset3/'#'./pic_dataset2/'#用于训练的数据集
sub_dir = '224_resnet50/'
pic_dir_out = './pic_out5/'#实验结果输出目录,主要是权重信息
batch_size=10
MODEL_FILE=os.path.join(pic_dir_out, 'cnn.h5')

#数据增强
train_datagen = ImageDataGenerator( 
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    #shear_range=0.1,
    zoom_range=0.1,
    #horizontal_flip=True,
    rescale=1./255,
    data_format = 'channels_last',
    validation_split=0.1
)
train_generator = train_datagen.flow_from_directory(
    pic_dir_data,
    target_size=(Width, Height),
    batch_size=batch_size,
    shuffle=True,
    seed=0,
    subset="training"
)

#验证集不用数据增强
# valid_datagen = ImageDataGenerator( 
#     #width_shift_range=0.1,
#     #height_shift_range=0.1,
#     #shear_range=0.1,
#     #zoom_range=0.1,
#     #horizontal_flip=True,
#     rescale=1./255
# )
vaild_generator = train_datagen.flow_from_directory(
    pic_dir_data,
    target_size=(Width, Height),
    batch_size=batch_size,
    shuffle=True,
    seed=0,
    subset="validation"
)
#画训练过程中loss和acc的曲线
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
    
    es = EarlyStopping(monitor='val_acc', patience=10)

    #定义各种网络模型用哪个就把哪个的注释去掉
    input_tensor = Input(shape=(Width, Height, 3),name='image_input')
    #base_model = ResNet50(input_tensor=input_tensor,include_top=False,weights='imagenet')
    #base_model = InceptionResNetV2(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = VGG19(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = InceptionV3(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    base_model = DenseNet201(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')#'imagenet'
    # get_resnet50_output = K.function([base_model.layers[0].input, K.learning_phase()],
    #                           [base_model.layers[-1].output])
    
    #控制预训练网络的参数可改变与否，实验发现改变可以获得更好效果
    # for layer in base_model.layers:
    #     layer.trainable = False
    #     #print (layer)

    #网络的尾部
    '''pooling=AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    fl = Flatten(name='flatten')(pooling)
    dense = Dense(1024, activation='relu', name='fc1')(fl)
    #drop = Dropout(0.5, name='drop')(dense)#ResNet不需要
    pred = Dense(num_classes, activation='softmax', name='predictions')(dense)'''
    #print(base_model.output.shape)
    pooling=AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    fl = Flatten(name='flatten')(pooling)
    dense = Dense(1024, activation='relu', name='fc1')(fl)
    #drop = Dropout(0.5, name='drop')(dense)#ResNet不需要
    pred = Dense(num_classes, activation='softmax', name='predictions')(dense)

    fine_model = Model(inputs=input_tensor,outputs=pred)
    
    '''
    ResNet50输出尺寸(7, 7, 2048)
    DenseNet201输出尺寸(7,7,1920)
    InceptionResNetV2输出尺寸(5,5,1536)不行的
    VGG19输出尺寸(7,7,512)
    InceptionV3输出尺寸(5,5,2048)
    '''
    
    fine_model.compile(optimizer=Adam(lr=0.00005,decay=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])
    #fine_model.summary()
    h=fine_model.fit_generator(train_generator,validation_data=vaild_generator,epochs=100,steps_per_epoch=train_generator.n//batch_size,
                        validation_steps=vaild_generator.n//batch_size,verbose=1,callbacks=[es,ModelCheckpoint(MODEL_FILE, save_best_only=True)])
    #model.fit(resnet50_train_output, y_train, epochs=100, batch_size=64,) #batch_size有没有影响
    accuracy_curve(h)
    #fine_model.save_weights(os.path.join(pic_dir_out,'cnn_model_Kaggle_resnet50.h5'))
    
      
if __name__ == '__main__':
    main()
