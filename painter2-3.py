'''
使用预训练的各种网络进行训练
'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import numpy as np
import h5py
import os
 
from keras.utils import np_utils, conv_utils
#from keras.utils.generic_utils import unpack_singleton
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
num_classes = 71 #71  #102       #Caltech101为102  cifar10为10
pic_dir_out = './pic_out5/'#'./pic_out2/'  
pic_dir_data = './pic_dataset5/'#'./pic_dataset2/'
sub_dir = '224_resnet50/'
batch_size=128
pic_dir_mine = os.path.join(pic_dir_out,sub_dir)

def get_name_list(filepath):                #获取各个类别的名字
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        if os.path.isdir(os.path.join(filepath,allDir)):
            #child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
            out.append(allDir)
    return out
    
def eachFile(filepath):                 #将目录内的文件名放入列表中
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        #child = allDir.decode('gbk')    # .decode('gbk')是解决中文显示乱码问题
        out.append(allDir)
    return out
 
def get_data(data_name,train_left=0.0,train_right=0.7,train_all=0.7,resize=True,data_format=None,t=''):   #从文件夹中获取图像数据
    file_name = os.path.join(pic_dir_out,data_name+t+'_'+str(train_left)+'_'+str(train_right)+'_'+str(Width)+"X"+str(Height)+".h5")   
    print (file_name)
    if os.path.exists(file_name):           #判断之前是否有存到文件中
        f = h5py.File(file_name,'r')
        if t=='train':
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            f.close()
            return (X_train, y_train)
        elif t=='test':
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]
            f.close()
            return (X_test, y_test)  
        else:
            return 
    #data_format = conv_utils.normalize_data_format(data_format)
    pic_dir_set = eachFile(pic_dir_data)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir in pic_dir_set:
        #print (pic_dir_data+pic_dir)
        if not os.path.isdir(os.path.join(pic_dir_data,pic_dir)):
            continue    
        pic_set = eachFile(os.path.join(pic_dir_data,pic_dir))
        pic_index = 0
        train_count = int(len(pic_set)*train_all)
        train_l = int(len(pic_set)*train_left)
        train_r = int(len(pic_set)*train_right)
        for pic_name in pic_set:  
            if (pic_index < train_count):
                if t=='train':
                    if not os.path.isfile(os.path.join(pic_dir_data,pic_dir,pic_name)):
                        continue        
                    img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                    if img is None:
                        continue
                    if (resize):
                        img = cv2.resize(img,(Width,Height))   
                        img = img.reshape(-1,Width,Height,3)
                    if (pic_index >= train_l and pic_index < train_r):
                        X_train.append(img)
                        y_train.append(label)  
            else:
                if t=='test':
                    if not os.path.isfile(os.path.join(pic_dir_data,pic_dir,pic_name)):
                        continue        
                    img = cv2.imread(os.path.join(pic_dir_data,pic_dir,pic_name))
                    if img is None:
                        continue
                    if (resize):
                        img = cv2.resize(img,(Width,Height))   
                        img = img.reshape(-1,Width,Height,3)
                    X_test.append(img)
                    y_test.append(label)
            pic_index += 1
        if len(pic_set) != 0:        
            label += 1
    
    f = h5py.File(file_name,'w') 
    if t=='train':
        X_train = np.concatenate(X_train,axis=0)     
        y_train = np.array(y_train)      
        f.create_dataset('X_train', data = X_train)
        f.create_dataset('y_train', data = y_train)
        f.close()
        return (X_train, y_train)
    elif t=='test':
        X_test = np.concatenate(X_test,axis=0) 
        y_test = np.array(y_test)
        f.create_dataset('X_test', data = X_test)
        f.create_dataset('y_test', data = y_test)
        f.close()
        return (X_test, y_test)   
    else:
        return
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
    
    (X_train, y_train) = get_data("Kaggle_data_",0.0,0.7,data_format='channels_last',t='train')
    y_train = np_utils.to_categorical(y_train, num_classes)
    (X_test, y_test) = get_data("Kaggle_data_",0.0,0.7,data_format='channels_last',t='test')    
    y_test = np_utils.to_categorical(y_test, num_classes)

    # resnet50_train_output = np.concatenate(X_train,axis=0)
    # resnet50_test_output = np.concatenate(X_test,axis=0)

    input_tensor = Input(shape=(Width, Height, 3),name='image_input')
    base_model = ResNet50(input_tensor=input_tensor,include_top=False,weights='imagenet')
    #base_model = InceptionResNetV2(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = VGG19(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = InceptionV3(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = DenseNet201(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')#'imagenet'
    # get_resnet50_output = K.function([base_model.layers[0].input, K.learning_phase()],
    #                           [base_model.layers[-1].output])
    for layer in base_model.layers:
        layer.trainable = False
        #print (layer)
    pooling=AveragePooling2D((7, 7), name='avg_pool')(base_model.output)
    fl = Flatten(name='flatten')(pooling)
    dense = Dense(1024, activation='relu', name='fc1')(fl)
    #drop = Dropout(0.5, name='drop')(dense)
    pred = Dense(num_classes, activation='softmax', name='predictions')(dense)

    fine_model = Model(inputs=input_tensor,outputs=pred)
    
    
    '''
    ResNet50输出尺寸(7, 7, 2048)     epoch30
    DenseNet201输出尺寸(7,7,1920)    epoch50
    InceptionResNetV2输出尺寸(5,5,1536)不行的
    VGG19输出尺寸(7,7,512)
    InceptionV3输出尺寸(5,5,2048)
    '''
    
    fine_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    #fine_model.summary()
    h=fine_model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=50,batch_size=batch_size,verbose=1)
    #model.fit(resnet50_train_output, y_train, epochs=100, batch_size=64,) #batch_size有没有影响
    #model.save_weights(os.path.join(pic_dir_mine,'cnn_model_Kaggle_resnet50_'+cm2_str+'.h5'))
    accuracy_curve(h)
      
if __name__ == '__main__':
    main()
