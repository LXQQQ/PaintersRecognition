'''
用ResNet50从头开始训练
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
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

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
 
def main():
    global Width, Height, pic_dir_out, pic_dir_data
    Width = 224
    Height = 224
    num_classes = 71       #Caltech101为102  cifar10为10
    pic_dir_out = './pic_out3/'#'./pic_out2/'  
    pic_dir_data = './pic_dataset/'#'./pic_dataset2/'
    sub_dir = '224_resnet50/'
    if not os.path.isdir(os.path.join(pic_dir_out,sub_dir)):
        os.mkdir(os.path.join(pic_dir_out,sub_dir))
    pic_dir_mine = os.path.join(pic_dir_out,sub_dir)
    (X_train, y_train) = get_data("Kaggle_data_",0.0,0.7,data_format='channels_last',t='train')
    y_train = np_utils.to_categorical(y_train, num_classes)
 
    # input_tensor = Input(shape=(224, 224, 3))
    # base_model = ResNet50(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')    #pic_out1
    #base_model = InceptionResNetV2(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = VGG19(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = InceptionV3(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')
    #base_model = DenseNet201(input_tensor=input_tensor,include_top=False,weights='imagenet')#,pooling = 'avg')#'imagenet'
    
    model = resnet_50(Width,Height,3,num_classes)
    #model.summary()
    
    # Save a PNG of the Model Build
    #plot_model(model, to_file='./resnet.png')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # get_resnet50_output = K.function([base_model.layers[0].input, K.learning_phase()],
    #                           [base_model.layers[-1].output])

    #file_name = os.path.join(pic_dir_mine,'resnet50_train_output'+'.h5')
    # if os.path.exists(file_name):
    #     f = h5py.File(file_name,'r')
    #     resnet50_train_output = f['resnet50_train_output'][:]
    #     f.close()
    # else:

    # resnet50_train_output = []
    # delta = 10
    # for i in range(0,len(X_train),delta):  
    #     one_resnet50_train_output = get_resnet50_output([X_train[i:i+delta], 0])[0]
    #     resnet50_train_output.append(one_resnet50_train_output)
    # resnet50_train_output = np.concatenate(resnet50_train_output,axis=0) 

        # f = h5py.File(file_name,'w')          
        # f.create_dataset('resnet50_train_output', data = resnet50_train_output)
        # f.close()
    #print('kankan'+str(resnet50_train_output.shape))
    #print(base_model.layers[-1].output.shape[1:4])
    #input_tensor = Input(shape=(7,7,2048))#shape=(7, 7, 2048)(5,5,1536)(7,7,1920)

    #input_tensor = Input(shape=(7,7,2048))
    '''
    ResNet50输出尺寸(7, 7, 2048)     epoch30
    DenseNet201输出尺寸(7,7,1920)    epoch50
    InceptionResNetV2输出尺寸(5,5,1536)不行的
    VGG19输出尺寸(7,7,512)
    InceptionV3输出尺寸(5,5,2048)
    '''
    
    # x = AveragePooling2D((5, 5), name='avg_pool')(input_tensor)
    # x = Flatten()(x)
    # x = Dense(1024, activation='relu')(x)

    # #x = GlobalAveragePooling2D(name='avg_pool')(input_tensor)
    # predictions = Dense(num_classes, activation='softmax')(x)   
    # model = Model(inputs=input_tensor, outputs=predictions)
    # model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
    
    print('\nTraining ------------')    #从文件中提取参数，训练后存在新的文件中
    cm = 0                              #修改这个参数可以多次训练
    cm_str = '' if cm==0 else str(cm)
    cm2_str = '' if (cm+1)==0 else str(cm+1) 
    #if cm >= 1:
    #    model.load_weights(os.path.join(pic_dir_mine,'cnn_model_Kaggle_resnet50_'+cm_str+'.h5'))
    model.fit(X_train, y_train, epochs=20, batch_size=25) #batch_size有没有影响
    model.save_weights(os.path.join(pic_dir_mine,'cnn_model_Kaggle_resnet50_'+cm2_str+'.h5'))
    
    (X_test, y_test) = get_data("Kaggle_data_",0.0,0.7,data_format='channels_last',t='test')    
    y_test = np_utils.to_categorical(y_test, num_classes)
        
    # file_name = os.path.join(pic_dir_mine,'resnet50_test_output'+'.h5')
    # if os.path.exists(file_name):
    #     f = h5py.File(file_name,'r')
    #     resnet50_test_output = f['resnet50_test_output'][:]
    #     f.close()
    # else:

    # resnet50_test_output = []
    # delta = 10
    # for i in range(0,len(X_test),delta):
    #         #print(i)
    #     one_resnet50_test_output = get_resnet50_output([X_test[i:i+delta], 0])[0]
    #     resnet50_test_output.append(one_resnet50_test_output)
    # resnet50_test_output = np.concatenate(resnet50_test_output,axis=0)

        # f = h5py.File(file_name,'w')          
        # f.create_dataset('resnet50_test_output', data = resnet50_test_output)
        # f.close()
    print('\nTesting ------------')     #对测试集进行评估
    class_name_list = get_name_list(pic_dir_data)    #获取top-N的每类的准确率
    pred = model.predict(X_test, batch_size=32)
    f = h5py.File(os.path.join(pic_dir_mine,'pred_'+cm2_str+'.h5'),'w')          
    f.create_dataset('pred', data = pred)
    f.close()
    
    N = 1
    pred_list = []
    for row in pred:
        pred_list.append(row.argsort()[-N:][::-1])  #获取最大的N个值的下标
    pred_array = np.array(pred_list)
    test_arg = np.argmax(y_test,axis=1)
    class_count = [0 for _ in range(num_classes)]
    class_acc = [0 for _ in range(num_classes)]
    for i in range(len(test_arg)):
        class_count[test_arg[i]] += 1
        if test_arg[i] in pred_array[i]:
            class_acc[test_arg[i]] += 1
    print('top-'+str(N)+' all acc:',str(sum(class_acc))+'/'+str(len(test_arg)),sum(class_acc)/float(len(test_arg)))
    #for i in range(num_classes):
    #    print (i, class_name_list[i], 'acc: '+str(class_acc[i])+'/'+str(class_count[i]))
    
    '''print('----------------------------------------------------')
    N = 5
    pred_list = []
    for row in pred:
        pred_list.append(row.argsort()[-N:][::-1])  #获取最大的N个值的下标
    pred_array = np.array(pred_list)
    test_arg = np.argmax(y_test,axis=1)
    class_count = [0 for _ in range(num_classes)]
    class_acc = [0 for _ in range(num_classes)]
    for i in range(len(test_arg)):
        class_count[test_arg[i]] += 1
        if test_arg[i] in pred_array[i]:
            class_acc[test_arg[i]] += 1
    print('top-'+str(N)+' all acc:',str(sum(class_acc))+'/'+str(len(test_arg)),sum(class_acc)/float(len(test_arg)))
    for i in range(num_classes):
        print (i, class_name_list[i], 'acc: '+str(class_acc[i])+'/'+str(class_count[i]))'''
      
if __name__ == '__main__':
    main()
