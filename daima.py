'''基于MXNET的分类网络的实现'''
from mxnet import autograd,gluon,init,nd,image
import gluonbook as gb
from mxnet.gluon import data as gdata,nn,loss as gloss,model_zoo
import os
import pandas as pd
import numpy as np
import datetime
import random
import csv

#prefix=r"/media/wzz/48D47553D47543EA/data/painter/"
#foldername=prefix+"train/"
DATA_DIR="./pic_dataset/"

def pro_csv():
    content=[]
    for dir in os.listdir(DATA_DIR):
        for file in os.listdir(DATA_DIR+dir):
            name=os.path.join(dir,file)
            label=int(dir)
            content.append([name,label])
    file=open('train_label.csv','w')
    writer=csv.writer(file)
    writer.writerows(content)
    file.close()

def test_iterator(batch_size,train=False):
    rgb_mean=nd.array([0.485,0.456,0.406])
    rgb_std=nd.array([0.229,0.224,0.225])
    #csv_fname='%sgt/minidata/%s'%(prefix,'train_label.csv' if train else 'test_label.csv')
    csv_fname='./train_label.csv'
    info=pd.read_csv(csv_fname).as_matrix()
    num=len(info)
    idx=list(range(num))
    random.shuffle(idx)
    for i in range(0,num,batch_size):
        j=idx[i:min(i+batch_size,num)]
        data,label=[],[]
        for k in j:
            try:
                img=image.imread(DATA_DIR+info[k][0])
                img=(img.astype('float32')/255-rgb_mean)/rgb_std
                img=img.transpose((2, 0, 1)).asnumpy()
                data.append(img)
                label.append(info[k][1])
                braek
            except:
                continue
        yield  nd.array(data),nd.array(label),len(label)


'''def train_iterator(train_folder,batch_size):
    rgb_mean=nd.array([0.485,0.456,0.406])
    rgb_std=nd.array([0.229,0.224,0.225])
    l=os.listdir(train_folder)
    num=len(l)
    idx=list(range(num))
    random.shuffle(idx)
    for i in range(0,num,batch_size):
        j=idx[i:min(i+batch_size,num)]
        data,label=[],[]
        for k in j:
            img=image.imread(train_folder+l[k])
            img=(img.astype('float32')/255-rgb_mean)/rgb_std
            img=img.transpose((2, 0, 1)).asnumpy()
            data.append(img)
            if l[k][1]!='_':
                label.append(l[k][:2])
            else:
                label.append(l[k][0])
        yield nd.array(data),nd.array(label),len(label)'''
def train_iterator(batch_size):
    rgb_mean=nd.array([0.485,0.456,0.406])
    rgb_std=nd.array([0.229,0.224,0.225])
    #csv_fname='%sgt/minidata/%s'%(prefix,'train_label.csv' if train else 'test_label.csv')
    csv_fname='./train_label.csv'
    info=pd.read_csv(csv_fname).as_matrix()
    num=len(info)
    idx=list(range(num))
    random.shuffle(idx)
    for i in range(0,num,batch_size):
        j=idx[i:min(i+batch_size,num)]
        data,label=[],[]
        for k in j:
            try:
                img=image.imread(DATA_DIR+info[k][0])
                img=(img.astype('float32')/255-rgb_mean)/rgb_std
                img=img.transpose((2, 0, 1)).asnumpy()
                data.append(img)
                label.append(info[k][1])
            except:
                continue
        yield nd.array(data),nd.array(label),len(label)

#train_folder=r"/media/wzz/48D47553D47543EA/data/painter/gt/augdata/"
batch_size=64
# train_iter=train_iterator(train_folder,batch_size)
# test_iter=test_iterator(prefix,foldername,batch_size,train=False)
# for img,label,n in train_iter:
#     print (img.shape)
#     print (label.shape)
#     print (n)
#     break



def get_net(ctx):
#     finetune_net=model_zoo.vision.resnet34_v2(pretrained=True)
#     finetune_net=model_zoo.vision.resnet50_v2(pretrained=True)
#     finetune_net=model_zoo.vision.resnet101_v2(pretrained=True)
    finetune_net=model_zoo.vision.resnet152_v2(pretrained=True)
    #finetune_net=model_zoo.vision.alexnet(pretrained=True)
#     finetune_net=model_zoo.vision.inception_v3(pretrained=True)
#     finetune_net=model_zoo.vision.densenet121(pretrained=True)
#     finetune_net=model_zoo.vision.densenet161(pretrained=True)
#     finetune_net=model_zoo.vision.densenet201(pretrained=True)
    finetune_net.output_new=nn.HybridSequential(prefix='')
    with finetune_net.name_scope():
        finetune_net.output_new.add(nn.Dense(1024,activation='relu'))
        finetune_net.output_new.add(nn.Dropout(0.5))
        finetune_net.output_new.add(nn.Dense(256,activation='relu'))
        finetune_net.output_new.add(nn.Dropout(0.5))
        #31是输出的类别数
        finetune_net.output_new.add(nn.Dense(71))
     
    finetune_net.output_new.initialize(init.Xavier(),ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net

loss=gloss.SoftmaxCrossEntropyLoss()


def evaluate_accuracy(data_iter,net,ctx):
    acc=0.
    num=0.
    test_n=0
    for X,y,n in data_iter:
        if n==0:
            continue
        y=y.astype('float32').as_in_context(ctx)
        output_features=net.features(X.as_in_context(ctx))
        y_hat=net.output_new(output_features)
        acc+=gb.accuracy(y_hat,y)
        num+=1
        test_n+=n
#     print ("test samples:%d"%(test_n))
    if num==0.:
        return acc
    else:
        return acc/num


def train(net,train_iterator,test_iterator,num_epochs,lr,wd,ctx,lr_period,lr_decay):
    # 只训练我们定义的输出网络
    trainer=gluon.Trainer(net.output_new.collect_params(),'sgd',
                          {'learning_rate':lr,'momentum':0.9,'wd':wd})

    prev_time=datetime.datetime.now()
    for epoch in range(num_epochs):
#         train_iter=data_iter(prefix,foldername,batch_size,True)
        train_iter=train_iterator(batch_size)
        test_iter=test_iterator(batch_size,False)
        train_l=0.0
        train_acc=0.0
        train_n=0
#         if epoch > 0 and epoch % lr_period == 0:
#             trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        num=0.
        for X,y,n in train_iter:
            if n==0:
                continue
            y=y.astype('float32').as_in_context(ctx)
            output_features=net.features(X.as_in_context(ctx))
            with autograd.record():
                y_hat=net.output_new(output_features)
                l=loss(y_hat,y)
            l.backward()
            trainer.step(n)
            train_l+=l.mean().asscalar()
            train_acc+=gb.accuracy(y_hat,y)
            num+=1
            train_n+=n
#         print ("train samples:%d"%(train_n))
        cur_time=datetime.datetime.now()
        h,remainder=divmod((cur_time-prev_time).seconds,3600)
        m,s=divmod(remainder,60)
        time_s="time %02d:%02d:%02d"%(h,m,s)
        test_acc=evaluate_accuracy(test_iter,net,ctx)
        if num==0.:
            epoch_s=("epoch %d, loss %f, train acc %f, test acc %f, "
                      % (epoch, train_l ,
                      train_acc, test_acc))
        else:
            epoch_s=("epoch %d, loss %f, train acc %f, test acc %f, "
                      % (epoch, train_l / num,
                      train_acc / num, test_acc))
        prev_time=cur_time
        print(epoch_s+time_s+',lr '+str(trainer.learning_rate))

if __name__ == '__main__':
    ctx=gb.try_gpu()
    print (ctx)
    wd=1e-4
    lr_period=10
    lr_decay=0.1
    net=get_net(ctx)
    net.hybridize()
    num_epochs=20
    lr=0.001
    train(net,train_iterator,test_iterator,num_epochs,lr,wd,ctx,lr_period,lr_decay)
    #pro_csv()

