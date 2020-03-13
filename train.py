"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,esNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
"""

from __future__ import print_function
from config import config
import numpy as np
import os,glob,itertools,tqdm,cv2,keras
from random import shuffle

from keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard

import tensorflow as tf
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)



import sys
sys.setrecursionlimit(10000)

from Build_model import Build_model

class Train(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)

    def get_file(self,path):
        ends = os.listdir(path)[0].split('.')[-1]
        img_list = glob.glob(os.path.join(path , '*.'+ends))
        return img_list

    def load_data(self):

        categories = list(map(self.get_file, list(map(lambda x: os.path.join(self.train_data_path, x), os.listdir(self.train_data_path)))))
        data_list = list(itertools.chain.from_iterable(categories))
        shuffle(data_list)
        images_data ,labels_idx,labels= [],[],[]

        with_platform = os.name

        for file in tqdm.tqdm(data_list):
            if self.channles == 3:
                img = cv2.imread(file)
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # img = cv2.threshold(img,128,255,cv2.THRESH_BINARY)[-1]
                _, w, h = img.shape[::-1]
            elif self.channles == 1:
                # img=cv2.threshold(cv2.imread(file,0), 128, 255, cv2.THRESH_BINARY)[-1]
                img = cv2.imread(file,0)
                # img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[-1]
                w, h = img.shape[::-1]

            if self.cut:
                img = img[slice(int(h*self.rat),int(h-h*self.rat)),slice( int(w*self.rat),int(w-w*self.rat) )]
            img = cv2.resize(img,(self.normal_size,self.normal_size))
            if with_platform == 'posix':
                label = file.split('/')[-2]
            elif with_platform=='nt':
                label = file.split('\\')[-2]

            # print('img:',file,' has label:',label)
            img = img_to_array(img)
            images_data.append(img)
            labels.append(label)

        with open('train_class_idx.txt','r') as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for label in labels:
                idx = lines.index(label.rstrip())
                labels_idx.append(idx)

        # images_data = np.array(images_data,dtype='float')/255.0
        images_data = np.array(images_data, dtype='float32') / 255.0
        labels = to_categorical(np.array(labels_idx),num_classes=self.classNumber)
        X_train, X_test, y_train, y_test = train_test_split(images_data,labels)
        return X_train, X_test, y_train, y_test

    def mkdir(self,path):
        if not os.path.exists(path):
            return os.mkdir(path)
        return path

    def train(self,X_train, X_test, y_train, y_test,model):
        print("*"*50)
        print("-"*20+"train",config.model_name+"-"*20)
        print("*"*50)

        tensorboard=TensorBoard(log_dir=self.mkdir(os.path.join(self.checkpoints,self.model_name) ))

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor=config.monitor,
                                                      factor=0.1,
                                                      patience=config.lr_reduce_patience,
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                   min_delta=0,
                                                   patience=config.early_stop_patience,
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.mkdir( os.path.join(self.checkpoints,self.model_name) ),self.model_name+'.h5'),
                                                     monitor=config.monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='auto',
                                                     period=1)

        if self.data_augmentation:
            print("using data augmentation method")
            data_aug = ImageDataGenerator(
                rotation_range=5,  # 图像旋转的角度
                width_shift_range=0.2,  # 左右平移参数
                height_shift_range=0.2,  # 上下平移参数
                zoom_range=0.3,  # 随机放大或者缩小
                horizontal_flip=True,  # 随机翻转
            )

            data_aug.fit(X_train)
            model.fit_generator(
                data_aug.flow(X_train, y_train, batch_size=config.batch_size),
                steps_per_epoch=X_train.shape[0] // self.batch_size,
                validation_data=(X_test, y_test),
                shuffle=True,
                epochs=self.epochs, verbose=1, max_queue_size=1000,
                callbacks=[early_stop,checkpoint,lr_reduce,tensorboard],
            )
        else:
            model.fit(x=X_train,y=y_train,
                      batch_size=self.batch_size,
                      validation_data=(X_test,y_test),
                      epochs=self.epochs,
                      callbacks=[early_stop,checkpoint,lr_reduce,tensorboard],
                      shuffle=True,
                      verbose=1)

    def start_train(self):
        X_train, X_test, y_train, y_test=self.load_data()
        model = Build_model(config).build_model()
        self.train(X_train, X_test, y_train, y_test,model)

    def remove_logdir(self):
        self.mkdir(self.checkpoints)
        self.mkdir(os.path.join(self.checkpoints,self.model_name))
        events = os.listdir(os.path.join(self.checkpoints,self.model_name))
        for evs in events:
            if "events" in evs:
                os.remove(os.path.join(os.path.join(self.checkpoints,self.model_name),evs))

    def mkdir(self,path):
        if os.path.exists(path):
            return path
        os.mkdir(path)
        return path


def main():
    train = Train(config)
    train.remove_logdir()
    train.start_train()
    print('Done')

if __name__=='__main__':
    main()
