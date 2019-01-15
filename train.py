"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,esNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
"""

from config import config
import numpy as np
import os,glob,itertools,tqdm,cv2,keras

from keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard


from MODEL import MODEL,ResnetBuilder

import tensorflow as tf

config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)


class Train(object):
    def __init__(self,config):
        self.train_data_path = config.train_data_path
        self.checkpoints = config.checkpoints
        self.normal_size = config.normal_size
        self.channles = config.channles
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.classes = config.classes
        self.model_name = config.model_name
        self.lr = config.lr
        self.config = config
        self.default_optimizers = config.default_optimizers
        self.data_augmentation = config.data_augmentation

    def get_file(self,path):
        ends = os.listdir(path)[0].split('.')[1]
        return glob.glob(path + '/*.'+ends)

    def load_data(self):
        categories = list(map(self.get_file, list(map(lambda x: self.train_data_path + x, os.listdir(self.train_data_path)))))
        data_list = itertools.chain.from_iterable(categories)
        images_data ,labels= [],[]

        for file in tqdm.tqdm(data_list):
            img = cv2.imread(file,int(self.channles/3))
            img = cv2.resize(img,(self.normal_size,self.normal_size))
            label = file.split('/')[3]
            img = img_to_array(img)
            images_data.append(img)
            labels.append(label)

        images_data = np.array(images_data,dtype='float')/255.0
        labels = to_categorical(np.array(labels),num_classes=self.classes)
        X_train, X_test, y_train, y_test = train_test_split(images_data,labels)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        if self.model_name == 'VGG16':
            model = keras.applications.VGG16(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)
        elif self.model_name == 'VGG19':
            model = keras.applications.VGG19(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)

        elif self.model_name == 'ResNet50':
            model = keras.applications.ResNet50(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)
        elif self.model_name == 'InceptionV3':
            model = keras.applications.InceptionV3(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)

        elif self.model_name == 'Xception':
            model = keras.applications.Xception(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=(self.normal_size,self.normal_size,self.channles),
                                                pooling='max',
                                                classes=self.classes)
        elif self.model_name == 'MobileNet':
            model = keras.applications.MobileNet(include_top=True,
                                                 weights=None,
                                                 input_tensor=None,
                                                 input_shape=(self.normal_size,self.normal_size,self.channles),
                                                 pooling='max',
                                                 classes=self.classes)

        elif self.model_name == 'AlexNet':
            model = MODEL(self.config).AlexNet()
        elif self.model_name == 'GoogLeNet':
            model = MODEL(self.config).GoogLeNet()
        elif self.model_name == 'LeNet':
            model = MODEL(self.config).LeNet()
        elif self.model_name == 'ZF_Net':
            model = MODEL(self.config).ZF_Net()
        elif self.model_name == 'ResNet18':
            model = ResnetBuilder().build_resnet_18(self.config)
        elif self.model_name == 'ResNet34':
            model = ResnetBuilder().build_resnet_34(self.config)
        elif self.model_name == 'ResNet_101':
            model = ResnetBuilder().build_resnet_101(self.config)
        elif self.model_name == 'ResNet_152':
            model = ResnetBuilder().build_resnet_152(self.config)


        if self.default_optimizers:
            adam = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  # compile之后才会更新权重和模型
        else:
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train(self,X_train, X_test, y_train, y_test,model):
        tensorboard=TensorBoard(log_dir=self.checkpoints)

        lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=5,
                                                      verbose=1,
                                                      mode='auto',
                                                      cooldown=0)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=10,
                                                   verbose=1,
                                                   mode='auto')
        checkpoint = keras.callbacks.ModelCheckpoint(self.checkpoints+self.model_name+'.h5',
                                                     monitor='val_loss',
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
                epochs=self.epochs, verbose=1, max_queue_size=100,
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
        model = self.build_model()
        self.train(X_train, X_test, y_train, y_test,model)

def main():
    train = Train(config)
    train.start_train()
    print('Done')

if __name__=='__main__':
    main()