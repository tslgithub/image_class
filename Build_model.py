#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:"tsl"
# email:"mymailwith163@163.com"
# datetime:19-1-17 下午3:07
# software: PyCharm

from __future__ import print_function
import keras
from MODEL import MODEL,ResnetBuilder
import sys
sys.setrecursionlimit(10000)

from keras import backend as K
import densenet

class Build_model(object):
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
        self.rat = config.rat
        self.cut = config.cut


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
        elif self.model_name == 'DesenNet':
            depth = 40
            nb_dense_block = 3
            growth_rate = 12
            nb_filter = 12
            bottleneck = False
            reduction = 0.0
            dropout_rate = 0.0

            img_dim = (self.channles, self.normal_size) if K.image_dim_ordering() == "th" else (
                self.normal_size, self.normal_size, self.channles)

            model = densenet.DenseNet(img_dim, classes=self.classes, depth=depth, nb_dense_block=nb_dense_block,
                                      growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                                      bottleneck=bottleneck, reduction=reduction, weights=None)

        elif self.model_name == 'AlexNet':
            model = MODEL(self.config).AlexNet()
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