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
# import densenet   #取消densenet模型

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

    def model_confirm(self,choosed_model):


        if choosed_model == 'VGG16':
            model = keras.applications.VGG16(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)
        elif choosed_model == 'VGG19':
            model = keras.applications.VGG19(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)

        elif choosed_model == 'ResNet50':
            model = keras.applications.ResNet50(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)
        elif choosed_model == 'InceptionV3':
            model = keras.applications.InceptionV3(include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   input_shape=(self.normal_size,self.normal_size,self.channles),
                                                   pooling='max',
                                                   classes=self.classes)

        elif choosed_model == 'Xception':
            model = keras.applications.Xception(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=(self.normal_size,self.normal_size,self.channles),
                                                pooling='max',
                                                classes=self.classes)
        elif choosed_model == 'MobileNet':
            model = keras.applications.MobileNet(include_top=True,
                                                 weights=None,
                                                 input_tensor=None,
                                                 input_shape=(self.normal_size,self.normal_size,self.channles),
                                                 pooling='max',
                                                 classes=self.classes)
        elif choosed_model == 'DenseNet':
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

        elif choosed_model == 'AlexNet':
            model = MODEL(self.config).AlexNet()
        elif choosed_model == 'LeNet':
            model = MODEL(self.config).LeNet()
        elif choosed_model == 'ZF_Net':
            model = MODEL(self.config).ZF_Net()
        elif choosed_model == 'ResNet18':
            model = ResnetBuilder().build_resnet18(self.config)
        elif choosed_model == 'ResNet34':
            model = ResnetBuilder().build_resnet34(self.config)
        elif choosed_model == 'ResNet101':
            model = ResnetBuilder().build_resnet101(self.config)
        elif choosed_model == 'ResNet152':
            model = ResnetBuilder().build_resnet152(self.config)
        elif choosed_model =='mnist_net':
            model = MODEL(self.config).mnist_net()
        elif choosed_model == 'VGG16_TSL':
            model = MODEL(self.config).VGG16_TSL()

        return model

    def model_compile(self,model):
        if self.default_optimizers:
            adam = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  # compile之后才会更新权重和模型
        else:
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def build_model(self):

        model = self.model_confirm(self.model_name)
        model = self.model_compile(model)

        return model