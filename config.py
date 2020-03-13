# -*- coding: utf-8 -*-
"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just choose in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,
    ResNet18,ResNet34,ResNet50,ResNet101,ResNet152,mnist_net,TSL16
"""

import sys
class DefaultConfig():

    try:
        model_name = sys.argv[1]
    except:
        print("use default model VGG16, see config.py")
        model_name = "VGG16"

    train_data_path = 'dataset/train'
    test_data_path = 'dataset/test'
    checkpoints = './checkpoints/'

    # if model_name == 'InceptionV3':
    #     normal_size = 75#minSize
    # elif model_name == 'Xception':
    #     normal_size = 71#minSize
    # else:
    #     normal_size = 224

    normal_size = 224
    epochs = 1
    batch_size = 2
    classNumber = 2 # see dataset
    channles = 3 # or 3 or 1
    lr = 0.001

    lr_reduce_patience = 5  # 需要降低学习率的训练步长
    early_stop_patience = 10  # 提前终止训练的步长

    data_augmentation = False
    monitor = 'val_loss'
    cut = False
    rat = 0.1 #if cut,img[slice(h*self.rat,h-h*self.rat),slice(w*self.rat,w-w*self.rat)]

config = DefaultConfig()
