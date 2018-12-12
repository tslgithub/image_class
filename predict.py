"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,esNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
"""

from config import config
import keras
import sys
import cv2
import os
from keras.preprocessing.image import img_to_array
import numpy as np

from MODEL import MODEL,ResnetBuilder

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config1 = tf.ConfigProto()
config1.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config1))

class PREDICT(object):
    def __init__(self,config):
        self.test_data_path = config.test_data_path+sys.argv[1]
        self.channles = config.channles
        self.checkpoints = config.checkpoints
        self.normal_size = config.normal_size
        self.model_name = config.model_name
        self.classes = config.classes
        self.config = config

    def Predict(self):
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
                                             input_shape=(self.normal_size, self.normal_size, self.channles),
                                             pooling='max',
                                             classes=self.classes)
        elif self.model_name=='ResNet50':
            model = keras.applications.ResNet50(include_top=True,
                                             weights=None,
                                             input_tensor=None,
                                             input_shape=(self.normal_size, self.normal_size, self.channles),
                                             pooling='max',
                                             classes=self.classes)

        elif self.model_name == 'InceptionV3':
            model = keras.applications.InceptionV3(include_top=True,
                                                weights=None,
                                                input_tensor=None,
                                                input_shape=(self.normal_size, self.normal_size, self.channles),
                                                pooling='max',
                                                classes=self.classes)

        elif self.model_name == 'Xception':
            model = keras.applications.Xception(include_top=True,
                                             weights=None,  # self.checkpoints+self.model_name+'.h5',
                                             input_tensor=None,
                                             input_shape=(self.normal_size, self.normal_size, self.channles),
                                             pooling='max',
                                             classes=self.classes)

        elif self.model_name == 'MobileNet':
            model = keras.applications.MobileNet(include_top=True,
                                             weights=None,  # self.checkpoints+self.model_name+'.h5',
                                             input_tensor=None,
                                             input_shape=(self.normal_size, self.normal_size, self.channles),
                                             pooling='max',
                                             classes=self.classes)
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

        model.load_weights(self.checkpoints+self.model_name+'.h5')
        data_list = list(map(lambda x: cv2.resize(cv2.imread(os.path.join(self.test_data_path,x),int(self.channles/3)),
                                                  (self.normal_size,self.normal_size)),os.listdir(self.test_data_path)     ))
        i,j,tmp = 0,0,[]
        for img in data_list:
            img = np.array([img_to_array(img)],dtype='float')/255.0
            pred = model.predict(img).tolist()[0]
            label = pred.index(max(pred))
            confidence = max(pred)
            print('predict label     is: ',label)
            print('predict confidect is: ',confidence)
            if label != sys.argv[1]:
                print('wrong label :_____________________________________________wrong ', label)
                i+=1
                tmp.append(label)
                i+=1
            else:
                j+=1
        print('error number: ', i, '\ntotal: ', i + j, '\naccuacy is: ', 1.0 - i / (i + j))
        print('error: ', ','.join(list(map(lambda x: str(x), tmp))))
        print('Done')

def main():
    predict = PREDICT(config)
    predict.Predict()

if __name__=='__main__':
    main()