"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,esNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
"""
from __future__ import print_function
from config import config
import sys
import cv2
import os
from keras.preprocessing.image import img_to_array
import numpy as np

import tensorflow as tf
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
tf.Session(config=config1)

from Build_model import Build_model

class PREDICT(Build_model):
    def __init__(self,config):
        Build_model.__init__(self,config)
        self.test_data_path = config.test_data_path+sys.argv[1]

    def Predict(self):
        model = Build_model(self.config).build_model()
        model.load_weights(self.checkpoints+'/'+self.model_name+'/'+self.model_name+'.h5')
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
         
            else:
                j+=1
        print('error number: ', i, '\ntotal: ', i + j, '\naccuacy is: ', 1.0 - i / (len(data_list))  )
        print('error: ', ','.join(list(map(lambda x: str(x), tmp))))
        print('Done')

def main():
    predict = PREDICT(config)
    predict.Predict()

if __name__=='__main__':
    main()
