# image classification with deep learning model： VGG16、VGG19、InceptionV3、Xception、MobileNet、AlexNet、LeNet、ZF_Net、ResNet18、ResNet34、ResNet50、ResNet101、ResNet152、DenseNet

## the project apply the following models:


* VGG16
* VGG19
* InceptionV3
* Xception
* MobileNet
* AlexNet
* LeNet
* ZF_Net
* ResNet18
* ResNet34
* ResNet50
* ResNet101
* ResNet152
* DenseNet(dismissed this time)


## your train or test datasets folder should be:


#### 0,1,2,3 is classes name or folder name,whose __path is__,and must start with '0'
__"training data set folder is:"__

/dataset/train/0/cat*.jpg,

/dataset/train/1/cat*.jpg,

/dataset/train/2/cat*.jpg,

/dataset/train/3/cat*.jpg,

__"testing data set folder is:"__

/dataset/test/0/cat*.jpg,

/dataset/test/1/cat*.jpg,

/dataset/test/2/cat*.jpg,

/dataset/test/3/cat*.jpg,

* Attentions ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
* classes name ‘0,1,2,3’ or folder name must be number,and __must start with '0'__


## environment
My environment is based on 
* __ubuntu16__ 
* __cuda8__ (__cuda9.0__)
* __tensorflow_gpu1.4__ (__tensorflow_gpu1.10__ )
* __keras2.0.8__
* __numpy__
* __tqdm__
* __opencv-python__
* __scikit-learn__
### Install packages
* pip3 install tensorflow_gpu==1.4
* pip3 install keras==2.0.8
* pip3 install numpy
* pip3 install tqdm
* pip3 install opencv-python
* pip3 install scikit-learn

# train and predict your model
* train model: python train.py  model_name

* predict model: python predict.py model_name classes_name

* (Atttention:classes_name should be 0,1,2,3,........)

### Any Questions???
Author email: mymailwith163@163.com
