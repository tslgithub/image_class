# 图像分类集成以下模型：ResNet18、ResNet34、ResNet50、ResNet101、ResNet152、VGG16、VGG19、InceptionV3、Xception、MobileNet、AlexNet、LeNet、ZF_Net、DenseNet，在config.py里面选择使用哪种模型，目前本人自己测试得结论：残差网络resnet的效果比较好。

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


##train or test dataset


####  classes name contained in folder name
__"training or testing dataset folder is:"__

/path/classes1/cat*.jpg,

/path/classes2/dog*.jpg,

/path/classes3/people*.jpg,

/path/classes4/*.jpg,


* Attentions ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
* classes name must be contained in folder name 

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

# train or test  dataset prepare
* python3 mk_class_idx.py

# train your model
* train model: python3 train.py modelName
* or run " __sh trainAll.sh__ " to train all model

# predict your model
* predict model: python3 predict.py model_name classes_name

### Any Questions???
Author email: mymailwith163@163.com
