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
* DenseNet


## your train or test datasets folder should be:

### /dataset/train/
* __dataset/train/0/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __dataset/train/1/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __dataset/train/2/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __dataset/train/3/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg

### /dataset/test/
* __dataset/test/0/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __dataset/test/1/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __dataset/test/2/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __dataset/test/3/__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg

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

* Attention: classes name ‘0,1,2,3’ or folder name must be number, __not string__,and __must start with '0'__

## environment
my environment is based on __ubuntu16、cuda8、tensorflow_gpu1.4__, all package needed can be installed with __'pip3 install package_name'__, and you can test which package is missed by run __'python train.py'__,then pip install the missed package

# train and predict your model
__train model: python train.py  model_name__

__predict model: python predict model_name classes_name_0_or_1_or_2_or_3__

### Any Questions???
Author email: mymailwith163@163.com
