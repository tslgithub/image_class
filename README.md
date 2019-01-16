# image classification with deep learning model： VGG16、VGG19、InceptionV3、Xception、MobileNet、AlexNet、LeNet、ZF_Net、ResNet18、ResNet34、ResNet50、ResNet_101、ResNet_152

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
* ResNet_101
* ResNet_152



## your train or test datasets folder should be:

### /dataset/train/
* __1__ 
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __2__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __3__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __4__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg

### /dataset/test/
* __1__ 
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __2__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __3__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg
* __4__
cat.jpg,
cat2.jpg,
cat3.jpg,
cat4.jpg,
cat5.jpg,
cat100.jpg
cat1000.jpg

#### 1,2,3,4 is classes name or folder name,whose __path is__
__"training data set folder is:"__

    /dataset/train/1/cat*.jpg,

    /dataset/train/2/cat*.jpg,

    /dataset/train/3/cat*.jpg,

    /dataset/train/4/cat*.jpg,

__"testing data set folder is:"__

    /dataset/test/1/cat*.jpg,

    /dataset/test/2/cat*.jpg,

    /dataset/test/3/cat*.jpg,

    /dataset/test/4/cat*.jpg,

* Attention: classes name ‘1,2,3,4’ or folder name must be number, __not string__

## environment 
 my environment  is based on __ubuntu16、cuda8、tensorflow_gpu1.4__, all package needed can be installed with __'pip3 install package_name'__, and you can test which package is missed by run __'python train.py'__,then pip install the missed package

 # train and predict your model
__train model: python train.py__

__predict model: python predict__

### Any Questions???
Author email: mymailwith163@163.com
