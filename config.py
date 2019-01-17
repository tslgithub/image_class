"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,ResNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
"""

"""
/dataset/train/
    cat
        cat.jpg
        cat2.jpg
        ...
        cat100.jpg
    dot
        dog.jpg
        dog2.jpg
        ...
        dog100.jpg
    ...

/dataset/test/
    cat
        cat.jpg
        cat2.jpg
        ...
        cat100.jpg
    dot
        dog.jpg
        dog2.jpg
        ...
        dog100.jpg
    ...
"""

class DefaultConfig():
    """
        # You can choose the following model:
        VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net
        ResNet18,ResNet34,ResNet50,ResNet_101,ResNet_152,DenseNet
        """
    model_name = 'DenseNet'

    train_data_path = './dataset/train/'
    # train_data_path = './dataset/test/'
    test_data_path = './dataset/test/'
    checkpoints = './checkpoints/'
    channles = 1

    if model_name == 'InceptionV3':
        normal_size = 75
    elif model_name == 'Xception':
        normal_size = 71
    elif model_name == 'DenseNet':
        normal_size = 128
    else:
        normal_size = 64

    epochs = 100
    batch_size = 16
    data_augmentation = True
    classes = 3
    lr = 0.1
    default_optimizers = False

    cut = True
    rat = 0.1#if cut,img[slice(h*self.rat,h-h*self.rat),slice(w*self.rat,w-w*self.rat)]

config = DefaultConfig()
