"""
author:tslgithub
email:mymailwith163@163.com
time:2018-12-12
msg: You can choose the following model to train your image, and just switch in config.py:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,esNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
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
    train_data_path = './dataset/train/'
    test_data_path = './dataset/test/'
    channles = 1
    checkpoints = './checkpoints/'
    normal_size = 64
    epochs = 100
    batch_size = 2
    data_augmentation = True
    classes = 2
    lr = 0.0001
    default_optimizers = False
    data_augmentation = True
    """
    # You can choose the following model:
    VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net
    ResNet18,ResNet34,ResNet50,ResNet_101,ResNet_152
    """
    model_name = 'VGG16'

config = DefaultConfig()
