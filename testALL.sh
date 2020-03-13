#!/usr/bin/env bash

modelName="VGG16,VGG19,InceptionV3,Xception,MobileNet,AlexNet,LeNet,ZF_Net,ResNet18,ResNet34,ResNet50,ResNet101,ResNet152"
className="person,down"

model=(${modelName//,/ })
clses=(${className//,/ })

for mdl in ${model[@]}
do
    for cls in ${clses[@]}
    do
        echo python predict.py $mdl $cls
        python predict.py $mdl $cls
    done

done