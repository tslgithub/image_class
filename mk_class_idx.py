import os
from config import config as cfg

def mk_class_idx():
    train_class_idx = open('train_class_idx.txt','w')
    test_class_idx = open('test_class_idx.txt','w')
    number = len(os.listdir(cfg.train_data_path))
    for idx,classes in enumerate(os.listdir(cfg.train_data_path)):
        if idx != (number-1):
            train_class_idx.write(classes+'\n')
            test_class_idx.write(classes+'\n')
        else:
            train_class_idx.write(classes)
            test_class_idx.write(classes)

def main():
    mk_class_idx()

if __name__ == '__main__':
    main()