import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class OPTDatset(Dataset):
    def __init__(self, image_path, mask_path , transform = None):
        """
        OpticDiscSeg dataset is extraced from iChallenge-AMD
        (https://ai.baidu.com/broad/subordinate?dataset=amd).

        Args:
            transforms (list): Transforms for image.
            dataset_root (str): The dataset directory. Default: None
            mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
            edge (bool, optional): Whether to compute edge while training. Default: False
        """


        def __init__(self,
                    dataset_root=None,
                    transforms=None,
                    mode='train',
                    edge=False):
            self.dataset_root = dataset_root
            self.transforms = Compose(transforms)
            mode = mode.lower()
            self.mode = mode
            self.file_list = list()
            self.num_classes = self.NUM_CLASSES
            self.ignore_index = self.IGNORE_INDEX
            self.edge = edge

            if mode == 'train':
                file_path = os.path.join(self.dataset_root, 'train_list.txt')
            elif mode == 'val':
                file_path = os.path.join(self.dataset_root, 'val_list.txt')
            else:
                file_path = os.path.join(self.dataset_root, 'test_list.txt')

            with open(file_path, 'r') as f:
                for line in f:
                    items = line.strip().split()
                    if len(items) != 2:
                        if mode == 'train' or mode == 'val':
                            raise Exception(
                                "File list format incorrect! It should be"
                                " image_name label_name\\n")
                        image_path = os.path.join(self.dataset_root, items[0])
                        grt_path = None
                    else:
                        image_path = os.path.join(self.dataset_root, items[0])
                        grt_path = os.path.join(self.dataset_root, items[1])
                    self.file_list.append([image_path, grt_path])