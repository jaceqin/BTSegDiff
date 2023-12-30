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
    def __init__(self, data_dir , transform = None):


        super().__init__()
        # self.image_path = os.path.expanduser(image_path)
        # self.mask_path=os.path.expanduser(mask_path)

        self.transform = transform
        self.dataset_root="optic_disc_seg/"
        self.file_list = list()
        # self.image_list=[]
        # self.mask_list=[]
        # for root, dirs, files in os.walk(self.image_path):
        #     for f in files:
        #         self.image_list.append(os.path.join(root, f))
        # for root, dirs, files in os.walk(self.mask_path):
        #     for f in files:
        #         self.mask_list.append(os.path.join(root, f))

        with open(data_dir, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = None
                else:
                    image_path = os.path.join(self.dataset_root, items[0])
                    grt_path = os.path.join(self.dataset_root, items[1])
                self.file_list.append([image_path, grt_path])

    def __getitem__(self, x):
       
        # path=self.image_list[x]
        # image = Image.open(self.image_list[x]).convert('RGB')
        # mask  = Image.open(self.mask_list[x]).convert('L')
        path=self.file_list[x][0]
        image=Image.open(self.file_list[x][0]).convert('RGB')
        mask=Image.open(self.file_list[x][1]).convert('L')
        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask  = self.transform(mask)
            mask=torch.where(mask > 0.1 ,1 ,0)
        return image,mask,path

    def __len__(self):
        return len(self.file_list)
