import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils
from PIL import Image
import cv2 as cv
import torch as th
from torchvision.transforms import transforms


class ThyroidDataset(torch.utils.data.Dataset):
    def __init__(self, image_path,mask_path,transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.image_path = os.path.expanduser(image_path)
        self.mask_path=os.path.expanduser(mask_path)

        self.transform = transform

        self.test_flag = test_flag    

        self.image_list=[]
        self.mask_list=[]

        for root, dirs, files in os.walk(self.image_path):
            for f in files:
                self.image_list.append(os.path.join(root, f))
        for root, dirs, files in os.walk(self.mask_path):
            for f in files:
                self.mask_list.append(os.path.join(root, f))

    def __getitem__(self, x):
        path=self.image_list[x]
        images=[]
        image=Image.open(self.image_list[x]).convert('L')
        mask  = Image.open(self.mask_list[x]).convert('L')
        

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            # for i in range(4):
            #     images.append(image)
            # images=torch.stack(images)
            # images=torch.squeeze(images)
            torch.set_rng_state(state)
            mask  = self.transform(mask)
        return image,mask,path

    def __len__(self):
        return len(self.image_list)


