#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:33:30 2017

@author: li
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:12:40 2017

@author: li
"""
import torch
import torch.utils.data as Data
from torchsample import transforms as tensor_tf
import glob
from skimage import io,color,transform,filters
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.signal import medfilt

import random



def data_parepare(img_rgb,img2_rgb,data_transform1,affine_transform1,data_transform2,affine_transform2):
    w,h=256,256

    
    if affine_transform1 is not None:
        img_rgb=affine_transform1(img_rgb)
    if data_transform1 is not None:
        img_rgb=data_transform1(img_rgb)  
    if affine_transform2 is not None:
        img2_rgb=affine_transform2(img2_rgb)
    if data_transform2 is not None:
        img2_rgb=data_transform2(img2_rgb)  
        
    
    return img_rgb,img2_rgb





class CustomDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, bg_list,RB_list):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bg_list = bg_list
#        self.rf_list = rf_list
        self.RB_list = RB_list


    def __len__(self):
        return len(self.bg_list)

    def __getitem__(self, idx):
        
        
        idd1=idx

        bg_rgb=io.imread(self.bg_list[idd1])/255.
        bg_rgb=transform.resize(bg_rgb,(256,256))
        
        
#        rf_rgb=io.imread(self.rf_list[idd1])/255.
#        rf_rgb=transform.resize(rf_rgb,(256,256))
#        for k in range(0,3):
#            img_rgb[:,:,k]=img_rgb[:,:,k]-np.mean(img_rgb[:,:,k])
#        
        
        RB_rgb=io.imread(self.RB_list[idd1])/255.
        RB_rgb=transform.resize(RB_rgb,(256,256))

        

        RB_rgb=RB_rgb.transpose(2,0,1)
        bg_rgb=bg_rgb.transpose(2,0,1)
#        rf_rgb=rf_rgb.transpose(2,0,1)
        
        bg_rgb = torch.FloatTensor(bg_rgb)
        RB_rgb = torch.FloatTensor(RB_rgb)
#        rf_rgb = torch.FloatTensor(rf_rgb)


        return bg_rgb, RB_rgb
    
