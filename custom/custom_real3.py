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
from skimage import io,transform






class CustomDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self,RB_list,bg_list):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.RB_list = RB_list
        self.bg_list = bg_list


    def __len__(self):
        return len(self.bg_list)

    def __getitem__(self, idx):
        
        
        idd1=idx
#        
        
        RB_rgb=io.imread(self.RB_list[idd1])/255.
        RB_rgb=transform.resize(RB_rgb,(256,256))
        RB_rgb=RB_rgb.transpose(2,0,1)
        RB_rgb = torch.FloatTensor(RB_rgb)

        bg_rgb=io.imread(self.bg_list[idd1])/255.
        bg_rgb=transform.resize(bg_rgb,(256,256))
        bg_rgb=RB_rgb.transpose(2,0,1)
        bg_rgb = torch.FloatTensor(bg_rgb)

        return RB_rgb, bg_rgb
    
