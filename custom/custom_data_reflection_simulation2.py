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
import scipy.stats as st
import random
from skimage.color import rgb2gray


def sobel_mag(input):
    gray=rgb2gray(input)
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    mag=np.sqrt(sobelx**2+sobely**2)

    return mag

def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel/kernel.max()
    return kernel




def syn_data(t,r,sigma,h):
    # create a vignetting mask
    g_mask=gkern(h,1)
    g_mask=np.dstack((g_mask,g_mask,g_mask))
    t=np.power(t,2.2)
    r=np.power(r,2.2)
    
    sz=int(2*np.ceil(2*sigma)+1)
    r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
    blend=r_blur+t
    
    att=1.08+np.random.random()/10.0
    
    for i in range(3):
        maski=blend[:,:,i]>1
        mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
        r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
    r_blur[r_blur>=1]=1
    r_blur[r_blur<=0]=0

    h,w=r_blur.shape[0:2]
    alpha2 = 1-np.random.random()/3.0;
    alpha1= np.random.random()*0.2*g_mask #usde
    r_blur_mask=np.multiply(r_blur,alpha1)
    t=np.power(t,1/2.2)*alpha2
    r_blur_mask=np.power(r_blur_mask,1/2.2)


    return t,r_blur_mask


def data_parepare(img_rgb,img2_rgb,img_rgb_mag,img2_rgb_mag,
                  data_transform1,affine_transform1,data_transform2,affine_transform2,affine_transform3):

    img_rgb_cat=torch.cat((img_rgb,img_rgb_mag),dim=0)
    img2_rgb_cat=torch.cat((img2_rgb,img2_rgb_mag),dim=0)
    
    if affine_transform1 is not None:
        img_rgb_cat=affine_transform1(img_rgb_cat)

    if data_transform1 is not None:
        img_rgb_cat=data_transform1(img_rgb_cat)  
        
    if affine_transform2 is not None:
        img2_rgb_cat=affine_transform2(img2_rgb_cat)

    if data_transform2 is not None:
        img2_rgb_cat=data_transform2(img2_rgb_cat)  

    if affine_transform3 is not None:
        img2_rgb_shadow=affine_transform3(img2_rgb)
        img2_rgb=img2_rgb+0*img2_rgb_shadow
        
        
    img_rgb=img_rgb_cat[0:3,:,:]
    img_rgb_mag=(img_rgb_cat[3,:,:]).unsqueeze(0)
    
    img2_rgb=img2_rgb_cat[0:3,:,:]
    img2_rgb_mag=img2_rgb_cat[3,:,:].unsqueeze(0)
    
    return img_rgb,img2_rgb,img_rgb_mag,img2_rgb_mag





class CustomDataset(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_list,img2_list,
                 data_transform1=None,affine_transform1=None,data_transform2=None,affine_transform2=None,affine_transform3=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_transform1 = data_transform1
        self.affine_transform1 = affine_transform1
        self.data_transform2 = data_transform2
        self.affine_transform2 = affine_transform2
        self.affine_transform3 = affine_transform3
        self.img_list=img_list
        self.img2_list=img2_list


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        
        idd1=idx
        idd2=random.randrange(0,len(self.img2_list))

        cof1=random.random()*0.3+0.67 ##used
        cof2=1-cof1    
        
        
        img_rgb=io.imread(self.img_list[idd1])/255.
        img_rgb=transform.resize(img_rgb,(256,256))

        img2_rgb=io.imread(self.img2_list[idd2])/255.
        img2_rgb_ori=transform.resize(img2_rgb,(256,256))
        
        sigma=random.random()*3

        img_rgb,img2_rgb=syn_data(img_rgb,img2_rgb_ori,sigma,256)
        
        

        img_rgb_mag=sobel_mag(img_rgb)
        img2_rgb_mag=sobel_mag(img2_rgb)
       

        img_rgb=img_rgb.transpose(2,0,1)
        img2_rgb=img2_rgb.transpose(2,0,1)
        img2_rgb_ori=img2_rgb_ori.transpose(2,0,1)
        
        
        img_rgb_mag=torch.FloatTensor(img_rgb_mag)
        img_rgb_mag=img_rgb_mag.unsqueeze(0)
        
        img2_rgb_mag=torch.FloatTensor(img2_rgb_mag)
        img2_rgb_mag=img2_rgb_mag.unsqueeze(0)
        
        img_rgb = torch.FloatTensor(img_rgb)
        img2_rgb = torch.FloatTensor(img2_rgb) 
        img2_rgb_ori = torch.FloatTensor(img2_rgb_ori) 




        img_rgb,img2_rgb,img_rgb_mag,img2_rgb_mag = data_parepare(img_rgb,img2_rgb,img_rgb_mag,img2_rgb_mag,
                                                    self.data_transform1,self.affine_transform1,
                                                     self.data_transform2,self.affine_transform2,self.affine_transform3)

        RB_rgb=img_rgb+img2_rgb
        RB_rgb[RB_rgb>=1]=1
        RB_rgb[RB_rgb<=0]=0
        
        RB_rgb_cpu=(RB_rgb).numpy()
        RB_rgb_cpu=RB_rgb_cpu.transpose(1,2,0)
        RB_mag=sobel_mag(RB_rgb_cpu)
        RB_mag=torch.FloatTensor(RB_mag)
        RB_mag=RB_mag.unsqueeze(0)        
        
        

        return img_rgb,img2_rgb,RB_rgb,img_rgb_mag,img2_rgb_mag,RB_mag

    
