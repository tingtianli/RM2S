#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:13:48 2017

@author: li
"""

from __future__ import print_function
import sys
sys.path.insert(0,'./models')
sys.path.insert(0,'./custom')

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import cv2
import custom_real3 as cd
import glob
import numpy as np
from sklearn.cluster import KMeans
from unet_full_cat3 import UNet as net_U
from skimage.color import rgb2gray
from skimage.io import imsave
import os

model_dir='./model_para/'
parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=38, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--outch', type=int, default=3)
parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.995,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')
parser.add_argument('--train_size', type=int,default=10)
parser.add_argument('--test_size', type=int,default=4)
parser.add_argument('--net_ini_pkl',default=model_dir+'Net_1st_stage.pkl')   # the pre-trained model  
parser.add_argument('--netG_img_pkl',default=model_dir+'Net_2nd_stage.pkl') # the pre-trained model
parser.add_argument('--bechmark_dir',default='none')  # please put and the name the SIR2 dataset here

opt = parser.parse_args()


cudnn.benchmark = True
opt.cuda=True


def list_images(folder, pattern='/*'):
    filenames = sorted(glob.glob(folder + pattern))
    return filenames


RB_list=list_images(opt.bechmark_dir,'/Postcard Dataset/*/*/*/*m*.png')+list_images(opt.bechmark_dir,'/SolidObjectDataset/*/*/*/m.jpg')\
                +list_images(opt.bechmark_dir,'/WildSceneDataset/withgt/*/m.jpg')


def sobel_mag(input):   # obtain the sobel gradient magnitude
    ref=(input.data).cpu().numpy()
    mag_tensor=torch.zeros([input.size()[0],1,input.size()[2],input.size()[3]])
    for ind in range(input.size()[0]):
        img=ref[ind,:,:,:]
        img=img.transpose(1,2,0)
        gray=rgb2gray(img)
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
        mag=np.sqrt(sobelx**2+sobely**2)
        mag_tensor[ind,0,:,:]=torch.FloatTensor(mag)
        
    return Variable(mag_tensor).cuda(0)

def DMAP_generation_BR(disparity):  # obtain two thresholds for confidence map
    MAP_B=disparity+0
    MAP_R=disparity+0
    d=(disparity.data).cpu().numpy()
    B_MAP=Variable(torch.zeros((disparity.size()))).cuda()
    R_MAP=Variable(torch.zeros((disparity.size()))).cuda()
    for i in range(0,d.shape[0]):
        ref=d[i,0,:,:]
        ind=np.where(ref!=0)
        km= KMeans(n_clusters=2, random_state=0).fit((ref[ind]).reshape(-1, 1) )             
        TH1=np.amax(km.cluster_centers_)
        TH2=np.amin(km.cluster_centers_)

        coff1=0.2
        coff2=0.8
        mask_R=(MAP_B[i,0,:,:]>(TH1-(TH1-TH2)*coff1))*(MAP_B[i,0,:,:]!=0)
        mask_B=(MAP_R[i,0,:,:]<(TH2+(TH1-TH2)*coff2))*(MAP_R[i,0,:,:]!=0)
        B_MAP[i,0,:,:]=mask_B
        R_MAP[i,0,:,:]=mask_R

    return B_MAP.float(),R_MAP.float()




imgs_data=cd.CustomDataset(RB_list)     



# load the first-stage network        
net_ini = net_U(opt)
net_ini.cuda(0)
net_ini.load_state_dict(torch.load(opt.net_ini_pkl))

# load the second-stage network   
opt.nc=9
opt.outch=3
netG_img = net_U(opt)
netG_img.cuda(0)
netG_img.load_state_dict(torch.load(opt.netG_img_pkl))

net_ini.eval()
netG_img.eval()
aver_psnr=0
try:
    os.mkdir('./benchmark_results/')
except:
    print ("folder exists")
    
for i in range(0, len(imgs_data)):


    BR_img= imgs_data[i] 

    BR_img=Variable(BR_img.cuda(0)).unsqueeze(0)
    BR_cpu=(BR_img.data).cpu().numpy()
    BR_cpu=BR_cpu[0,:,:,:]
    BR_cpu=BR_cpu.transpose(1,2,0)
    Input=BR_img
           
    fake_ini = net_ini(Input)                 #obatin the initial result
    fake_ini=nn.Sigmoid()(fake_ini)
    fake_ini_mag=sobel_mag(fake_ini)
    fake_rf_ini=BR_img-fake_ini
    fake_rf_ini_mag=sobel_mag(fake_rf_ini)
    BR_img_mag=sobel_mag(BR_img)             # obtain the soble edge magnitudes
    
    e=0.0000001
    conf=torch.log(fake_rf_ini_mag/(fake_ini_mag+e)**1+1) # obtain confidence map
    MAP=(BR_img_mag>1).float()
    conf_map=MAP*conf
    DMAP_B,DMAP_R=DMAP_generation_BR(conf_map)  # obtain two thresholds for confidence map

    MAP1=MAP*DMAP_B
    MAP2=MAP*DMAP_R
        
    edges1=BR_img*torch.cat((MAP1,MAP1,MAP1),dim=1)  # obtain two partial edges
    edges2=BR_img*torch.cat((MAP2,MAP2,MAP2),dim=1)        
    Input2=torch.cat((BR_img,edges1,edges2),dim=1)    #obtain the final result
    
    fake_bg = netG_img(Input2)
    fake_bg=nn.Sigmoid()(fake_bg)     
########

#    
    fake_ini_cpu=fake_ini.data.cpu().numpy()[0,:,:,:]*255
    fake_bg_cpu=fake_bg.data.cpu().numpy()[0,:,:,:]*255
    BR_img_cpu=BR_img.data.cpu().numpy()[0,:,:,:]*255
    fake_ini_cpu=fake_ini_cpu.transpose(1,2,0)
    fake_bg_cpu=fake_bg_cpu.transpose(1,2,0)
    BR_img_cpu=BR_img_cpu.transpose(1,2,0)
    imsave('./benchmark_results/result_'+str(i)+'.png', fake_bg_cpu/255.)

