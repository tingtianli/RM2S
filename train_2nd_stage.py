#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:13:48 2017

@author: li
"""

from __future__ import print_function
import sys
sys.path.insert(0,'./models')
sys.path.insert(1,'./custom')
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import cv2
import custom_data_reflection_simulation2 as cd
from torchsample import transforms as tensor_tf
import glob
import torchvision.models as models
import numpy as np
from sklearn.cluster import KMeans
from unet_full_cat3 import UNet as _netG
from skimage.color import rgb2gray
from Wgan_model_arbitrary import _netlocalD

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
parser.add_argument('--train_size', type=int,default=4)
parser.add_argument('--test_size', type=int,default=4)
parser.add_argument('--netG_ini_pkl',default=model_dir+'Net_1st_stage.pkl')    
parser.add_argument('--netG_pkl',default=model_dir+'NetG_2_S.pkl') 
parser.add_argument('--netD_pkl',default=model_dir+'NetD_2_S.pkl')
parser.add_argument('--img_dir',default='./VOCdevkit/VOC2012/JPEGImages/')

opt = parser.parse_args()

torch.cuda.set_device(1)
cudnn.benchmark = True
opt.cuda=True

def list_images2(folder, pattern='*.jpg'):
    filenames = sorted(glob.glob(folder + pattern))
    return filenames


img_list=list_images2(opt.img_dir)
img2_list=list_images2(opt.img_dir)


def sobel_mag(input):       # obtain the sobel gradient magnitude
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
        
    return Variable(mag_tensor).cuda()



def DMAP_generation_BR(disparity): # obtain two thresholds for confidence map
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
        coff1=0.5
        coff2=0.5   
        mask_R=(MAP_B[i,0,:,:]>(TH1-(TH1-TH2)*coff1))*(MAP_B[i,0,:,:]!=0)
        mask_B=(MAP_R[i,0,:,:]<(TH2+(TH1-TH2)*coff2))*(MAP_R[i,0,:,:]!=0)
        B_MAP[i,0,:,:]=mask_B
        R_MAP[i,0,:,:]=mask_R
    return B_MAP.float(),R_MAP.float()

data_transform1 = tensor_tf.Compose([
          tensor_tf.RandomFlip(h=True, v=True, p=0.75),
      ])    
data_transform2 = tensor_tf.Compose([
          tensor_tf.RandomFlip(h=True, v=True, p=0.75),
      ])

affine_transform1=tensor_tf.RandomChoiceRotate([0,90,180,270])
affine_transform2=tensor_tf.RandomChoiceRotate([0,90,180,270])
affine_transform3=tensor_tf.RandomTranslate([20/255.,20/255.])

#data augmentation and reflection image processing for simulation,such as blurring and ghost effects
train_set=cd.CustomDataset(img_list,img2_list,
                           data_transform1=data_transform1,
                           data_transform2=data_transform2,
                           affine_transform1=affine_transform1,
                           affine_transform2=affine_transform2,
                           affine_transform3=affine_transform3,
                           )

trainloader = torch.utils.data.DataLoader(train_set, batch_size=opt.train_size, 
                                          shuffle=True, num_workers=opt.workers)      


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
# Generator and discriminator initialization        
netG_ini = _netG(opt)
netG_ini.cuda()

opt.nc=9
opt.outch=3
netG = _netG(opt)
netG.cuda()

netD = _netlocalD(opt)
netD.apply(weights_init)
netD.cuda()

vgg19 = models.vgg19_bn(pretrained=True).cuda()
subclass=nn.Sequential(*list(vgg19.children())[0])
conv1_1=nn.Sequential(*list(subclass.children())[0:2])
conv1_2=nn.Sequential(*list(subclass.children())[0:4])
conv2_2=nn.Sequential(*list(subclass.children())[0:11])
conv3_2=nn.Sequential(*list(subclass.children())[0:18])
conv4_2=nn.Sequential(*list(subclass.children())[0:31])
conv5_2=nn.Sequential(*list(subclass.children())[0:44])


for param in conv1_1.parameters():
    param.requires_grad = False
for param in conv1_2.parameters():
    param.requires_grad = False
for param in conv2_2.parameters():
    param.requires_grad = False
for param in conv3_2.parameters():
    param.requires_grad = False
for param in conv4_2.parameters():
    param.requires_grad = False
for param in conv5_2.parameters():
    param.requires_grad = False   
    

optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lr*10)
optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lr)
netG_ini.eval()
for epoch in range(0,15000):

    for i, data in enumerate(trainloader, 0):
        netG.train()
        netD.train()
        img_rgb,img2_rgb,BR_img,img_rgb_mag,img2_rgb_mag,_ = data

        batch_size = img_rgb.size(0)

        bg=Variable(img_rgb.cuda())  # ground truth background
        rf=Variable(img2_rgb.cuda()) # ground truth reflection
        BR_img=Variable(BR_img.cuda())   #Input
        bg_mag=Variable(img_rgb_mag.cuda())  # ground truth background gradient magnitude
        rf_mag=Variable(img2_rgb_mag.cuda()) # ground truth reflection gradient magnitude


        BR_img_mag=sobel_mag(BR_img)
        
        netG.zero_grad() 
        
        
        MAP=(BR_img_mag>0.5).float() # edge mask
        #Assume the edge are independent. Assign the edge to the layer with higher gradient magnitudes
        MAP1=MAP*(bg_mag>rf_mag).float() # edges belonging to the background layer
        MAP2=MAP*(bg_mag<rf_mag).float() # edges belonging to the reflection layer
        

        edges1=BR_img*torch.cat((MAP1,MAP1,MAP1),dim=1)
        edges2=BR_img*torch.cat((MAP2,MAP2,MAP2),dim=1)
        
        Input2=torch.cat((BR_img,edges1.detach(),edges2.detach()),dim=1)

        fake = netG(Input2)   # reconsturct the image
        fake=nn.Sigmoid()(fake)   

#########################################################################################
       # train the discriminator
        netD.zero_grad()
        errD_real = netD(bg)
        errD_real = errD_real.mean()
        errD_fake = netD(fake.detach())
        errD_fake = errD_fake.mean()
        D_G_z1 = errD_fake.data.mean()
        errD = errD_fake - errD_real
        errD.backward()
        optimizerD.step()
        
        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)
        
##########################################################################################     
            
        # train the generator

        
        # pixel L1 loss
        errG_l1_bg = torch.abs(fake - bg)
        errG_l1_bg = (fake - bg).abs()
        errG_l1_bg = errG_l1_bg *  1
        errG_l1 = errG_l1_bg.mean()

        c1_fake=conv1_2(fake)
        c2_fake=conv2_2(fake)
        c3_fake=conv3_2(fake)
        c4_fake=conv4_2(fake)
        c5_fake=conv5_2(fake)

        
        c1_bg=conv1_2(bg)
        c2_bg=conv2_2(bg)
        c3_bg=conv3_2(bg)
        c4_bg=conv4_2(bg)
        c5_bg=conv5_2(bg)

        # perceptial feature loss
        errG_vgg_bg = (c1_fake-c1_bg).pow(2).mean()+(c2_fake-c2_bg).pow(2).mean()+(c3_fake-c3_bg).pow(2).mean()\
                        +(c4_fake-c4_bg).pow(2).mean()+(c5_fake-c5_bg).pow(2).mean()
                        
        # adversarial loss
        errG_D = netD(fake.cuda())
        errG_D = -errG_D.mean()

        errG =3*errG_vgg_bg+errG_l1*0.4+0.05*errG_D
        
        errG.backward()

        optimizerG.step()
        

        if i % 100 == 0:

            print('[%d/%d][%d/%d] ED: errG_vgg1: %.4f l_loss:%.4f errG_D:%.4f'
                  % (epoch, opt.niter, i, len(trainloader),
                     errG_vgg_bg.data[0],
                     errG_l1.data[0],errD.data[0]))
            
            torch.save(netG.state_dict(),opt.netG_pkl)
#            torch.save(netD.state_dict(),opt.netD_pkl)



        

    
    
    