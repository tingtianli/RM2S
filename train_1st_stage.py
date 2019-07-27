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
import custom_data_reflection_simulation as cd
from torchsample import transforms as tensor_tf
import glob
import torchvision.models as models
from unet_full_cat3 import UNet as Net_


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
parser.add_argument('--Net', default='', help="path to Net (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.995,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')
parser.add_argument('--train_size', type=int,default=6)
parser.add_argument('--test_size', type=int,default=4)
parser.add_argument('--Net_pkl',default=model_dir+'Net_1_S.pkl') 
parser.add_argument('--img_dir',default='./VOCdevkit/VOC2012/JPEGImages/')

opt = parser.parse_args()

cudnn.benchmark = True
opt.cuda=True

torch.cuda.set_device(1)


def list_images2(folder, pattern='*.jpg'):
    filenames = sorted(glob.glob(folder + pattern))
    return filenames

img_list=list_images2(opt.img_dir)  # image1 for simulation
img2_list=list_images2(opt.img_dir) # image2 for simulation

# data augmentation
data_transform1 = tensor_tf.Compose([
          tensor_tf.RandomFlip(h=True, v=True, p=0.75),
      ])
    
data_transform2 = tensor_tf.Compose([
          tensor_tf.RandomFlip(h=True, v=True, p=0.75),
      ])

affine_transform1=tensor_tf.RandomChoiceRotate([0,90,180,270])
affine_transform2=tensor_tf.RandomChoiceRotate([0,90,180,270])
affine_transform3=tensor_tf.RandomTranslate([15/256.,15/256.])

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
# network initialization
Net = Net_(opt)
Net.apply(weights_init)
Net.cuda()

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
    

optimizerG = optim.RMSprop(Net.parameters(), lr = opt.lr*10)

for epoch in range(0,15000):


    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        Net.train()
        img_rgb,img2_rgb,BR_img,_ = data # image1 for simulation, image2 for simulation, overlapped image

        batch_size = img_rgb.size(0)

        bg=Variable(img_rgb.cuda())
        rf=Variable(img2_rgb.cuda())
        BR_img=Variable(BR_img.cuda())

        Input=BR_img        
        fake = Net(Input)

        fake=nn.Sigmoid()(fake)
        Net.zero_grad() 

        # pixel L1 loss
        errG_l1_bg = torch.abs(fake - bg)
        errG_l1_bg = (fake - bg).abs()
        errG_l1_bg = errG_l1_bg *  1
        errG_l1_bg = errG_l1_bg.mean()
        errG_l1=errG_l1_bg
        
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
        
        # perceptual featrue loss
        errG_vgg1 = (c1_fake-c1_bg).pow(2).mean()+(c2_fake-c2_bg).pow(2).mean()+(c3_fake-c3_bg).pow(2).mean()\
                        +(c4_fake-c4_bg).pow(2).mean()+(c5_fake-c5_bg).pow(2).mean()
               
        # Feature reduction term L_FR
        vggf1=c1_fake.abs().mean()+c2_fake.abs().mean()+c3_fake.abs().mean()
        errG =3*errG_vgg1+errG_l1*0.4+3*vggf1
        errG.backward()
        optimizerG.step()
        
        if i % 100 == 0:
            torch.save(Net.state_dict(),opt.Net_pkl)

            print('[%d/%d][%d/%d] ED: errG_vgg1: %.4f l_loss:%.4f vggf1:%.4f'
                  % (epoch, opt.niter, i, len(trainloader),
                     errG_vgg1.data[0],
                     errG_l1.data[0],vggf1.data[0]))



    
    
    