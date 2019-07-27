#!/usr/bin/python

# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True)
#            nn.LeakyReLU(0.2, inplace=True)
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
#            nn.MaxPool2d(2),
            nn.Conv2d(in_ch, in_ch, 4,2,1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch,5,1,2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 5,1,2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False,cat=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
#            if cat is True:
#                self.up = nn.ConvTranspose2d(in_ch/2, in_ch/2, 4, 2, 1, bias=False)
#            else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1, bias=False)
            

        self.conv = nn.Sequential(
        ###############################3
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
          ###############################3     
            nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cat=cat

    def forward(self, x1, x2=None):

        if self.cat is True:
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
            x = torch.cat([x2, x1], dim=1)
        else:
            x=x1             
        x = self.up(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
#            nn.BatchNorm2d(in_ch),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Conv2d(in_ch, in_ch, 5, padding=2, bias=False),
#            nn.BatchNorm2d(in_ch),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class fcn_4down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fcn_4down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,4, bias=False),
            # tate size: (nBottleneck) x 1 x 1l
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class fcn_4up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fcn_4up, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
         # state size. (ngf*8) x 4 x 4
        return x    
    
    
class fcn_2down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fcn_2down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,2, bias=False),
            # tate size: (nBottleneck) x 1 x 1l
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class fcn_2up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fcn_2up, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
         # state size. (ngf*8) x 4 x 4
        return x    