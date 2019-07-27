import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
#import unet_parts2 as u
import unet_parts3 as u

class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()    
        self.inc = u.down(opt.nc, opt.nef)    
         # state size: (nef) x 64 x 64
        self.down1 = u.down(opt.nef, opt.nef)   
         # state size: (nef) x 32 x 32
        self.down2 = u.down(opt.nef, opt.nef*2)  
        # state size: (nef*2) x 16 x 16
        self.down3 = u.down(opt.nef*2, opt.nef*4)
        # state size: (nef*4) x 8 x 8
        self.down4 =u.down(opt.nef*4, opt.nef*8)
        # state size: (nef*8) x 4 x 4
        self.down5 = u.fcn_4down(opt.nef*8, opt.nBottleneck)  
        # tate size: (nBottleneck) x 1 x 1
        self.up1 = u.fcn_4up(opt.nBottleneck, opt.ngf * 8)   
        # state size. (ngf*8) x 4 x 4
        self.up2 = u.up(opt.ngf * 8 + opt.nef*8, opt.ngf * 4)        
        # state size. (ngf*4) x 8 x 8
        self.up3 = u.up(opt.ngf * 4 + opt.ngf * 4, opt.ngf * 2)        
        # state size. (ngf*2) x 16 x 16
        self.up4 = u.up(opt.ngf * 2 + opt.nef*2, opt.ngf * 2)
        # state size. (ngf*2) x 32 x 32
        self.up5 = u.up(opt.ngf * 2 + opt.nef , opt.ngf * 1)
        # state size. (ngf*2) x 64 x 64
        self.up6 = u.up(opt.ngf * 1 + opt.nef, opt.ngf)          
        # state size. (ngf) x 128 x 128
        self.outc = u.outconv(opt.nef, opt.outch)
        

        
#        self.up1 = fcn_4up(4000, 512)   #4
#        self.up2 = up(512, 256,cat=False)       #8  
#        self.up3 = up(256, 256,cat=False)        #16
#        self.up4 = up(256, 256,cat=False)             #32
#        self.up5 = up(256, 128,cat=False)           #64
#        self.up6 = up(128, 64,cat=False)           #64
#        self.outc = outconv(64, n_classes)

    def forward(self, x):
#        h=128
#        w=128
        input = x
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.up6(x, x0)
        x = self.outc(x)
        
#        x = input[:,0:3,:,:] - x 
#        x = self.up1(x6)
#        x = self.up2(x, 0)
#        x = self.up3(x, 0)
#        x = self.up4(x, 0)
#        x = self.up5(x, 0)
#        x = self.up6(x, 0)
#        x = self.outc(x)
#        x = input - x 
#        x = input - x 
        return x