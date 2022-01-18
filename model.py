from typing import Optional
import torch.nn as nn
from torch import Tensor

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class netG(nn.Module):
    def __init__(self,input_dim,out_dim=64,is_bn:Optional[None]=False,activ_func=nn.ReLU) -> None:
        super().__init__()
        if is_bn:
            self.stage1 = nn.Sequential(
                nn.Linear(input_dim,out_dim*8*4*4,bias=False),
                nn.BatchNorm1d(out_dim*8*4*4),
                activ_func(inplace=True)
            )
        elif not is_bn:
            self.stage1 = nn.Sequential(
                nn.Linear(input_dim,out_dim),
                activ_func(inplace=True)
            )
        self.stage2 = nn.Sequential(
            deconv(8*out_dim,4*out_dim,True,nn.LeakyReLU),
            deconv(4*out_dim,2*out_dim,True,nn.LeakyReLU),
            deconv(2*out_dim,out_dim,True,nn.LeakyReLU),
            nn.ConvTranspose2d(out_dim,3,5,2,2,1),
            nn.Tanh()
        )

    def forward(self,x:Tensor):
        x = self.stage1(x)
        x = x.view(x.size(0),-1,4,4)
        return self.stage2(x)
    
class deconv(nn.Module):
    def __init__(self,in_c,out_c,is_bn:Optional[None]=False,activ_func=nn.ReLU) -> None:
        super().__init__()
        if is_bn:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_c,out_c,5,2,2,1,bias=False),
                nn.BatchNorm2d(out_c),
                activ_func(inplace=True)
            )
        elif not is_bn:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_c,out_c,5,2,2,1,bias=False),
                activ_func(inplace=True)
            )
    
    def forward(self,x):
        return self.deconv(x)

class netD(nn.Module):
    def __init__(self,in_c,dim=64,is_bn:Optional[None]=True,activ_func=nn.LeakyReLU) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            conv(in_c,dim,False),
            conv(dim,dim*2),
            conv(dim*2,dim*4),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            conv(dim*4,dim*8),
            nn.Conv2d(dim*8,1,4),
            nn.Sigmoid(),
        )
    
    def forward(self,x:Tensor):
        x = self.stage1(x)
        x = x.view(-1)
        return x

class conv(nn.Module):
    def __init__(self,in_c,out_c,is_bn:Optional[None]=True,activ_func=nn.LeakyReLU) -> None:
        super().__init__()
        self.f=nn.Sequential(
            nn.Conv2d(in_c,out_c,5,2,2),
            nn.BatchNorm2d(out_c),
            activ_func(inplace=True)
        )
    
    def forward(self,x):
        return self.f(x)
