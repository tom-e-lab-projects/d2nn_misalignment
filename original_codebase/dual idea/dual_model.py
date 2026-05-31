# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:05:41 2022

@author: Chen Chunyuan
"""
import torch
from torch import nn
#from OpticalLayers import Information, LCOS
from dual_layers import LCOS, Info

#need to change to put parameters in the layers

#https://www.codegenes.net/blog/netmodule-pytorch/ for understanding nn.module
class Onn(nn.Module):
    def __init__(self, M, L, lambda0, params, z, d=torch.zeros(2,2), tilt=torch.zeros(2,2)):
        super(Onn, self).__init__()
        #define starting params
        self.parameter = nn.Parameter(params)
        self.Info_forward = Info(M, L, lambda0, z[0], d[0,:], tilt[0,:])
        self.LCOS_forward = LCOS(M, L, lambda0, self.parameter, z[1], d[1,:], tilt[1,:] )

        self.Info_backward = Info(M, L, lambda0, z[0], d[0,:], tilt[0,:])
        self.LCOS_backward = LCOS(M, L, lambda0, self.parameter, z[1], d[1,:], tilt[1,:] )

    def forward(self, u1):       
        u = self.Info_forward(u1)
        u = self.LCOS_forward(u)
        return u

    def back(self, u1):       
        u = self.Info_backward(u1)
        u = self.LCOS_backward(u)
        return u
    

    
