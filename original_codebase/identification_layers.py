# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""


#torch just numpy equivalent for GPU
import torch
from torch import nn

from torch.fft import fft2, fftshift, ifft2, ifftshift


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PI = torch.pi


'''
This is a kernel which is the object used to work out the field 
z away from a plane where it is known
U(x,y,z) = F-1(F(U(x,y,0))H(fx,fy)
This is for a true unaproximated diffraction
angular spectrum method
https://phys.libretexts.org/Bookshelves/Optics/BSc_Optics_(Konijnenberg_Adam_and_Urbach)/06%3A_Scalar_diffraction_optics/6.04%3A_Angular_Spectrum_Method
'''

#phase modulation layer
#z is the propogation after the layer
#x,y is displacement of this layer from info layer
#tilt_x, tilt_y is the angle of tilt of the layer compared to neutral

class LCOS(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(LCOS, self).__init__()
        #just creating a tensor of parameters for the phase modulation layer
        self.M = M
        self.L = L
        self.z = z
        self.lambda0 = lambda0
        
    def forward(self, u1, u2, d, theta):
        #fftshift puts 0 freq component in centre
        u = u1*u2
        u = u*self.tilt(theta)
        U1 = fft2(fftshift(u))
        U2 = U1*self.get_kernel(d)
        return ifftshift(ifft2(U2))
    
    def tilt(self, theta):
        dx = self.L/self.M
        k = 2 * PI / self.lambda0
        # just build grid of positional co-ords
        lx = torch.linspace(-self.L/2, self.L/2-dx, self.M)  
        LX, LY = torch.meshgrid(lx, lx, indexing='ij')
        #assume small angle, this is phase
        #note that for multilayer- correct is phase[n]-phase[n-1]
        T = torch.exp(1j * k * ( theta[0] * LX + theta[1] * LY ) )
        T = T.to(device)
        return T.unsqueeze(0)
    
    def get_kernel(self, d):
        dx = self.L/self.M
        k = 2 * PI / self.lambda0
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/self.L,self.M)  
        #repeats fx as x vector, then as y vector for each to form 2 matrices
        FX, FY = torch.meshgrid(fx, fx, indexing='xy')
        
        #normalised k_z
        A=1 - ((self.lambda0 *FX)**2 + (self.lambda0 *FY)**2)
        #accounting for shift
        B = d[0] * FX + d[1] * FY
        #changing type to complex no.
        A = A+0j
        B = B+0j
        #k_z = sqrt(K**2-k_x**2-k_y**2)
        H_prop = torch.exp(1j * k * ( self.z * torch.sqrt(A)))
        #shift term for offcentre
        H_shift = torch.exp(-1j * 2 * PI * B)
        H = H_prop
        H = H* H_shift
        H = fftshift(H)
        H = H.to(device)
        #unsqueeze adds a dimension of size 1 to the H tensor
        return H.unsqueeze(0)    
  

#free space propogation layer
#used for encoding the information at input
class Info(nn.Module):
    def __init__(self, M, L, lambda0, z):
        super(Info, self).__init__()
        self.M = M
        self.L = L
        self.z = z
        self.lambda0 = lambda0
     
    def forward(self, u1, d, theta):
        u1 = u1*self.tilt(theta)
        U1 = fft2(fftshift(u1))  
        U2 = U1*self.get_kernel(d)
        return ifftshift(ifft2(U2))
    
    
    def tilt(self, theta):
        dx = self.L/self.M
        k = 2 * PI / self.lambda0
        # just build grid of positional co-ords
        lx = torch.linspace(-self.L/2, self.L/2-dx, self.M)  
        LX, LY = torch.meshgrid(lx, lx, indexing='ij')
        #assume small angle, this is phase
        #note that for multilayer- correct is phase[n]-phase[n-1]
        T = torch.exp(1j * k * ( theta[0] * LX + theta[1] * LY ) )
        T = T.to(device)
        return T.unsqueeze(0)
    

    def get_kernel(self, d):
        dx = self.L/self.M
        k = 2 * PI / self.lambda0
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/self.L,self.M)  
        #repeats fx as x vector, then as y vector for each to form 2 matrices
        FX, FY = torch.meshgrid(fx, fx, indexing='xy')
        
        #normalised k_z
        A=1 - ((self.lambda0 *FX)**2 + (self.lambda0 *FY)**2)
        #accounting for shift
        B = d[0] * FX + d[1] * FY
        #changing type to complex no.
        A = A+0j
        B = B+0j
        #k_z = sqrt(K**2-k_x**2-k_y**2)
        H_prop = torch.exp(1j * k * ( self.z * torch.sqrt(A)))
        #shift term for offcentre
        H_shift = torch.exp(-1j * 2 * PI * B)
        H = H_prop
        H=H*H_shift
        H = fftshift(H)
        H = H.to(device)
        #unsqueeze adds a dimension of size 1 to the H tensor
        return H.unsqueeze(0)   
    
    '''   
    def get_gridXY(self, M, L):
        dx = L/M
        # just build grid of positional co-ords
        lx = torch.linspace(-L/2, L/2-dx, M)  
        LX, LY = torch.meshgrid(lx, lx, indexing='ij')
        return LX, LY
    '''

#class sensor(nn.module):
