
'''
nn classes for propogation through an LCOS system with angular and linear missalignmnet
Info takes field and propgates some distance
LCOS modulates with a phase only layer and then propogates

the diffraction is defined using angular spectrum method:
https://phys.libretexts.org/Bookshelves/Optics/BSc_Optics_(Konijnenberg_Adam_and_Urbach)/06%3A_Scalar_diffraction_optics/6.04%3A_Angular_Spectrum_Method

the inputs are wavelength, dimensions which is a list of physical height, physical width, pixel height, pixel width
then propogation distance 

angular missalignment is assumed small angle and leads to multiplication in real space by the response 
exp( j * 2pi / lambda * (theta_x * x + theta_y * y)) where theta_x and theta_y are theangular missalignment

displacement is convolution with delta so we use equivalently multiplication with this response in spacial frequency
domain: exp( j * (d_x * k_x + d_y * k_y)) where d_x, d_y are the displacement missalignments

'''

import torch
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
PI = torch.pi


class LCOS(nn.Module):
    def __init__(self, lambda0, dimensions):
        super(LCOS, self).__init__()
        #just creating a tensor of parameters for the phase modulation layer
        self.H = dimensions[0].item()
        self.W = dimensions[1].item()
        self.h = dimensions[2].item()
        self.w = dimensions[3].item()
        self.lambda0 = lambda0
        
    def forward(self, input, phase_response, tilt):
        u = input * phase_response
        u = u * self.tilt(tilt)
        return u
    
    def tilt(self, theta):
        dx = self.H/self.h
        dy = self.W/self.w
        k = 2 * PI / self.lambda0
        # just build grid of positional co-ords
        lx = torch.linspace(-self.H/2, self.H/2-dx, self.h) 
        ly = torch.linspace(-self.W/2, self.W/2-dx, self.w) 
        LX, LY = torch.meshgrid(lx, ly, indexing='ij')
        #assume small angle, this is phase
        T = torch.exp(1j * k * ( theta[0] * LX + theta[1] * LY ) )
        T = T.to(device)
        return T.unsqueeze(0)
       
class Propogate(nn.Module):
    def __init__(self, lambda0, dimensions, distance):
        super(Propogate, self).__init__()
        self.H = dimensions[0].item()
        self.W = dimensions[1].item()
        self.h = dimensions[2].item()
        self.w = dimensions[3].item()
        self.z = distance
        self.lambda0 = lambda0 
     
    def forward(self, input, displacement=0):
        U1 = fft2(fftshift(input))  
        U2 = U1*self.get_kernel(displacement)
        return ifftshift(ifft2(U2))

    def get_kernel(self, d):
        dx = self.H/self.h
        dy = self.W/self.w
        k = 2 * torch.pi / self.lambda0
        fx = torch.linspace(-1/(2*dx), 1/(2*dx)-1/self.H,self.h)  
        fy = torch.linspace(-1/(2*dy), 1/(2*dy)-1/self.W,self.w) 
        #repeats fx as x vector, then as y vector for each to form 2 matrices
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
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
        H_shift = torch.exp(-1j * 2 * torch.pi * B)
        H = H_prop
        H=H*H_shift
        H = fftshift(H)
        H = H.to(device)
        #unsqueeze adds a dimension of size 1 to the H tensor
        return H.unsqueeze(0) 
  

