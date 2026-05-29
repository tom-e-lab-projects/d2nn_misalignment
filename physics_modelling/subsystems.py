'''
nn module classes for free space optical propogation between two layers
and propogation accoutning for the fact that fft
has cyclic effects

the configs are for storing standard propagation variables
'''

import torch
from torch import nn
from torch.fft import fft2, fftshift, ifft2, ifftshift
from dataclasses import dataclass

@dataclass
class OpticalConfig:
    '''
    lambda0 is wavelength
    assume square pixels
    height and width are then in terms of number of pixels
    width for the assumed Gaussian beam
    padding is for zero-padding
    '''
    lambda0: float
    pixel_size: float
    height: int
    width: int
    beam_waist: float

    def padded(self, factor=2):
        return OpticalConfig(
            self.lambda0,
            self.pixel_size,
            self.height * factor,
            self.width * factor,
            self.beam_waist
        ) 
    
################## helper fns for building tensors which are spacially varying ###################

def k_space(pixel_size, l):
    #the values of the k space in a tensor
    #return torch.linspace(-1/(2*pixel_size), 1/(2*pixel_size)-1/(pixel_size * l),l)
    return torch.fft.fftfreq(l,pixel_size)

def real_space(pixel_size, l):
    # real version
    L = l * pixel_size
    return torch.linspace(-L/2, L/2-pixel_size, l) 

def grid(pixel_size, h, w, real):
    #grid for tensors which are functions of x,y
    #real si bool for real or imaginary
    if real == True:
        space = real_space
    else:
        space = k_space
    lx = space(pixel_size, w)
    ly = space(pixel_size, h)
    LX, LY = torch.meshgrid(lx, ly, indexing='xy')
    return LX.unsqueeze(0), LY.unsqueeze(0)

def reshape(tensor, n):
    # 1D tensor
    return tensor.view(n,1,1)

#for splitting a misalignment tensor of shape n,2 into 2 of n,1,1
def split(m):
    n = m.size(0)
    return m[:,0].view(n,1,1), m[:,1].view(n,1,1)


####################### propagation classes ###########################

class Propagate(nn.Module):
    '''
    propagating without zero padding 
    tensor of shape num_samples, num_systems, h, w
    contains addable tilt, displacement offset and axial offset
    the tilt and displacement are shape n,2 with the n being
    the number of different misaligned systems to apply
    and are in radians and metres respectively
    can be applied as intialisation parameters, or training or neither
    if initialisation, applied to dim 1
    if training, applied to dim 2

    distance is in m
    propagation is done by angular spectrum method
    https://phys.libretexts.org/Bookshelves/Optics/BSc_Optics_(Konijnenberg_Adam_and_Urbach)/06%3A_Scalar_diffraction_optics/6.04%3A_Angular_Spectrum_Method

    
    '''

    def __init__(self, 
                 config , 
                 distance, 
                 tilt_displacement=None, 
                 lateral_displacement=None
                 ):
        
        super().__init__()
        self.l = config.lambda0 
        self.h = config.height
        self.w = config.width
        self.p = config.pixel_size
        self.z = distance

        self.LX, self.LY = grid(self.p, self.h,self.w, True)
        self.FX, self.FY = grid(self.p, self.h, self.w, False)

        # register buffer mean the tensors move to GPU with the model
        if tilt_displacement is not None:
            tilted =self.tilt( tilt_displacement ).unsqueeze(0)
        else:
            tilted = torch.tensor(1.0)

        self.register_buffer("tilted",tilted)

        kernel = self.get_kernel()

        if lateral_displacement != None:
            dis = self.xy_displacement( lateral_displacement )
            kernel = kernel * dis

        self.register_buffer("kernel",kernel.unsqueeze(0))

    def forward(self, 
                input, 
                tilt_displacement=None, 
                lateral_displacement=None
                ):
        
        u = input
        u = self.tilted*input

        if tilt_displacement is not None:
            u = self.tilt( tilt_displacement ).unsqueeze(1)*u

        u = fft2(u)
        u = u*self.kernel

        if lateral_displacement is not None:
            u = u * self.xy_displacement( lateral_displacement ).unsqueeze(1)

        return ifft2(u)
        
    def get_kernel(self):
        '''
        angular spectrum propagation kernel
        '''
        k = 2 * torch.pi / self.l

        A=1 - ((self.l *self.FX)**2 + (self.l *self.FY)**2)

        n = self.z.size(0)
        z = self.z.view(n,1,1)

        H = torch.exp(1j * k * ( z * torch.sqrt(A) ))                  
        return H
    
    def xy_displacement(self, lateral_displacement):
        '''
        lateral displacement can be treated as phase ramp in k space
        '''
        d_0, d_1 = split( lateral_displacement )
        B = d_0 * self.FX + d_1 * self.FY
        H_shift = torch.exp(1j * 2 * torch.pi * B)
        return H_shift

    def tilt(self, tilt_displacement):
        '''
        small angle tilt is phase ramp on input
        '''
        k = 2 * torch.pi / self.l
        #taking angles to shapes n,1,1
        theta_0, theta_1 = split( tilt_displacement )
        T = torch.exp(1j * k * ( theta_0 * self.LX + theta_1 * self.LY ) )
        return T
    


class PropagatePadded(nn.Module):
    '''
    adding zero padding by factor of 2 to prevent cyclic effects of fft
    unpads after propagating
    '''
    def __init__(self, config, distance, tilt_displacement = None ,lateral_displacement = None):
        super().__init__()
        self.h = config.height
        self.w = config.width
        self.z = distance

        self.h__2 = self.h*2
        self.w__2 = self.w*2
        self.h_2 = self.h//2
        self.w_2 = self.w//2
        self.h3_2 = self.h + self.h_2
        self.w3_2 = self.w + self.w_2

        self.Propagate = Propagate(config.padded(), 
                                    self.z, 
                                    tilt_displacement,
                                    lateral_displacement
                                    )  

    def forward(self,
                 input, 
                 tilt_displacement=None, 
                 lateral_displacement=None 
                 ):

        pad_in = self.pad_input(input)

        pad_out = self.Propagate(pad_in, 
                                 tilt_displacement,
                                 lateral_displacement
                                 )

        out = pad_out[:,:,
                      self.h_2 : self.h3_2,
                      self.w_2 : self.w3_2
                     ]
        return out

    def pad_input(self, input):
        pad_in = input.new_zeros(input.size(0), input.size(1), self.h__2, self.w__2)
        pad_in[:,:,
               self.h_2:self.h3_2,
               self.w_2:self.w3_2
              ] = input
        return pad_in


class GaussianIntensity(nn.Module):
    '''
    creating a gaussian intensity beam to illuminate width
    the assumption is that the width of the beam is reasonably well known
    however where the centre will fall on the input modulator is not
    '''
    def __init__(self,
                 config, 
                 centre=None
                 ):
        super().__init__()
        self.h = config.height
        self.w = config.width
        self.p = config.pixel_size
        self.r = config.beam_waist
        self.c = centre

        light = self.profile()
        self.register_buffer("light",light)

    def forward(self,input):
        return input*self.light

    def profile(self):
        LX, LY = grid(self.p, self.h, self.w, True)
        if self.c == None:
            c_0 = 0
            c_1 = 0
        else:
            c_0, c_1 = split(self.c)

        return torch.exp(- ( ( LX - c_0 )**2 + ( LY - c_1 ) **2 ) / self.r**2 )
