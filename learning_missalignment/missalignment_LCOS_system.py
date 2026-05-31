
'''
defining an N layer optical system for missasligment testing
so images are used as an instance constant

tilt and displacement missaligment are equivalent up to fresnel order
according to the relation theta * z = d
where theta is angular missalignment, z is propogation, d is displacement missalignment

so we choose angular missalignment (tilt) as an instance constant and
displacement as a method variable

additionally we zero pad the images since fft uses cyclic convolution
'''

import torch
from torch import nn
from missalignment_layers import LCOS, Propogate

class System(nn.Module):

    def __init__(self, lambda0, dimensions, distance, images):
        super(System, self).__init__()
        self.dimensions = dimensions
        self.h = self.dimensions[2].item()
        self.w = self.dimensions[3].item()
        self.distance = distance
        self.N = len(self.distance)
        self.lambda0 = lambda0
        self.images = images
        self.padded_images = self.zero_pad()
        self.padded_dimensions = self.dimensions * 3
        self.propogate = []
        self.lcos = []
        self.lcos.append(LCOS(self.lambda0, self.padded_dimensions))
        self.propogate.append(Propogate(self.lambda0, self.padded_dimensions , self.distance[0]))
        for i in range (self.N-1):
            self.lcos.append(LCOS(self.lambda0, self.padded_dimensions))
            self.propogate.append(Propogate(self.lambda0, self.padded_dimensions, self.distance[i+1]))


    def forward(self, displacement, tilt):   
        input =  self.padded_images[0].unsqueeze(0)  
        u = self.lcos[0](input, torch.ones_like(input), tilt[0])
        u = self.propogate[0](self.padded_images[0].unsqueeze(0), displacement[0])
        for i in range (self.N - 1):
            u = self.lcos[i+1](u, self.padded_images[i+1].unsqueeze(0), tilt[i+1])
            u = self.propogate[i+1](u, displacement[i+1])
            out = (u * u.conj()).real
        return out
    
    def zero_pad(self):
        padded_images = torch.zeros( self.N , 3 * self.h , 3 * self.w )
        padded_images[:,self.h: 2 * self.h, self.w : 2 * self.w] = self.images
        return padded_images
    
if __name__ == '__main__':
        
    c = 3e8*1e3             # speed of light
    f = 400e9               # 400GHz
    lambda0 = c/f           # wavelength
    p = 128
    dimensions = torch.tensor([80,80,p,p])
    distances = [30,30,30]
    images = torch.ones(3,p,p)
    displacement = torch.tensor([[1.0,1],[-3,2],[2,1]])
    tilt = torch.zeros(3,2)
    system = System(lambda0, dimensions, distances, images)
    output = system(displacement, tilt)



    

    


    
