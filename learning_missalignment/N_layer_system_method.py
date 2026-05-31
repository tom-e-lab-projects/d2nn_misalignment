'''
here we define a function that can be used to create a set of 
N-1 inputs to a missaligned system that uniquely identifies it
and hence can be used with a learning method (for us cma es)
to identify the missalignment parameters of the system

the idea is that at all inputs just to use a uniform phase input
and then in each different case of the system to apply beam steering
by some amount to an individual layer

beam steering can then be seen as a hyper parameter to be optimised

'''

import torch
from torch import nn
from missalignment_LCOS_system import System

class N_layer_missaligned_generate(nn.Module):
    def __init__(self, lambda0, dimensions, distances, images, tilt):
        super( N_layer_missaligned_generate , self).__init__()
        self.N = len(distances)
        self.tilt = tilt
        self.tilt_systems = self.beam_steering()
        self.system = System(lambda0, dimensions, distances, images)
    
    def forward(self, displacement):
        images = []
        for i in range (self.N-1):
            image = self.system(displacement, self.tilt_systems[i])
            images.append(image)
        images = torch.stack(images,dim=1)
        outputs = self.moment(images)
        return outputs

    def beam_steering(self):
        tilt_systems = torch.zeros(self.N-1,self.N,2)
        for i in range (self.N-1):
            tilt_systems[i,i,:] = self.tilt
        return tilt_systems
    
    '''
    generating first three moments used as a method to uniquely identify
    each input in a compressed form
    outputs to shape batch_size,images/output,9
    '''
        
    def moment(self,image):

        num_samples, _ , h, w = image.shape # (num_samples, images/sample, h, w)
        device = image.device
        dtype  = image.dtype
        a = torch.arange(h, device=device, dtype=dtype)
        b = torch.arange(w, device=device, dtype=dtype)
        y = a.view(1,1,h,1)
        x = a.view(1,1,1,w)

        P = image.sum(dim=(2,3),keepdim=True) + 1e-8  # prevent divide-by-zero

        mu_x = (image * x).sum(dim=(2,3),keepdim =True) / P  
        mu_y = (image * y).sum(dim=(2,3), keepdim=True) / P

        std_x = torch.sqrt(((x - mu_x)**2 * image).sum(dim=(2,3),keepdim=True) / P)
        std_y = torch.sqrt(((y - mu_y)**2 * image).sum(dim=(2,3),keepdim=True) / P)
        cov_xy = ((x - mu_x)*(y - mu_y)*image).sum(dim=(2,3), keepdim=True) / P

        skew_x = ((x - mu_x)**3 * image).sum(dim=(2,3),keepdim=True) / P / std_x**3
        skew_y = ((y - mu_y)**3 * image).sum(dim=(2,3),keepdim=True) / P / std_y**3
        skew_xy = ((x - mu_x)**2 * (y - mu_y) * image).sum(dim=(2,3),keepdim=True) / P / std_x**2 / std_y
        skew_yx = ((y - mu_y)**2 * (x-mu_x) * image).sum(dim=(2,3),keepdim=True) / P / std_y**2 / std_x

        # stacking into vector
        moment = torch.stack([mu_x,mu_y,std_x,std_y,cov_xy,skew_x,skew_y, skew_xy,skew_yx], dim=2).squeeze(dim=(3,4))
        return moment
    

if __name__ == '__main__':
        
    c = 3e8*1e3             # speed of light
    f = 400e9               # 400GHz
    lambda0 = c/f           # wavelength
    p = 128
    dimensions = torch.tensor([80,80,p,p])
    distances = [30,30,30]
    images = torch.ones(3,p,p)
    displacement = torch.tensor([[1.0,1],[-3,2],[2,1]])
    tilt = torch.tensor([0.1,0.2])
    model = N_layer_missaligned_generate(lambda0, dimensions, distances, images, tilt)
    output = model(displacement)
    print(output.shape)



