
import torch
from torch import nn
from identification_layers import LCOS, Info

'''
through the optical system with missalignments set as parameters for training 
for learning missalignment
'''

class system(nn.Module):

    def __init__(self, M, L, lambda0, z, d=torch.zeros(2,2), tilt=torch.zeros(2,2)):
        super(system, self).__init__()
        self.d = nn.Parameter(d)
        self.theta = tilt
        self.info = Info(M, L, lambda0, z[0])
        self.lcos = LCOS(M, L, lambda0, z[1])

    def forward(self, u1, u2):        
        d = 10*5*1e-6*torch.tanh(self.d)
        u = self.info(u1, d[0,:], self.theta[0,:])
        u = self.lcos(u, u2, d[1, :], self.theta[1,:])
        return u
    


    
