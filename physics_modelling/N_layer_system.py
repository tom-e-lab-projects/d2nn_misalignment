'''
N layer diffractive system for training of phase modulation layers

distances, tilt and xy_displacement should be input as tensors
(n_different_systems, n_layers + m)
where different systems means ones with different imperfections added

hence if having just one system, .unsqueeze(0) on your tensor for safety
with an additional ,2) at the end for tilt, xy_displacement

m is 1 for distances, tilt
2 for xy_displacement as 0th is the xy_displacement of input beam centre


'''

import torch
from torch import nn
from physics_modelling.subsystems import PropagatePadded, GaussianIntensity


class System(nn.Module):
    '''
    N layer diffractive neural network
    o_e is a Boolean or whether to have optical to electrical conversion
    between each layer or not

    '''

    def __init__(self, 
                 config, 
                 distances,  
                 o_e, 
                 modulations,
                 tilt_displacement=None, 
                 lateral_displacement=None
                 ):
        
        super().__init__()

        self.config = config
        self.z = distances
        self.o_e = o_e
        
        self.n = self.z.size(1)-1

        # assumption that modulation are the learnable parameters
        self.mod = nn.ParameterList()

        # applys zero is default if no modulation is given
        if modulations is None:
            modulations = torch.zeros( self.n,
                                      self.config.height,
                                      self.config.width)
            modulations = torch.unbind(modulations, dim=0)

        for modulation in modulations:
            self.mod.append(
                nn.Parameter(modulation)
            )

        tilt_list = self.displacement_list(tilt_displacement, self.n + 1)

        lateral_list = self.displacement_list(lateral_displacement, self.n+2)
            
        # assume Gaussian input and first xy_displacement 
        self.beam = GaussianIntensity( self.config, lateral_list[0])

        self.propagate = nn.ModuleList()
        for i in range (self.n+1):
            self.propagate.append( PropagatePadded( self.config, 
                                                   self.z[:,i],
                                                   tilt_list[i],
                                                   lateral_list[i+1]
                                                   ))

        if o_e == True:
            self.o_e = self.o_e_on
        else:
            self.o_e = nn.Identity()

    def forward(self, 
                input, 
                tilt_displacement=None, 
                lateral_displacement=None
                ):

        tilt_list = self.displacement_list(tilt_displacement, self.n + 1)

        lateral_list = self.displacement_list(lateral_displacement, self.n + 1)
        
        u = self.beam(input)
        u = self.propagate[0](u, 
                              tilt_list[0], 
                              lateral_list[0])

        for i in range (self.n):
            u = self.o_e(u)
            phase_mod = torch.exp(1j * self.mod[i])
            u = phase_mod * u
            u = self.propagate[i+1](u, 
                                    tilt_list[i+1],
                                    lateral_list[i+1])
    
        return self.o_e_on(u)
    
    def o_e_on(self,input):
        return (torch.abs(input)**2).real
    
    def displacement_list(self, displacement, n):
        '''
        a function for handling Nones in the splitting of tensors
        '''
        return [
                None if displacement is None else displacement[:, i, :]
                for i in range(n)
            ]
    

    


    
