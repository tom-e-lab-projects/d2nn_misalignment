
'''
classes for creating computational misalignment experiments
the intended form of experiment is a chosen set of misalignemnts are tested
sizes of these misalignments are swept 
for each size of misalignment, samples are generated according to Dirichlet
'''

import torch
from dataclasses import dataclass, field
from physics_modelling.N_layer_system import System
from imperfection_testing.extract_modulation import extract_params
import torch.distributions as dist

@dataclass
class MisalignConfig:
    '''
    tilt etc are supposed to be one-hot encoded vectors to signal which stage you want to misalign
    e.g xy_displacement as [1,0,0,0] will be for a 2 modulation layer system where
    you only want to investigate input beam misalignment

    xy_v_tilt is for experiments to test what length scale the effects of these are degenerate too
    the idea is to input as you would for xy_displacement, then tilt is negative this * z which is degenerate up
    to second order

    samples is the number of samples you want to take parrallelise   

    '''
    tilt: torch.Tensor # modulation layers + 1
    xy_displacement: torch.Tensor #modulation layers + 2
    z_displacement: torch.Tensor #modulation layers + 1
    xy_v_tilt: bool
    samples: int

class Misalign(torch.nn.Module):
    '''
    This is a class that initiates misalignment.Config.samples systems with sampled misalignment
    of a given scale for computational experiments of D2NN under misalignment

    scale is supposed to be a size 3 tensor containing scale of [ tilt, xy_displacement, z_displacement]
    that you want to sample from

    '''
    def __init__(self, 
                 config,
                 distances, 
                 o_e, 
                 misalignment_config, 
                 scale,
                 modulation=None
                ):
        
        super().__init__()
        self.config = config
        self.z = distances
        self.o_e = o_e
        self.mis_con = misalignment_config
        self.scale = scale
        self.mod = modulation

        self.samples = self.mis_con.samples

        self.n_layers = len(self.z)-1

        self.tilt, self.xy_displacement, self.z_displacement = self.input_misalignments()

        if self.mis_con.xy_v_tilt is True:
            self.tilt = (self.xy_displacement[:,1:,:] -
                                self.xy_displacement[:,:-1,:]
                                ) / self.z.unsqueeze(-1) 
            
        # the z distances are the ideal values added to displacement
        self.total_z = self.z + self.z_displacement

        self.system = System(self.config, 
                             self.total_z, 
                             self.o_e,
                             self.mod,
                             self.tilt, 
                             self.xy_displacement,
                             )
    
    def forward(self, 
                input, 
                tilt_misalignment=None,
                lateral_misalignment=None
                ):
        
        out = self.system(input,
                          tilt_misalignment,
                          lateral_misalignment
                          )
        return out
    
    def input_misalignments(self):
        '''
        enconding vectors, scale -> samples of misalignment vectors
        '''
        tilt = self.mis_con.tilt
        xy_displacement = self.mis_con.xy_displacement
        z_displacement = self.mis_con.z_displacement

        tilt_mis = self.scale[0] * self.expand_xy( tilt )

        xy_mis = self.scale[1] * self.expand_xy( xy_displacement )

        if self.scale[2] >0:
            z_mis = self.scale[2] * self.sample_dirichlet(z_displacement)
        else:
            z_mis = torch.zeros(self.samples, z_displacement.size(0))
        return tilt_mis, xy_mis, z_mis

    def expand_xy(self,encoded):
        '''
        assuming independance between x,y direction misalignment and same size
        '''
        if encoded.abs().sum().item() > 0 :
            mis = []
            mis.append(self.sample_dirichlet(encoded))
            mis.append(self.sample_dirichlet(encoded))
            mis =torch.stack(mis, dim=-1)
        else:
            mis = torch.zeros( self.samples, encoded.size(0), 2 )
        return mis

    def sample_dirichlet(self, encoded):
        '''
        takes encoding one-hot vectors from MisalignConfig
        to give a Dirichlet sampled output

        also multiplies with Bernoulli on 1,-1 to randomise sign
        '''
        sample_size = self.mask_distribution(encoded, dist.dirichlet.Dirichlet)
        # want the mean to be at scale not the total
        scale_factor = encoded.size(0)
        sample_size = sample_size * scale_factor
        # 0.5 as we want Bernoulli(0.5) for each entry- equal 0,1
        sample_sign = self.mask_distribution(encoded * 0.5, dist.bernoulli.Bernoulli)
        #going from 0,1 to 1,-1
        sample_sign = (-1)**sample_sign

        return sample_size * sample_sign
    
    def mask_distribution(self, encoded, distribution):
        '''
        The distributions require non-zero entries
        so these need to be masked
        '''
        mask = encoded > 0
        encoded_reduced = encoded[mask]

        dist_tmp = distribution(encoded_reduced)
        sample_reduced = dist_tmp.sample((self.samples,))

        sample = torch.zeros(self.samples,encoded.size(0))
        sample[:, mask] = sample_reduced

        return sample
    






