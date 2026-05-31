import torch
from N_layer_system_method import N_layer_missaligned_generate

'''
defining a normalised loss and a function for the N layer system
which includes flattening the tensor so the shape is correct for 
the cma es method 
'''

# squared error loss with chosen normalisation
class loss:

    def __init__(self, true_output, initial_outputs):
        self.true_output = true_output
        self.initial_outputs = initial_outputs
        self.sd = self.sigma()

    def eval(self,trial_outputs):
        return torch.mean(((self.true_output-trial_outputs)/self.sd)**2, dim=(1,2))
    
    '''
    initla outputs of shape (num_samples, outputs/sample,9)
    we want to calc sd over the 0th dimension and keep shape
    adn use this to normalise
    '''
    def sigma(self):
        mean_out = torch.mean(self.initial_outputs, dim=0, keepdim = True)
        var = torch.mean((self.initial_outputs - mean_out)**2 )
        return torch.sqrt(var)

    
#we need to flatten tensor to use with the cma es algorithm
class system_with_unflatten:

    def __init__(self, lambda0, dimensions, distances, images, tilt):
        self.model = N_layer_missaligned_generate(lambda0, dimensions, distances, images, tilt)

    #assume displacement is a (N*2) tensor 
    def eval(self, displacement):
        length = int(displacement.size(0)/2)
        unflat_displacement = torch.unflatten(displacement, 0, (length,2) )
        return self.model(unflat_displacement)
    

    


