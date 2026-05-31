from dual_model import Onn
import torch

#here we are applying mis-alignment position shifts of x and y to the model
#assuption is the trained values stored in a file

def apply_missalign(file, M, L, lambda0, z, d = torch.zeros(2,2), tilt = torch.zeros(2,2)):
    trained_model = torch.load(file, weights_only=False)

    #initialise shifted model
    shifted_model = Onn(M, L, lambda0, trained_model.parameter, z, d, tilt)

    
    return shifted_model


