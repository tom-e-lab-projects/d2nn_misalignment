
import torch 
import torchvision 
from torchvision import transforms 
from torchvision.transforms import Resize, ToTensor 
from model import system 
from torch.utils.data import Dataset 
from moment_fns import moment

'''
this generates a dataset of normalised offset for the two lcos
and the normalised first three moments of the corresponding image
as a an input output dataset pair
'''

class random_data(Dataset):
    def __init__(self, num_samples, M, system, device="cpu"):

        self.L = 80
        c = 3e8 * 1e3
        f = 400e9
        max_offset = self.L // 10

        self.lambda0 = c / f
        self.M = M
        self.z = torch.tensor([30, 30])
        self.device = device

        # Generate parameters
        std = 1/(12)**(0.5)  #variance of a 0,1 uniform is 1/12
        self.tilt = torch.zeros(num_samples, 2, 2)
        self.normalised_offset = (torch.rand(num_samples, 2, 2) - 0.5)/std
        self.offset = (self.normalised_offset) * 2 * max_offset * std
        self.normalised_offset_flat = self.normalised_offset.flatten(start_dim=1)
        
        #generate intensity image
        self.image = self.light().to(device)

        # Precompute outputs
        outputs = []
        with torch.no_grad():
            for i in range(num_samples):
                t = self.tilt[i]
                o = self.offset[i]
                sys = system(self.M, self.L, self.lambda0, self.z, o,t)
                sys.eval()
                for param in sys.parameters():
                    param.requires_grad_(False)

                out = sys(self.image, self.image)
                intensity = (out * out.conj()).real
                outputs.append(intensity.cpu())

        self.outputs = torch.stack(outputs)
        self.moment = moment(self.outputs)
        self.moment = self.moment.squeeze()

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.normalised_offset[idx], self.moment[idx]

    ##### generates a standard intensity image to represent uniform phase lcos #####

    def light(self):
        I1 = torch.zeros((1, self.M, self.M))
        split = self.M // 3
        I1[0, split:2*split, split:2*split] = 1
        return I1
