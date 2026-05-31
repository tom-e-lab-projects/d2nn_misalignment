import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from dual_model import Onn
from torch.utils.data import Dataset


'''
For the purpose of training by testing against the output of a different pre-trained 
system. So this genarates a set of data which is input nist images, 
outputs of these from trained system along with labels

the specifics of super().__init__(), __len__, __getitem__
are used as this means initialising the class creates a dataset which can be 
then batched by Pytorch Dataloader

M as input is the size of LCOS 
file is the file where the pre-trained ONN is stored

'''

class ground(Dataset):
    def __init__(self, file, train_set):
        super().__init__()
        self.file = file 
        self.train_set = train_set
        self.model = self.ground_model()

    def __len__(self):
        return len(self.train_set)
    
    def __getitem__(self, idx):
        image, label = self.train_set[idx]

        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            output = (output* output.conj()).real
            output = output.squeeze(0)

        return image, output, label
    
#extracting pre-trained model

    def ground_model(self):
        model = torch.load(self.file ,weights_only = False)
        return model

'''
M = 256
data = ground(M)
train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)
'''


    