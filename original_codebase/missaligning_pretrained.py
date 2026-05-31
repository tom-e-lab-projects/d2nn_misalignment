import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from dual_model import Onn
from torch import optim
from label_generator import label_generator,eval_accuracy
from npcc_loss import npcc_loss
import time 
import os 
from customisable_onn_train import validation
from missaligning import apply_missalign



if __name__ == '__main__':
    # loading the trained data- it is stored as the class onn
    #onn = torch.load("model/onn19.pt", weights_only=False)

    #this stores the onn class with all information
    #u2d = tensor.squeeze(0) for 1,256,256 would remove the spare dimension

    #reloading standard parameters for inference 
    ######################## Optical parameters /mm ###########################
    c = 3e8*1e3             # speed of light
    f = 400e9               # 400GHz
    lambda0 = c/f           # wavelength
    L = 80                  # DOE size
    z = [30,30,30,30]    

    ######################### Digtital parameters   ###########################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    M = 256                 # sample nums
    batch_size = 128  


    trans =transforms.Compose([Resize(M),ToTensor()])

    ############################### load data #################################
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True) 
    val_set, _ = torch.utils.data.random_split(mnist_train, [512, 60000-512])

    criterion = npcc_loss

    tilt = torch.tensor([[0.01,0.01],[0.01,0.01]])
    d = torch.tensor([[0,0],[0,0]])
    


    file = "trained_models/aligned_model/onn11.pt"

    network = apply_missalign(file, M, L, lambda0, z, d, tilt)

    #performing validation
    val_loader = torch.utils.data.DataLoader(val_set, 
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4)
    with torch.no_grad():
        val_loss, val_acc, I_val, labels_val = validation(network, val_loader,criterion)
   
    print(val_acc)
