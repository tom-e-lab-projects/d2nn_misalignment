'''
generates images from mnist, resizes them to correct
size and then converts them into phase scaling
then splits them into a training and validation set
and returns DataLoader structures for each

phase depends on if the input layer is phase or amplitude modulation
'''

import torch
import torchvision
from torchvision import transforms

def phase_transform(x):
    return torch.exp(1j * torch.pi * x)

def data_set_generation(h, w, train_size, val_size, batch_size, phase=False ):

    trans = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ])

    if phase == True:
        trans = transforms.Compose([
            trans,
            phase_transform
        ])

    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True) 
    train_set, val_set, _ = torch.utils.data.random_split(mnist_train, [train_size, val_size, 60000-train_size-val_size])

    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=8,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_set, 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=4)

    return train_loader, val_loader