# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from dual_model import Onn
from torch import optim
from label_generator import label_generator,eval_accuracy
import matplotlib.pyplot as plt
from npcc_loss import npcc_loss
import time 
import os 
from trained_model_output_class import ground

def train(onn,
          criterion, 
          optimizer, 
          train_loader,
          val_loader,
          save_path,
          epoch_num=50,
          device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
    
    label_set = label_generator()
    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []

    for epoch in range(epoch_num):  
        train_loss = 0.0
        acc_sum = 0.0
        for i, data in enumerate(train_loader, 0):
            #inputs, trained_outputs, labels = data
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            targets = label_set[labels]
            #targets = targets.to(device)

            optimizer.zero_grad()

            #forward through main model 
            outputs = onn(inputs)
            I = (outputs * outputs.conj()).real
            loss = criterion(I, targets)

            #upstream gradient: dL/dO
            dL_dO, = torch.autograd.grad(loss, outputs, retain_graph=True)

            #forward through backward model
            outputs_prime = onn.back(inputs)

            #gradient of O' wrt p, using dL/dO as upstream grad
            grad_p, = torch.autograd.grad(
                outputs_prime,
                onn.parameter,        # the shared learnable parameter p
                grad_outputs=dL_dO,  # inject upstream gradient
                retain_graph=True
            )

            # assign custom gradient
            onn.parameter.grad = grad_p

            # update
            optimizer.step()

            train_loss += loss.item()
            train_acc, label_hat= eval_accuracy(I ,labels)
            acc_sum += train_acc.item()

            #which ends here
           
            if (i+1) % 32 == 0:    # print every 2000 mini-batches
                # acc = accuracy(outputs, labels)
                train_log = f'epoch {epoch + 1} {i+1}, train loss: {train_loss/32: 5f},train accuracy: {acc_sum/32: 5f}'
    
                train_losses.append(train_loss /32)
                train_accies.append(acc_sum/32)
                train_loss = 0.0
                acc_sum = 0.0
                #saving onn for each epoch
                torch.save(onn, save_path+'/onn'+str(epoch+1)+'.pt')
                with torch.no_grad():
                    val_loss, val_acc, I_val, labels_val = validation(onn, val_loader,criterion)
                    val_log =f'validation loss:{val_loss :5f}, validation accuracy: {val_acc:5f}'
                    val_losses.append(val_loss)
                    val_accies.append(val_acc)
                print(train_log,'\n', val_log)
                with open(save_path + '/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log+'\n')
                    f.write(val_log+'\n')
    return onn, train_losses, train_accies, val_losses, val_accies, I_val, labels_val

def validation(onn, val_loader,criterion, device= 'cuda:0' if torch.cuda.is_available() else 'cpu' ):
    val_loss_sum = .0
    val_acc_sum = .0
    label_set = label_generator()
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        targets = label_set[labels]
        targets = targets.to(device)
        outputs = onn(inputs)
        I = torch.abs(outputs)**2
        val_loss = criterion(I, targets)
        val_acc, _= eval_accuracy(I ,labels)
        val_loss_sum += val_loss.item()
        val_acc_sum += val_acc.item()
        
    return val_loss_sum/(i+1), val_acc_sum/(i+1), I, labels

# line below to stop the code running if this is imported into another file
if __name__ == '__main__':
    file_path = 'model'
    if  os.path.exists(file_path) == False:
        os.makedirs(file_path)
 
    ######################## Optical parameters /mm ###########################
    c = 3e8*1e3             # speed of light
    f = 400e9               # 400GHz
    lambda0 = c/f           # wavelength
    L = 80                  # DOE size
    z = [30,30,30,30]    
    
    tilt = torch.tensor([[2,2],[2,2]])
    d = torch.tensor([[10,10],[10,10]])
    
    ######################### Digtital parameters   ###########################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    

    M = 256                 # sample nums
    file = "simple_model/onn12.pt"

    #model = torch.load(file ,weights_only = False)
    #parameter = model.parameter
    parameter = torch.zeros(1,M,M)
    batch_size = 128  
    
    trans =transforms.Compose([Resize(M),ToTensor()])
    
    ############################### load data #################################
    
    #downloading torchvision downloads mnist data set so here we are recieving
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True) 
    train_set, val_set, _ = torch.utils.data.random_split(mnist_train, [4096, 512, 60000-4096-512])

    #bringing in trained outputs
    #data = ground(file, train_set)
    
    # train, validation and test
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=0,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_set, 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=0)
    
    # test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
    #                                             batch_size=batch_size,
    #                                             shuffle=True, num_workers=1)

    ################################ train ####################################

    onn = Onn(M, L, lambda0, parameter, z).to(device)
    onn = onn.to(device)
    epoch_num = 30
    optimizer = optim.Adam(onn.parameters(), lr=1e-2)
    criterion = npcc_loss
    
    ################################ train ####################################
    start_time = time.time()
    onn, train_losses, train_accies, val_losses, val_accies, I_val, labels_val =\
            train(onn,criterion, optimizer,
                  train_loader,val_loader, 
                  file_path, epoch_num)    
    
    end_time = time.time()
    print(f'runing time: {end_time - start_time: 5f}s')











    
    ########################### show results ##################################    

    '''
    epochs = [k for k in range(1,epoch_num+1)] 
    plt.figure(dpi = 300, figsize=(12,4))
    plt.subplot(121)
    plt.plot(epochs, train_losses, '-o')
    plt.plot(epochs, val_losses, '-s')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(['train','validation'])
    plt.subplot(122)
    plt.plot(epochs, train_accies, '-o')
    plt.plot(epochs, val_accies, '-s')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train','validation'])
    plt.show()
    
    plt.figure(dpi = 300, figsize=(8, 8))
    b = I_val.cpu().data.numpy()
    for k in range(9):
        plt.subplot(3,3, k+1)
        plt.imshow(b[k,:].squeeze(0)[64:256-64, 64:256-64],cmap='gray')
        plt.title('True Label: '+str(labels_val[k].cpu().numpy()))
        plt.axis('off')
    plt.show()
    
    plt.figure(dpi = 300, figsize=(12,4))
    plt.subplot(131)
    plt.imshow(onn.DiffLayer1.params[0].detach().cpu().numpy(), cmap='gray')    
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(onn.DiffLayer2.params[0].detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(onn.DiffLayer3.params[0].detach().cpu().numpy(), cmap='gray')
    plt.colorbar()
    '''