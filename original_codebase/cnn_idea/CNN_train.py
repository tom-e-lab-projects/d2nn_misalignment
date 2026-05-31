
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from torch import optim
from generated_images import random_data
from cnn_architecture import CNN_1
import matplotlib.pyplot as plt
from model import system 
import time 
import os 


def train(cnn,
          criterion, 
          optimizer, 
          train_loader,
          val_loader,
          save_path,
          epoch_num=100,
          device='cpu'):
    
    train_losses = []
    val_losses = []

    for epoch in range(epoch_num):  
        train_loss = 0.0
        acc_sum = 0.0
        for i, data in enumerate(train_loader, 0):
            cropped, centroids, tilt, offset  = data
            optimizer.zero_grad()
            model_tilt, model_offset = cnn(cropped, centroids)

            loss = criterion(model_tilt, tilt ) + criterion(model_offset, offset)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
           
            if (i+1) % 5 == 0:    # print every 2000 mini-batches
                # acc = accuracy(outputs, labels)
                train_log = f'epoch {epoch + 1} {i+1}, train loss: {train_loss/5: 5f}'
    
                train_losses.append(train_loss /5)
                train_loss = 0.0
                acc_sum = 0.0
                torch.save(cnn, save_path+'/cnn'+str(epoch+1)+'.pt')
                with torch.no_grad():
                    val_loss = validation(cnn, val_loader,criterion)
                    val_log =f'validation loss:{val_loss :5f}'
                    val_losses.append(val_loss)
                print(train_log,'\n', val_log)
                with open(save_path + '/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log+'\n')
                    f.write(val_log+'\n')

    return cnn, train_losses, val_losses

def validation(cnn, val_loader,criterion, device='cuda:0'):
    val_loss_sum = .0
    val_acc_sum = .0
    for i, data in enumerate(val_loader, 0):
        cropped, centroids, tilt, offset  = data
        model_tilt, model_offset = cnn(cropped, centroids)
        val_loss = criterion(model_tilt, tilt ) + criterion(model_offset, offset)
        val_loss_sum += val_loss.item()

    return val_loss_sum/(i+1)


if __name__ == '__main__':
    file_path = 'model'
    if  os.path.exists(file_path) == False:
        os.makedirs(file_path)
 
    ######################### Digtital parameters   ###########################
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    M = 64                # sample nums
    batch_size = 10
    
    trans =transforms.Compose([Resize(M),ToTensor()])
    
    ############################### load data #################################
    num_samples = 768
    image_set = random_data(num_samples , M, system)
    train_set, val_set = torch.utils.data.random_split(image_set, [256, 512])
    
    # train, validation and test
    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=8,pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_set, 
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=4)
    
    # test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
    #                                             batch_size=batch_size,
    #                                             shuffle=True, num_workers=1)

    ################################ train ####################################
    cnn = CNN_1()
    cnn = cnn.to(device)
    epoch_num = 50
    optimizer = torch.optim.AdamW(
            cnn.parameters(),
            lr=1e-3,
            weight_decay=1e-4
    )
    criterion = torch.nn.MSELoss()
    
    ################################ train ####################################
    start_time = time.time()
    cnn, train_losses, val_losses, I_val, labels_val =\
            train(cnn,criterion, optimizer,
                  train_loader,val_loader, 
                  file_path, epoch_num)    
    
    end_time = time.time()
    print(f'runing time: {end_time - start_time: 5f}s')
    
    ########################### show results ##################################    
    epochs = [k for k in range(1,epoch_num+1)] 
    plt.figure(dpi = 300, figsize=(12,4))
    plt.subplot(121)
    plt.plot(epochs, train_losses, '-o')
    plt.plot(epochs, val_losses, '-s')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(['train','validation'])
    plt.subplot(122)
    plt.show()
    
    
  