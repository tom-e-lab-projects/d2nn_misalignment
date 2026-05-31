# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:41:18 2023

@author: sleepingcat
github: https://github.com/sleepingcat42
e-mail: sleepingcat@aliyun.com
"""

import torch
from model import system
from torch import optim
import matplotlib.pyplot as plt
from moment_fns import moment
from generated_images import random_data
import time 
import os  


def train(learning_system,
          criterion, 
          optimizer, 
          true_moment,
          I1,
          I2,
          moment,
          std_moment,
          save_path,
          epoch_num=50,
          device = 'cuda:0' if torch.cuda.is_available() else 'cpu'):

    for epoch in range(epoch_num):  
        
        #def closure():
        optimizer.zero_grad()

        #forward through main model 

        learning_output = learning_system(I1, I2)

        learning_intensity = (learning_output * learning_output.conj()).real

        learning_intensity = learning_intensity.unsqueeze(0)
        learning_moment = moment(learning_intensity)
        loss = criterion(learning_moment/std_moment, true_moment/std_moment)

        loss.backward(retain_graph = True)
          
            #return loss

        # updat
        
        #loss = optimizer.step(closure)
        optimizer.step()

        train_loss = loss.detach()

            #which ends here
        if (epoch+1) % 100 == 0:    # print every 2000 mini-batches
                # acc = accuracy(outputs, labels)

                with torch.no_grad():
                    names_values = []
                    for name, param in learning_system.named_parameters():
                        value = param.detach().clone()
                        names_values.append((name, value))

                train_log = f'epoch {epoch + 1}, train loss: {train_loss}, params: {names_values}'
                #saving onn for each epoch
                torch.save(learning_system, save_path+'/system'+str(epoch+1)+'.pt')
                print(train_log,'\n')
                with open(save_path + '/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log+'\n')

    l = loss.detach().clone()
    param = next(learning_system.parameters())
    return param.detach().cpu().clone(), l
        



# line below to stop the code running if this is imported into another file
if __name__ == '__main__':
    file_path = 'model'
    if  os.path.exists(file_path) == False:
        os.makedirs(file_path)
 
    ######################## Optical parameters /mm ###########################

    lambda0 = 500e-9
    P  = 5e-6
    M = 256
    L = M * P
    z = [0.02,0.02,0.2,0.2] 
    tilt = torch.tensor([[0,0],[0,0.0]])
    d = torch.tensor([[4*P,P],[-5*P,P]])


    d_ = torch.arctanh(1/(10*5e-6)*d)
    print(d_)
    I1 = torch.zeros(((1,M,M)))
    split = M // 3
    I1[0,split:2*split,split:2*split] = 1
    I2 = I1.clone()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    num_samples = 500
    image_set = random_data(num_samples , M, system)
    loader = torch.utils.data.DataLoader(image_set, batch_size=len(image_set))
    X, moments = next(iter(loader))
    mu_moment = moments.sum(dim=0,keepdim=True)/num_samples
    std_moment = torch.sqrt(((moments - mu_moment)**2).sum(dim=0, keepdim=True)/(num_samples-1))
    normalised_moment = ( moments - mu_moment ) / std_moment
    true_system = system(M, L, lambda0, z, d_, tilt)

    d2 =  torch.tensor([[0,0],[0,0.0]])

    learning_system = system(M, L, lambda0, z, d2, tilt).to(device)

    true_output = true_system(I1, I2)
    true_intensity = (true_output * true_output.conj()).real
    true_moment = moment(true_intensity.unsqueeze(0))
    
    epoch_num = 3000

    '''
    optimizer = torch.optim.LBFGS(
    learning_system.parameters(),
    lr=0.5,                # start here
    max_iter=20,           # per .step() call
    max_eval=25,
    history_size=10,
    line_search_fn="strong_wolfe"
    )
    '''
    '''
    optimizer = torch.optim.SGD(
    learning_system.parameters(),
    lr=1e-14
)  
'''
    optimizer = torch.optim.AdamW(
            learning_system.parameters(),
            lr=1e-2,
            weight_decay=1e-4
    )

    criterion = torch.nn.MSELoss()

    
    start_time = time.time()

    best_d, loss = train(learning_system, 
            criterion, 
            optimizer,
            true_moment,
            I1,
            I2,
            moment,
            std_moment,
            file_path, 
            epoch_num)    
    
    end_time = time.time()
    print(f'runing time: {end_time - start_time: 5f}s')
    print(best_d)
    print(loss)






    