'''
one function for training an optical neural network (or another
classfier would use the same structure)
another for validating classification accuracy
needs an instance of a target class which has
methods target_generator and eval_accuracy

for each batch, the system outputs are calculated,
then transformed to estimated target vectors
then loss and accuracy are calculeted from this and logged
and backpropagation gradient descent on the parameters is performed
'''

import torch
from typing import Callable, Any
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class TestConfig:
    """
    A way to store the standard stuff
    """
    criterion: Callable
    val_loader: DataLoader
    target_method: Any
    device: torch.device | None = None

def train(system,
          test_config,
          optimizer, 
          train_loader,
          save_path,
          log_freq,
          epoch_num=50,
        ):
    
    criterion = test_config.criterion
    val_loader = test_config.val_loader
    target_method = test_config.target_method
    device = test_config.device

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    system = system.to(device)
    target_set = target_method.target_generator().to(device)

    # early stopping
    best_loss = float("inf")
    patience = 5
    counter = 0

    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []

    for epoch in range(epoch_num):  
        system.train()
        train_loss = 0.0
        acc_sum = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            targets = target_set[labels]

            optimizer.zero_grad()

            outputs = system(inputs)
            loss = criterion(outputs, targets)


            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_acc, _ = target_method.eval_accuracy(outputs, labels)
            acc_sum += train_acc.item()

            if (i+1) % log_freq == 0:
                train_log = (
                    f'epoch {epoch + 1} {i+1}, '
                    f'train loss: {train_loss/log_freq:5f}, '
                    f'train accuracy: {acc_sum/log_freq:5f}'
                )

                train_losses.append(train_loss/log_freq)
                train_accies.append(acc_sum/log_freq)

                train_loss = 0.0
                acc_sum = 0.0

                torch.save(system.state_dict(), save_path + '/nn' + str(epoch+1) + '.pt')

                with torch.no_grad():
                    val_loss, val_acc, I_val, labels_val = validation(
                        system, val_loader, criterion, target_method, device
                    )

                    val_log = (
                        f'validation loss:{val_loss:5f}, '
                        f'validation accuracy: {val_acc:5f}'
                    )

                    val_losses.append(val_loss)
                    val_accies.append(val_acc)

                print(train_log, '\n', val_log)

                with open(save_path + '/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log + '\n')
                    f.write(val_log + '\n')
            
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    return system, train_losses, train_accies, val_losses, val_accies, I_val, labels_val


def validation(system, test_config):

    criterion = test_config.criterion
    val_loader = test_config.val_loader
    target_method = test_config.target_method
    device = test_config.device
    system.eval()

    val_loss_sum = 0.0
    val_acc_sum = 0.0

    target_set = target_method.target_generator().to(device)

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            targets = target_set[labels]

            outputs = system(inputs)

            val_loss = criterion(outputs, targets)
            val_acc, _ = target_method.eval_accuracy(outputs, labels)

            val_loss_sum += val_loss.item()
            val_acc_sum += val_acc.item()

    return val_loss_sum/(i+1), val_acc_sum/(i+1), outputs, labels


