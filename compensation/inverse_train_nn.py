import torch

def train(fcnn,
          criterion, 
          optimizer, 
          train_loader,
          val_loader,
          save_path,
          device,
          log_freq,
          epoch_num= 1000
          ):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcnn = fcnn.to(device)
    
    train_losses = []
    val_losses = []

    for epoch in range(epoch_num):  
        train_loss = 0.0
        acc_sum = 0.0
        for i, data in enumerate(train_loader, 0):
            true_input, output  = data

            true_input = true_input.to( device )
            output = output.to(device)

            optimizer.zero_grad()

            input = fcnn(output)
            loss = criterion(input, true_input) 
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)

            train_loss += loss.item()
           
            if (i+1) % log_freq == 0:  
                train_log = f'epoch {epoch + 1} {i+1}, train loss: {train_loss/log_freq: 5f}'
    
                train_losses.append(train_loss /log_freq)
                train_loss = 0.0

                torch.save(fcnn.state_dict, save_path+'/cnn'+str(epoch+1)+'.pt')

                with torch.no_grad():
                    val_loss = validation(fcnn, val_loader,criterion)
                    val_log =f'validation loss:{val_loss :5f}'
                    val_losses.append(val_loss)

                print(train_log,'\n', val_log)

                with open(save_path + '/log.txt', "a", encoding='utf-8') as f:
                    f.write(train_log+'\n')
                    f.write(val_log+'\n')

    return fcnn, train_losses, val_losses

def validation(fcnn, 
               val_loader,
               criterion, 
               device):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcnn.eval()

    val_loss_sum = .0
     
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            true_input, output  = data

            true_input.to(device)
            output.to(device)

            input = fcnn( output )

            val_loss = criterion(input, true_input)
            val_loss_sum += val_loss.item()

    return val_loss_sum/(i+1)



