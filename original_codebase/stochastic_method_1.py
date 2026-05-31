
import torch 
import torchvision 
from torchvision import transforms 
from torchvision.transforms import Resize, ToTensor 
from model import system 
from torch.utils.data import Dataset 
from moment_fns import moment
from identification_train import train
import os  

'''
A solution to finding the input offset variables of a 2 stage lcos system.
the method goes by: 1. stochastic optimization to find a starting point
                    2. deterministic optimization following
this is used because I don't think one can find a globally convex loss
the loss used in this case is in moment_fns and is based on first three
moments of the output image from system

to use, declare class, them call sequentially generate_data,
moments_from_data, closest_moment to obtain best start and
then train to obtain complete optimization
'''

class inverse_by_optimization:

    def __init__(self, true_offset, moment, system, train):
        self.M = 256
        self.L = 80
        self.z = torch.tensor([30, 30])
        c = 3e8 * 1e3
        f = 400e9
        self.lambda0 = c/ f
        self.max_offset = self.L // 10
        self.true_offset = true_offset
        self.image = self.light()
        self.system = system
        self.moment = moment
        self.train = train
        self.true_moment = self.input_moment()

    
    #####   takes a set of offsets as input and finds the lowest local minima near to the set   #####

    def offsets_to_minimum(self, offset_current, losses_current, perturbation_size, samples_next_gen, samples_cycle):
        offsets_list = []
        losses_list = []
        for i in range (samples_next_gen):
            offset_data = self.mutation(offset_current, losses_current, perturbation_size, samples_cycle)
            moments_data, std_moment_data = self.moments_from_data(offset_data)
            best_offset, best_loss = self.closest_moment(moments_data, std_moment_data, offset_data)
            #offset, loss = self.deterministic_optimization(best_offset, std_moment_data)
            print(f'offset {i+1} is {best_offset} and has loss {best_loss.item()}')
            offsets_list.append(best_offset)
            losses_list.append(best_loss.item())

        return offsets_list, losses_list
    
    '''
    optimizes quadratic loss between true offset and input 
    idea is that it will find the nearest local minima
    '''

    def deterministic_optimization(self,best_offset, std_moment):
        
        file_path = 'optimization_training'
        if  os.path.exists(file_path) == False:
            os.makedirs(file_path)

        learning_system = system(self.M, self.L, self.lambda0, self.z, best_offset)
        epoch_num = 100

        optimizer = torch.optim.AdamW(
                learning_system.parameters(),
                lr=1e-3,
                weight_decay=1e-4
        )

        criterion = torch.nn.MSELoss()

        offset, loss = train(learning_system, 
                criterion, 
                optimizer,
                self.true_moment,
                self.image,
                self.image,
                self.moment,
                std_moment,
                file_path, 
                epoch_num)    
        
        return offset,loss

    ##### just finds the closest moment from a set to the true moment   #####

    def closest_moment(self, data_moments, std_moments, data_offset):
        loss = (((data_moments - self.true_moment) / std_moments) ** 2).sum(dim=1).squeeze()
        best_idx = torch.argmin(loss)
        best_offset = data_offset[best_idx]
        best_loss = loss[best_idx]
        return best_offset, best_loss
    
    #####   generating moment associated with true offset   #####
    
    def input_moment(self):
        true_intensity = self.through_system(self.true_offset).unsqueeze(0)
        true_moment = self.moment(true_intensity)
        return true_moment
    
    #####   takes a set of offset data and returns moment from system   #####

    def moments_from_data(self, data_offset):
        samples = len(data_offset)
        outputs = []
        with torch.no_grad():
            for i in range(samples):
                o = data_offset[i]
                intensity = self.through_system(o)
                outputs.append(intensity)
        self.outputs = torch.stack(outputs)
        moments = self.moment(self.outputs)
        mu_moment = moments.sum(dim=0,keepdim=True)/samples
        std_moment = torch.sqrt(((moments - mu_moment)**2).sum(dim=0, keepdim=True)/(samples-1))
        return moments, std_moment
    
    #####   taking pure intensity and going through the system with chosen offset   #####
    
    def through_system(self, offset):
        sys = self.system(self.M, self.L, self.lambda0, self.z, offset)
        sys.eval()
        for param in sys.parameters():
            param.requires_grad_(False)
        out = sys(self.image, self.image)
        intensity = (out * out.conj()).real
        return intensity
    
    #####   just generating a uniform phase intensity patch     #####       
    
    def light(self):
        I1 = torch.zeros((1, self.M, self.M))
        split = self.M // 3
        I1[0, split:2*split, split:2*split] = 1
        return I1
    
    '''
    two algorithms for data generation
    first is uniform
    second is using loss weighting from previous stage to generate 
    '''  
    
    def mutation(self, offsets_list, losses_list, perturb_size, num_samples):
        offsets_tensor = torch.stack(offsets_list)    
        losses_tensor = torch.tensor(losses_list)
        eps = 1e-8
        #using normalised inverse loss on previous set as prior distribuiton
        weights = 1.0 / (losses_tensor + eps)
        weights = weights / weights.sum()
        #generating samples from this prior
        indices = torch.multinomial(weights, num_samples, replacement=True)
        selected_offsets = offsets_tensor[indices]   
        #and then we perterb from the previous
        perturb = perturb_size*2*(torch.rand_like(selected_offsets) -0.5)
        new_offsets = selected_offsets + perturb
        return new_offsets
    
    
if __name__ == '__main__':


    '''
    checklist of experiments: 
    first we see how much gain the local optimization is actually giving: is it really producing that much better to justify
    second if so we can see how well we do without
    third we try and use cma es instead
    
    '''
    true_offset =  torch.tensor([[4,-4],[3,3.0]])
    num_cycle = 100
    num_generated =10
    solve = inverse_by_optimization( true_offset, moment, system, train)
    print("first round")
    initial_offset = torch.zeros(2,2)
    initial_loss = 1.0
    initial_offset_list = []
    initial_loss_list = []
    for i in range (num_cycle):
        initial_offset_list.append(initial_offset)
        initial_loss_list.append(initial_loss)
    minimized_first_offset, minimized_first_loss = solve.offsets_to_minimum( initial_offset_list ,initial_loss_list, 8.0, num_generated, num_cycle)
    print("second round")
    minimized_second_offset, minimized_second_loss = solve.offsets_to_minimum(minimized_first_offset, minimized_first_loss, 2.0, num_generated, num_cycle)
    print("third round")
    minimized_third_offset, minimized_third_loss = solve.offsets_to_minimum(minimized_second_offset, minimized_second_loss, 1.0, num_generated, num_cycle)
    print("fourth round")
    minimized_fourth_offset, minimized_fourth_loss = solve.offsets_to_minimum(minimized_third_offset, minimized_third_loss, 0.5, num_generated, num_cycle)

    

    
