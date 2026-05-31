import torch 
from moment_fns import moment

#need to creater z normalisation and include this in loss function


class missalignment_solver:
# put moment into the system definition
    def __init__(self, true_offset, moment, system):
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
        self.true_moment = self.input_moment


    '''
    system should be defined externally and the function should combine system with 
    moment

    '''

     #####   generating moment associated with true offset   #####
    
    def input_moment(self):
        true_intensity = self.through_system(self.true_offset).unsqueeze(0)
        true_moment = self.moment(true_intensity)
        return true_moment
    

    
        moments = self.moment(self.outputs)
        mu_moment = moments.sum(dimo=0,keepdim=True)/samples
        std_moment = torch.sqrt(((moments - mu_moment)**2).sum(dim=0, keepdim=True)/(samples-1))


        #####   taking pure intensity and going through the system with chosen offset   #####
    #####   just generating a uniform phase intensity patch     #####     

    def through_system(self, offset):
        sys = self.system(self.M, self.L, self.lambda0, self.z, offset)
        sys.eval()
        for param in sys.parameters():
            param.requires_grad_(False)
        out = sys(self.image, self.image)
        intensity = (out * out.conj()).real
        return intensity
    

    def light(self):
        I1 = torch.zeros((1, self.M, self.M))
        split = self.M // 3
        I1[0, split:2*split, split:2*split] = 1
        return I1
    
        samples_out = torch.unflatten(samples_out, dim=1, sizes = (2,2))
  
    
if __name__ == '__main__':


    '''
    checklist of experiments: 
    first we see how much gain the local optimization is actually giving: is it really producing that much better to justify
    second if so we can see how well we do without
    third we try and use cma es instead
    '''

    true_offset =  torch.tensor([[4,-4],[3,3.0]])
    num_cycle = 50
    num_generated =10
    solve = inverse_by_optimization( true_offset, moment, system)
    print("first round")
    initial_loss = 1.0
    initial_offset_list = []
    initial_loss_list = []
    for i in range (num_cycle):
        initial_offset = torch.unflatten(20*torch.rand(4)-10, 0 ,sizes = (2,2))
        initial_offset_list.append(initial_offset)
        initial_loss_list.append(initial_loss)

    minimized_first_offset, minimized_first_loss = solve.offsets_to_minimum( initial_offset_list ,initial_loss_list, num_generated, num_cycle)
    print("second round")
    minimized_second_offset, minimized_second_loss = solve.offsets_to_minimum(minimized_first_offset, minimized_first_loss, num_generated, num_cycle)
    print("third round")
    minimized_third_offset, minimized_third_loss = solve.offsets_to_minimum(minimized_second_offset, minimized_second_loss, num_generated, num_cycle)
    print("fourth round")
    minimized_fourth_offset, minimized_fourth_loss = solve.offsets_to_minimum(minimized_third_offset, minimized_third_loss, num_generated, num_cycle)

    

    
