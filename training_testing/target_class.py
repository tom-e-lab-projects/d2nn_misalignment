'''
the labels in this system
are tensors which are 0 except in
a square 'target' 

all distance values are in units of number of pixels

calling label_generator generates a tensor of
the 10 targets

calling eval_accuracy measures the classification
accuracy of a trained system
'''

import torch

class targets():
    def __init__(self,
                 height,
                 width,
                 target_pos_x,
                 target_pos_y,
                 half_target_size
                ):
        self.h = height
        self.w = width
        self.x = target_pos_x
        self.y = target_pos_y
        self.h_t_s = half_target_size
        self.centres = self.target_centres()
    
    #generates targets
    def target_generator(self): 
        targets = torch.zeros(10,self.h,self.w)
        for k, (cx, cy) in enumerate(self.centres):
            targets[k, 
                     cx-self.h_t_s:cx+self.h_t_s, 
                     cy-self.h_t_s:cy+self.h_t_s
                    ] = 1
        return targets.unsqueeze(1)
    
    #evaluates classification accuracy
    def eval_accuracy(self, output, label):
        device = output.device
        # output shape is B, n_systems, h,w
        patches = []
        for (cx, cy) in self.centres:
            patch = output[:, :, 
                        cx-self.h_t_s:cx+self.h_t_s, 
                        cy-self.h_t_s:cy+self.h_t_s]
            #create each patch as B, n_system, 1
            patches.append(patch.sum(dim=(2,3)))
        #stacks to B, n_system, 10
        post = torch.stack(patches, dim=-1)  
        label_hat = post.argmax(dim=-1)
        acc = (label_hat == label.unsqueeze(1)).float().mean()
        return acc, label_hat
    
    #the centres of the 10 targets in the reciever plane
    def target_centres(self):
        cens = [[-self.x, self.y],
                 [0, self.y],
                 [self.x, self.y],
                 [-self.x/2-self.x,0],
                 [-self.x/2,0],
                 [self.x/2,0],
                 [self.x/2+self.x,0],
                 [-self.x, -self.y],
                 [0, -self.y], 
                 [self.x, -self.y]
                ]
        cens2 = []
        for x, y in cens:
            cens2.append([int(self.h/2+x),int(self.w/2+y)]) 
        return cens2
    
   
    
