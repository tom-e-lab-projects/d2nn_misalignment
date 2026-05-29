import torch
import physics_modelling.N_layer_system as nls
import imperfection_testing.misalignment_classes as mc
import torch.utils.data as d


class momentOperations:

    '''
    a class for operations related to my moment method for measuring lateral misalignment
    for identification of the system
    
    the initialised operations are:
    for n_images more than 1, generate beam_steering modulation for the other images
    and to draw n_samples lateral misalignments on a uniform random set defined by displacement_scale
    and calculate a corresponding tensor of moments and normalization based on this set

    n_split is for serialising a tensor calculation on my weak computer

    the lateral_to_moment function can then be used for example for stochastic learning as in cma_es
    and the normalised difference function for this too

    create dataloader creates a dataloader object of the misalignments and moments for training a neural network
    between them
    '''

    def __init__(self, 
                 config,
                 input_image,
                 distances,
                 displacement_scale,
                 n_samples,
                 n_split,
                 n_image=1,
                 steer_scale = None
                 ):
        
        super().__init__()
        self.config = config
        self.input_image = input_image
        self.distances = distances
        self.n_split = n_split
        self.n_layers = distances.size(1) - 1
        self.n_image = n_image
        self.steer_scale = steer_scale

        self.o_e = False

        # defining system
        if self.n_image ==1 :
            self.system = nls.System(config,
                                     distances,
                                     self.o_e,
                                     None
                                     )
        else: 
            self.system = self.random_steering_system()

        # drawing random misalignments 
        std = 1/(12)**(0.5)
        self.random_misalignment_normalised = ( torch.rand(n_samples, self.n_layers + 1, 2) - 0.5 ) / std
        self.random_misalignment = displacement_scale * 2 * self.random_misalignment_normalised *std
        
        # finding moment from misalignment, of shape n_samples, n_images, 9
        self.random_moment = self.lateral_to_moment( self.random_misalignment)
        self.mean_moment, self.std_moment = self.mean_std( self.random_moment)

    def random_steering_system(self):
        '''
        I use the random drawing from Misalign on tilt to generate randomised 
        beam steering between layers as a configuration
        '''
        misalignment_config = mc.MisalginConfig(torch.ones( self.n_layers + 1),
                                                torch.zeros( self.n_layers + 2),
                                                torch.zeros( self.n_layers + 1 ),
                                                False,
                                                self.n_image
                                                )

        system = mc.Misalign(self.config,
                             self.distances,
                             self.o_e,
                             misalignment_config,
                             self.steer_scale,
                             None
                             )

        return system

    def moment(self, image):
        '''
        producing a normalised vector of first three moments across set of size
        num_samples, images/sample, H,W
        '''

        _, _ , H, W = image.shape # n_samples, n_images, H, W
    
        device = image.device
        dtype  = image.dtype
        a = torch.arange(H, device=device, dtype=dtype)
        b = torch.arange(W, device=device, dtype=dtype)
        y = a.view(1,1,H,1)
        x = a.view(1,1,1,W)

        P = image.sum(dim=(2,3),keepdim=True) + 1e-8  # prevent divide-by-zero

        mu_x = (image * x).sum(dim=(2,3),keepdim =True) / P  
        mu_y = (image * y).sum(dim=(2,3), keepdim=True) / P

        std_x = torch.sqrt(((x - mu_x)**2 * image).sum(dim=(2,3),keepdim=True) / P)
        std_y = torch.sqrt(((y - mu_y)**2 * image).sum(dim=(2,3),keepdim=True) / P)
        cov_xy = ((x - mu_x)*(y - mu_y)*image).sum(dim=(2,3), keepdim=True) / P

        skew_x = ((x - mu_x)**3 * image).sum(dim=(2,3),keepdim=True) / P / std_x**3
        skew_y = ((y - mu_y)**3 * image).sum(dim=(2,3),keepdim=True) / P / std_y**3
        skew_xy = ((x - mu_x)**2 * (y - mu_y) * image).sum(dim=(2,3),keepdim=True) / P / std_x**2 / std_y
        skew_yx = ((y - mu_y)**2 * (x-mu_x) * image).sum(dim=(2,3),keepdim=True) / P / std_y**2 / std_x

        ########## stacking into vector ##########

        moment = torch.stack([mu_x,
                              mu_y,
                              std_x,
                              std_y,
                              cov_xy,
                              skew_x,
                              skew_y, 
                              skew_xy,
                              skew_yx
                              ], dim=1)

        return moment

    def mean_std(self, moment):
        mean = torch.mean( moment, dim=1, keepdim=True)
        std = std = torch.sqrt(torch.mean((moment - mean)**2, dim=1, keepdim=True))
        return mean, std
    

    ####################### methods to use after initialisation ########################
    
    def lateral_to_moment(self, lateral_misalignments):
        lateral_misalignments = torch.split(lateral_misalignments, self.n_split, dim=0)
        moments = []

        with torch.no_grad():
            for lateral_misalignment in lateral_misalignments:
                intensity = self.system(
                    self.input_image,
                    lateral_displacement=lateral_misalignment
                )
                moments.append(self.moment(intensity))

        return torch.cat(moments, dim=0)
    
    def squared_normalised_diff(self, true_moment, moment):
        return torch.mean( (moment - true_moment)**2/self.std_moment )
    
    def create_train_val_loader(self, 
                                batch_size, 
                                train_size
                                ):

        normalised_moment = ( self.random_moment - self.mean_moment )/self.std_moment

        dataset = d.TensorDataset( self.random_misalignment_normalised, normalised_moment)

        train_set, val_set = d.random_split(dataset, [train_size, self.n_samples - train_size ])
        
        train_loader = d.DataLoader(train_set, 
                                                batch_size=batch_size,
                                                shuffle=True, 
                                                num_workers=8,
                                                pin_memory = True
                                                )
        
        val_loader = torch.utils.data.DataLoader(val_set, 
                                                batch_size=batch_size,
                                                shuffle=True, 
                                                num_workers=4
                                                )
        
        return train_loader, val_loader

