import torch

'''
producing a normalised vector of first three moments across set of size
num_samples, images/sample, H,W
'''

def moment(image):

    num_samples, _ , H, W = image.shape # (num_samples, images/sample, H, W)
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

    normalised_moment = torch.stack([mu_x,mu_y,std_x,std_y,cov_xy,skew_x,skew_y, skew_xy,skew_yx], dim=1)

    #mu_moment = moment.sum(dim=0,keepdim=True)/B
    #std_moment = torch.sqrt(((moment - mu_moment)**2).sum(dim=0, keepdim=True)/(B-1))
    #normalised_moment = ( moment - mu_moment ) / std_moment

    return normalised_moment

'''
defining the loss function as the normalized square moment difference
it makes sense to z normalize according to an initial uniform search
'''

def squared_normalised_moment_diff(true_moment, image, normalization, moment):
    data_moment = moment(image) 
    return torch.mean((data_moment-true_moment)**2/normalization)




