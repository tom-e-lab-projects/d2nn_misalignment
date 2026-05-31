import torch


'''
producing a normalised vector of first three moments across batch
takes image tensor input of shape B,M,M and outputs moment tensor of shape B,6
'''


def moment(image):

    B, _ , H, _ = image.shape
    device = image.device
    dtype  = image.dtype
    a = torch.arange(H, device=device, dtype=dtype)
    y = a.view(1,1,H,1)
    x = a.view(1,1,1,H)

    P = image.sum(dim=(2,3),keepdim=True) + 1e-8  # prevent divide-by-zero
    ##########      calculating moments      ##########


    mu_x = (image * x).sum(dim=(2,3),keepdim =True) / P  
    mu_y = (image * y).sum(dim=(2,3), keepdim=True) / P

    std_x = torch.sqrt(((x - mu_x)**2 * image).sum(dim=(2,3),keepdim=True) / P)
    std_y = torch.sqrt(((y - mu_y)**2 * image).sum(dim=(2,3),keepdim=True) / P)
    cov_xy = ((x - mu_x)*(y - mu_y)*image).sum(dim=(2,3), keepdim=True) / P

    skew_x = ((x - mu_x)**3 * image).sum(dim=(2,3),keepdim=True) / P / std_x**3
    skew_y = ((y - mu_y)**3 * image).sum(dim=(2,3),keepdim=True) / P / std_y**3
    skew_xy = ((x - mu_x)**2 * (y - mu_y) * image).sum(dim=(2,3),keepdim=True) / P / std_x**2 / std_y
    skew_yx = ((y - mu_y)**2 * (x-mu_x) * image).sum(dim=(2,3),keepdim=True) / P / std_y**2 / std_x
    ##########      stacking into vector and normalising     ##########

    normalised_moment = torch.stack([mu_x,mu_y,std_x,std_y,cov_xy,skew_x,skew_y, skew_xy,skew_yx], dim=1)
    #mu_moment = moment.sum(dim=0,keepdim=True)/B
    #std_moment = torch.sqrt(((moment - mu_moment)**2).sum(dim=0, keepdim=True)/(B-1))
    #normalised_moment = ( moment - mu_moment ) / std_moment

    return normalised_moment




    