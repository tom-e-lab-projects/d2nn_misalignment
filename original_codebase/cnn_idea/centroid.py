import torch

"""

takes image stacks of shape (B,1,M,M)

and returns:
    centroids: (B,2) normalized to [-1,1]
    cropped and normalised wrt max intensity:   (B,1,M//3,M//3)
"""

def centroids(images):

    B, C, M, _ = images.shape
    device = images.device
    crop_size = M // 3
    half = crop_size // 2
    images = images/torch.max(images)

    # ----- Compute centroid -----

    y_coords = torch.linspace(-1, 1, M, device=device)
    x_coords = torch.linspace(-1, 1, M, device=device)
    Y, X = torch.meshgrid(y_coords, x_coords, indexing="ij")

    mass = images.sum(dim=(2,3), keepdim=True) + 1e-8

    x_c = (images * X).sum(dim=(2,3), keepdim=True) / mass
    y_c = (images * Y).sum(dim=(2,3), keepdim=True) / mass

    centroids = torch.cat([x_c, y_c], dim=1)
    centroids = centroids.squeeze(-1).squeeze(-1)  # (B,2)

    # ----- Convert centroid from [-1,1] to pixel indices -----

    x_pix = ((centroids[:,0] + 1) * 0.5 * (M-1)).long()
    y_pix = ((centroids[:,1] + 1) * 0.5 * (M-1)).long()

    # ----- Cyclic crop -----

    cropped = torch.zeros((B, C, crop_size, crop_size), device=device)

    for b in range(B):
        start_x = x_pix[b] - half
        start_y = y_pix[b] - half

        xs = torch.arange(start_x, start_x + crop_size, device=device) % M
        ys = torch.arange(start_y, start_y + crop_size, device=device) % M

        cropped[b] = images[b][:, ys][:, :, xs]

    return centroids, cropped
