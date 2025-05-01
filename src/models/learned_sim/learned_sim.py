
import cv2
import killeengeo as geo
import torch
import torch.nn.functional as F
import numpy as np

# TO BE DONE BY ERIC (WHY DOES THIS NEED TO BE DONE BY DEEPDRR?)
def whole_volume_to_drr(x: torch.Tensor, proj:geo.CameraProjection) -> torch.Tensor:
    """
    Creates a DRR from a 3D volume using the camera projection.
    Args:
        x (torch.Tensor): 3D volume tensor.
        proj (geo.CameraProjection): Camera projection object.
    Returns:
        torch.Tensor: 2D DRR tensor.
    """
    # TODO: diffdrr projection of the vert patch, encode the proj using DiffDRR arguments.
    pass

# TO BE DONE BY RIDA
def vertebral_volume_to_drr(x: torch.Tensor, proj:geo.CameraProjection) -> torch.Tensor:
    #NEEDS TO BE DIFFERENTIABLE FOR BACKPROP (Need to test for this)
    """
    Creates a DRR from a 3D volume using the camera projection.
    Args:
        x (torch.Tensor): 3D volume tensor.
        proj (geo.CameraProjection): Camera projection object.
    Returns:
        torch.Tensor: 2D DRR tensor.
    """
    # Same as above, but for the diffusion volume


_SOBEL_X = torch.tensor([[[[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[[[1, 2, 1],
                           [ 0,  0,  0],
                           [-1,  -2,  -1]]]], dtype=torch.float32)

def sobel_grad(img: torch.Tensor, method: str = "magnitude") -> torch.Tensor:
    """
    img: (B,1,H,W) float32 on GPU
    method: "magnitude" or "average"
    returns: (B,1,H,W) float32 on GPU
    """
    # ensure kernels live on the same device & channel-grouped conv
    kx = _SOBEL_X.to(img.device)
    ky = _SOBEL_Y.to(img.device)
    # convolve (groups=1 since channel=1)
    gx = F.conv2d(img, kx, padding='same') # convolve with the image to get X grad,  idk about padding
    gy = F.conv2d(img, ky, padding='same') # convolve with the image to get Y grad, idk about padding

    if method == "magnitude":
        return torch.sqrt(gx*gx + gy*gy + 1e-6)
    elif method == "average":
        return 0.5 * gx + 0.5 * gy
    else:
        raise ValueError(f"Unknown method {method}")

def ncc_2d(X, Y):
    N = X.shape[-1] * X.shape[-2]
    assert N > 1

    # print('X: {}'.format(X.shape))
    # print('Y: {}'.format(Y.shape))

    dim = X.dim()
    d1 = dim - 2
    d2 = dim - 1

    # compute means of each 2D "image"
    mu_X = torch.mean(X, dim=[d1, d2])

    # make the 2D images have zero mean
    X_zm = X - (mu_X.reshape(*mu_X.shape, 1, 1) * torch.ones_like(X))

    # compute sample standard deviations
    X_sd = torch.sqrt(torch.sum(X_zm * X_zm, dim=[d1, d2]) / (N - 1))

    mu_Y = torch.mean(Y, dim=[d1, d2])

    Y_zm = Y - (mu_Y.reshape(*mu_Y.shape, 1, 1) * torch.ones_like(Y))

    Y_sd = torch.sqrt(torch.sum(Y_zm * Y_zm, dim=[d1, d2]) / (N - 1))

    return torch.sum(X_zm * Y_zm, dim=[d1, d2]) / ((N * (X_sd * Y_sd)) + 1.0e-8)


def grad_ncc_loss(
    vol_whole: torch.Tensor,     
    vol_vertebra: torch.Tensor,   
    proj: geo.CameraProjection,
    ksize: int = 3,
) -> torch.Tensor:
    """
    Returns: scalar = lambda_weight * (1 - NCC(∇drr_whole, ∇drr_vert))
    Everything stays on GPU.
    """
    # 1) DRRs
    with torch.no_grad():
        drr_whole = whole_volume_to_drr(vol_whole, proj)       
    drr_vert = vertebral_volume_to_drr(vol_vertebra, proj)    # idk if we need to make this differentiable
    # it also might be a bad idea to generate thse on the fly 

    #get grads 
    g_whole = sobel_grad(drr_whole, method="magnitude")
    g_vert  = sobel_grad(drr_vert,  method="magnitude")

    ncc = ncc_2d(g_whole, g_vert)
    return (1 - ncc) / 2 # [0, 1]