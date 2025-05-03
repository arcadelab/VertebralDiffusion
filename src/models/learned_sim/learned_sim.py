import torch
import torch.nn.functional as F
import torchio as tio
from diffdrr.drr import DRR
from diffdrr.data import read
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

#torch.autograd.set_detect_anomaly(True)
#TODO: I need to add the camera projection to all fxns

# TO BE DONE BY ERIC (WHY DOES THIS NEED TO BE DONE BY DEEPDRR?)
def whole_volume_to_drr(x: torch.Tensor) -> torch.Tensor:
    """
    Creates a DRR from a 3D volume using the camera projection.
    Args:
        x (torch.Tensor): 3D volume tensor.
        proj (geo.CameraProjection): Camera projection object.
    Returns:
        torch.Tensor: 2D DRR tensor.
    """
    #x = x.unsqueeze(0) # 3d to 4d
    # TODO: diffdrr projection of the vert patch, encode the proj using DiffDRR arguments.
    torchio_img = tio.ScalarImage(tensor=x) # yeah i probably shouldn't use an identity matrix
    # but this is just a placeholder for now
    torchio_subject = tio.Subject(volume=torchio_img)
    diffdrr_subject = read(torchio_subject['volume'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drr = DRR(
        diffdrr_subject,
        sdd=1020,
        height=200,
        delx=2.0,
    ).to(device)
    # Example params for now
    rotations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    translations = torch.tensor([[0.0, 550.0, -20.0]], dtype=torch.float32, device=device)
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    return img[0, 0, :, :] # tensor of shape (1, 1, 200, 200) -> (200, 200) for the DRR image

# TO BE DONE BY RIDA
def vertebral_volume_to_drr(x: torch.Tensor) -> torch.Tensor:
    #NEEDS TO BE DIFFERENTIABLE FOR BACKPROP (Need to test for this)
    """
    Creates a DRR from a 3D volume using the camera projection.
    Args:
        x (torch.Tensor): 3D volume tensor.
        proj (geo.CameraProjection): Camera projection object.
    Returns:
        torch.Tensor: 2D DRR tensor.
    """
    #x = x.unsqueeze(0) # 3d to 4d  
    # Same as above, but for the diffusion volume
    torchio_img = tio.ScalarImage(tensor=x) # yeah i probably shouldn't use an identity matrix
    # but this is just a placeholder for now
    torchio_subject = tio.Subject(volume=torchio_img)
    diffdrr_subject = read(torchio_subject['volume'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drr = DRR(
        diffdrr_subject,
        sdd=1020,
        height=200,
        delx=2.0,
    ).to(device)
    # Example params for now
    rotations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    translations = torch.tensor([[0.0, 180.0, 0.0]], dtype=torch.float32, device=device)
    #translations = torch.tensor([[0.0, 850.0, 0.0]], dtype=torch.float32, device=device)
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    return img # tensor of shape (1, 1, 200, 200) -> (200, 200) for the DRR image

def whole_volume_to_drr_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, D, H, W) or (B, D, H, W)
    returns: (B, 1, H_drr, W_drr)  – a batched DRR volume
    """
    drrs = []
    for i in range(x.shape[0]):
        sample = x[i]
        #log.error(f"whole_volume_to_drr_batch: {sample.shape}")
        drri = whole_volume_to_drr(sample)    # 3D
        drrs.append(drri)
    drrs = torch.stack(drrs, dim=0)           # (B, H_drr, W_drr)
    drrs = drrs.squeeze(2)
    return drrs.squeeze(1)                 # (B, 1, H_drr, W_drr)

def vertebral_volume_to_drr_batch(x: torch.Tensor) -> torch.Tensor:
    drrs = []
    for i in range(x.shape[0]):
        sample = x[i]
        drrs.append(vertebral_volume_to_drr(sample))
    drrs = torch.stack(drrs, dim=0)
    drrs = drrs.squeeze(2)
    return drrs.squeeze(1)   

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
    ksize: int = 3,
    results_folder: Path = Path("results"),
    step: int = 0,
) -> torch.Tensor:
    """
    Returns: scalar = lambda_weight * (1 - NCC(∇drr_whole, ∇drr_vert))
    Everything stays on GPU.
    """
    # 1) DRRs
    #with torch.no_grad():
    #    drr_whole = whole_volume_to_drr(vol_whole)       
    #drr_vert = vertebral_volume_to_drr(vol_vertebra)    # idk if we need to make this differentiable
    # it also might be a bad idea to generate thse on the fly 
    #log.error(f"vol_whole: {vol_whole.shape}") 
    #log.error(f"vol_vertebra: {vol_vertebra.shape}")
    drr_whole = whole_volume_to_drr_batch(vol_whole.detach())
    drr_vert  = vertebral_volume_to_drr_batch(vol_vertebra) 
    #log.error(drr_vert.shape)
    #log.error(drr_whole.shape)
    DRR_folder = results_folder / 'DRRs'
    if not DRR_folder.exists():
        DRR_folder.mkdir(exist_ok=True, parents=True)

    if step % 10 == 0:
        save_and_plot_batch(drr_whole, drr_vert, tag = "drr", out_dir=DRR_folder, step = step)
    #img = torch.rand(1, 200, 200) 
    #img_2 = torch.rand(1, 200, 200) #ok so these work which the problem is in not in DRR generation or in gradient calculation
    g_whole = sobel_grad(drr_whole, method="magnitude")
    g_vert  = sobel_grad(drr_vert,  method="magnitude")
    ncc = ncc_2d(g_whole, g_vert)
    return ((1 - ncc) / 2).mean() # [0, 1]

def save_and_plot_batch(drr_whole, drr_vert, tag, out_dir="drr_debug", show_first=2, step=0):
    """
    Plot and save pairs of DRR images side by side
    
    Parameters:
    drr_whole  : (B,1,H,W) or (B,H,W) torch tensor for whole CT DRRs
    drr_vert   : (B,1,H,W) or (B,H,W) torch tensor for vertebrae DRRs
    tag        : base name for saved images
    out_dir    : folder is created if it does not exist
    show_first : how many of the batch to display inline (set 0 to skip)
    step       : current step number to append to filenames
    """
    # Process drr_whole
    drr_whole = drr_whole.detach().cpu()
    if drr_whole.ndim == 4:  # (B,1,H,W) -> (B,H,W)
        drr_whole = drr_whole[:,0]
    
    # Process drr_vert
    drr_vert = drr_vert.detach().cpu()
    if drr_vert.ndim == 4:  # (B,1,H,W) -> (B,H,W)
        drr_vert = drr_vert[:,0]
    
    # Create output directory
    #out_dir = pathlib.Path(out_dir)
    #out_dir.mkdir(exist_ok=True, parents=True)
    
    # Batch size should be the same for both tensors
    batch_size = drr_whole.shape[0]
    
    for i in range(batch_size):
        # Normalize each image to [0,1]
        whole_img = drr_whole[i].float()
        whole_img = (whole_img - whole_img.min()) / (whole_img.max() - whole_img.min() + 1e-6)
        
        vert_img = drr_vert[i].float()
        vert_img = (vert_img - vert_img.min()) / (vert_img.max() - vert_img.min() + 1e-6)
        
        # Save individual images
        #whole_fn = out_dir / f"{tag}_whole_{i:03d}_{step:06d}.png"
        #vert_fn = out_dir / f"{tag}_vert_{i:03d}_{step:06d}.png"
        #plt.imsave(whole_fn, whole_img.numpy(), cmap="gray")
        #plt.imsave(vert_fn, vert_img.numpy(), cmap="gray")
        
        # Create side-by-side plot and save
        combined_fn = out_dir / f"{tag}_combined_{i:03d}_{step:06d}.png"
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 2, 1)
        plt.imshow(whole_img.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Whole #{i}")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(vert_img.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Vertebrae #{i}")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(combined_fn)
        plt.close()
        
        # Show first n images if requested
        #if i < show_first:
        #    plt.show()
        #else:
        #    plt.close()