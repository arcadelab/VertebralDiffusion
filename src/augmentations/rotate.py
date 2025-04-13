import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os 
import killeengeo as kg
from src.utils import pylogger
log = pylogger.get_pylogger(__name__)

import warnings
warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0."
)


def rotate_sphere_uniformly(volume_tensor, max_azimuth=2*np.pi, max_elevation=np.pi):
    """
    Rotate a 3D volume uniformly using PyTorch.
    
    Parameters:
        volume_tensor: 3D tensor of shape [1, 1, D, H, W] (batch, channel, depth, height, width)
        max_angle: maximum rotation angle in radians
        
    Returns:
        Rotated 3D volume
    """
    # Get rotation vector and matrix as before
    max_azimuth = (180/np.pi) * max_azimuth
    #print('max_azimuth', max_azimuth)
    max_azimuth = np.random.uniform(0, max_azimuth) #in degrees
    #z-axis is perp to x-y plane so that's axis of rotation and we're rotating y-axis 
    azimuth_vector = kg.vector(0, 1,0).rotate(n = kg.vector(0, 0, 1), theta = max_azimuth)
    # Get the rot. matrix for current random rotation vector to z-axis (0, 0, 1) bc we need the opposite for F.affine_grid
    azimuth_matrix = kg.vector([0, 1, 0]).rotfrom(azimuth_vector) # rotation_vector.rotfrom(kg.vector([0, 0, 1])) #

    max_elevation = np.random.uniform(0, np.pi) 
    # Get a random unit vector on sphere sampled uniformly on the sphere btwn 0 and max_angle
    rotation_vector = kg.random.spherical_uniform(center=kg.vector(0, 0, 1), d_phi=max_elevation)
    # Get the rot. matrix for current random rotation vector to z-axis (0, 0, 1) bc we need the opposite for F.affine_grid
    elevation_matrix = kg.vector([0, 0, 1]).rotfrom(rotation_vector) # rotation_vector.rotfrom(kg.vector([0, 0, 1])) #

    rotation_matrix = azimuth_matrix @ elevation_matrix
    theta = torch.tensor(np.array(rotation_matrix), dtype=torch.float32)
    theta = theta[:3, :]
    #log.error(theta)
    #log.error(volume_tensor)
    # Needs to be 5D [B, C, D, H, W]
    if len(volume_tensor.shape) == 4:
        volume_tensor = volume_tensor.unsqueeze(1)
    elif len(volume_tensor.shape) == 3:
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
    
    #log.error(theta.shape)
    #log.error(volume_tensor.shape)
    grid = F.affine_grid(
        theta.unsqueeze(0),  
        volume_tensor.size(),
    )
 
    rotated_volume = F.grid_sample(
        volume_tensor,
        grid,
        mode='bilinear', # for 3D volumes, 'bilinear' actually refers to trilinear interpolation
        padding_mode='-1024', # no constant to -1024 
    )
    rotated_volume = rotated_volume.squeeze(0)
    return rotated_volume 