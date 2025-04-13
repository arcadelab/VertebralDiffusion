from .rotate import rotate_sphere_uniformly
import torchio as tio
import numpy as np 
from src.utils import pylogger
log = pylogger.get_pylogger(__name__)

class RandomUniformRotation(tio.Transform):
    """
    TorchIO augmentation that applies a random elevation rotation to 3D volumes.
    """
    def __init__(self, max_azimuth=2*np.pi, max_elevation = np.pi, **kwargs):
        """
        Args:
            max_angle (float): Maximum rotation angle in radians.
            keys (tuple): Keys of the images in the subject to which the transform is applied.
        """
        super().__init__(**kwargs)
        
        self.max_azimuth = max_azimuth
        self.max_elevation = max_elevation

    def apply_transform(self, subject):
        #log.error(subject.keys())
        #log.error(subject.values())
        for image in subject.values():
            data = image.data
            #log.error(data.shape) 
            #assert(1==2)  
            rotated_data = rotate_sphere_uniformly(data, max_azimuth=self.max_azimuth, max_elevation=self.max_elevation)
            #assert(1==2)
            # Update the subject with the rotated image data
            ##if lenrotated_data.:
            #    rotated_data = rotated_data.unsqueeze(0)
            #log.error(rotated_data.shape)
            #log.error(data.shape)
            image.set_data(rotated_data)
       # log.error(image.shape)
        return subject
