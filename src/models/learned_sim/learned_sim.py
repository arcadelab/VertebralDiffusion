
import cv2
import killeengeo as geo
import torch
import torch.nn.functional as F

# TO BE DONE BY ERIC 
def whole_volume_to_drr(self, x: torch.Tensor, proj:geo.CameraProjection) -> torch.Tensor:
    """
    Creates a DRR from a 3D volume using the camera projection.
    Args:
        x (torch.Tensor): 3D volume tensor.
        proj (geo.CameraProjection): Camera projection object.
    Returns:
        torch.Tensor: 2D DRR tensor.
    """
    pass

# TO BE DONE BY RIDA
def vertebral_volume_to_drr(self, x: torch.Tensor, proj:geo.CameraProjection) -> torch.Tensor:
    #NEEDS TO BE DIFFERENTIABLE FOR BACKPROP (Need to test for this)
    """
    Creates a DRR from a 3D volume using the camera projection.
    Args:
        x (torch.Tensor): 3D volume tensor.
        proj (geo.CameraProjection): Camera projection object.
    Returns:
        torch.Tensor: 2D DRR tensor.
    """
    pass
def sobel_grad_fp(gray: np.ndarray, 
              ksize: int = 3, 
              method: str = "magnitude"
             ) -> np.ndarray:
    """
    Compute floating-point Sobel gradient of a grayscale image.
    
    Args:
        gray:      single-channel image, uint8 or float32
        ksize:     Sobel kernel size (1, 3, 5, or 7)
        method:    how to combine Gx & Gy:
                - "magnitude":   sqrt(Gx^2 + Gy^2)
                - "average":     0.5*Gx + 0.5*Gy
    
    Returns:
        Float32 image of the same H×W shape, gradient map.
    """
    # 1) compute float32 Sobels
    gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    
    # 2) combine
    if method == "magnitude":
        # true Euclidean norm
        grad = cv2.magnitude(gX, gY)
    elif method == "average":
        # simple weighted sum
        grad = cv2.addWeighted(gX, 0.5, gY, 0.5, 0.0)
    else:
        raise ValueError(f"Unknown method '{method}'; use 'magnitude' or 'average'.")
    return grad