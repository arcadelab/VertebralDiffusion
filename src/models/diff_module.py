from typing import Any, Optional
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from ddpm import Unet3D, GaussianDiffusion

class DiffusionModule(pl.LightningModule):
    """
    LightningModule for training a diffusion model.

    This module wraps a UNet-based diffusion model (e.g. Unet3D) and its associated 
    GaussianDiffusion process. The GaussianDiffusion class is responsible for:
      - Defining a noise schedule and progressively adding noise to the input.
      - Computing the reverse process that denoises and reconstructs the original data.
      - Calculating the training loss (e.g. via L1 or L2 loss between predicted and actual noise).

    Args:
        unet (Unet3D): The denoising model architecture.
        diffusion (GaussianDiffusion): The diffusion process that encapsulates the forward
            and reverse diffusion dynamics, as well as the loss computation.
        optimizer_config (dict): Configuration for the optimizer (e.g. learning rate).
    """
    def __init__(self, unet: Unet3D, diffusion: GaussianDiffusion, optimizer_config: dict):
        super().__init__()
        # Save hyperparameters to be stored in checkpoints
        self.save_hyperparameters(logger=False)
        self.unet = unet
        self.diffusion = diffusion
        self.optimizer_config = optimizer_config

    def forward(self, x, t, cond=None):
        """
        Inference pass through the underlying UNet model.
        
        Args:
            x (Tensor): Input tensor.
            t (Tensor): Time-step tensor.
            cond (Optional[Tensor]): Optional conditioning information.
        Returns:
            Tensor: Output of the UNet.
        """
        return self.unet(x, t, cond=cond)

    def training_step(self, batch: Any, batch_idx: int):
        """
        Computes the loss for a batch by passing it through the GaussianDiffusion process.
        
        The GaussianDiffusion instance internally performs the forward noising,
        predicts the noise using the UNet model, and computes the loss between the
        predicted and the actual noise.
        """
        # Assume 'batch' is a tensor of shape (B, C, F, H, W)
        loss = self.diffusion(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """
        Computes validation loss similarly to training_step.
        """
        loss = self.diffusion(batch)
        self.log("val/loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Sets up the optimizer. Here we use Adam with the learning rate specified in the config.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_config.get("lr", 1e-4))
        return optimizer

    def sample_images(self, num_samples: int = 4):
        """
        Optionally generate samples from the diffusion model.
        
        This method uses the diffusion process to sample images starting from random noise.
        It can be useful for visual logging during validation.
        """
        with torch.no_grad():
            samples = self.diffusion.sample(batch_size=num_samples)
        return samples
