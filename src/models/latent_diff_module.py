from typing import Any, Optional
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from .ddpm import Unet3D, GaussianDiffusion
from .ddpm import *
import numpy as np
import lightning
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from src.utils.pylogger import get_pylogger
import nibabel as nib
#from  .vqgan_module import VQGAN3D # this is the og pixel-space diffusion model 


log = get_pylogger(__name__)


# Function to convert a tensor to a GIF and save it.
def volume_tensor_to_nifti(tensor, path):
    """
    Save a 3D volume tensor to a NIfTI file.
    
    Args:
        tensor (torch.Tensor): A tensor of shape (channels, depth, height, width). 
                               If channels == 1, it will be squeezed.
        path (str): File path to save the NIfTI file, e.g. 'output_volume.nii'
    """
    # Normalize the tensor to [0, 1] range.
    #tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # If there's a single channel, remove that dimension.
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Convert the tensor to a NumPy array. If needed, move to CPU.
    np_volume = tensor.cpu().numpy()
    #assert(tensor.max() <= 1.0)
    #assert(tensor.min() >= -1.0)
    log.error(np_volume.shape)
    
    # Create an identity affine. You can change this if you have spatial metadata.
    affine = np.eye(4)
    
    # Create a NIfTI image and save it.
    nifti_img = nib.Nifti1Image(np_volume, affine)
    nib.save(nifti_img, path)
    print(f"Saved volume to {path}")

#helper class for the diff model 

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    

class LatentDiffusionModule(LightningModule):
    """LighningModule for training a SWIN module.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """
    def __init__(
        self,
        unet3d: Unet3D,
        diffusion: GaussianDiffusion,
        ema_decay: float,
        update_ema_every: int,
        step_start_ema: int,
        save_and_sample_every: int,
        gradient_accumulate_every: int,
        max_grad_norm: float,
        num_sample_rows: int,
        results_folder: str,
        amp: bool,
        optimizer: list[torch.optim.Optimizer],
        lr_scheduler: torch.optim.lr_scheduler,
        batch_size: int,
        cache_dir: Optional[str] = None,
        cache_threshold: int = 20000, # User-specified threshold for num of cache vectors
    ):
        """
        Args:
            diffusion (nn.Module): Your GaussianDiffusion instance (wrapping your UNet3D).
            optimizer_config (dict): Optimizer settings (e.g. {"lr": 3e-4}).
            ema_decay (float): EMA decay factor.
            update_ema_every (int): Frequency (in steps) for updating the EMA model.
            step_start_ema (int): Step number after which to start EMA updates.
            save_and_sample_every (int): Frequency (in steps) to save checkpoints and sample outputs.
            gradient_accumulate_every (int): Number of steps to accumulate gradients.
            train_num_steps (int): Total number of training steps.
            max_grad_norm (float): Maximum gradient norm for clipping (or None).
            num_sample_rows (int): Number of rows for arranging samples in a grid.
            results_folder (str): Folder path for saving checkpoints and samples.
            amp (bool): Whether to use mixed precision training.
            batch_size (int): Batch size used for sampling (and checkpointing).
        """
        super().__init__()
        # Save hyperparameters (except objects that don't need saving)
        self.save_hyperparameters()
        self.diffusion = diffusion
        self.diffusion.denoise_fn = unet3d
        self.ema_decay = ema_decay
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.gradient_accumulate_every = gradient_accumulate_every
        #self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.amp = amp
        self.batch_size = batch_size

        # Setup AMP scaler.
        self.scaler = GradScaler(enabled=self.amp)

        # Create EMA model.
        self.ema = EMA(self.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion)

        # Manual optimization.
        self.automatic_optimization = False

        # Initialize a training step counter.
        self.step = 0

        #implement caching for better computational efficiency

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_threshold = cache_threshold
        self._cached_vectors_count = 0 # To track how many vectors are cached
        self._only_read_cache = False # Flag to switch mode
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            # Optional: Count existing files on startup
            if self.cache_dir.exists():
                 # A more robust check might parse filenames if they contain identifiers
                 self._cached_vectors_count = len(list(self.cache_dir.glob('*.pt'))) # Assuming .pt files


    #def configure_callbacks(self, ckpt_folder):
    #    checkpoint_callback = ModelCheckpoint(
    #    dirpath=ckpt_folder,
    #    filename='model-{epoch}',
    #    save_top_k=-1,  # Keep all checkpoints
    #    save_weights_only=False  # Make sure to save optimizer state
    #    )
    #    return [checkpoint_callback]
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        opt = self.hparams.optimizer
        opt = opt(params=self.parameters())

        # no lr scheduler for now
        if self.hparams.lr_scheduler is not None:
            sched = self.hparams.lr_scheduler
            sched = sched(optimizer=opt)
            return {"optimizer": opt, "lr_scheduler": sched}

        return opt

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None) -> torch.Tensor: # fxn not used
        """
        Forward pass for inference/sampling. Calls the UNet inside the diffusion model.
        """
        assert(1==2)
        return self.diffusion.unet(x, t, cond=cond)

    def training_step(self, batch: Any) -> torch.Tensor:
        # Retrieve the optimizer.
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        
        # Log the learning rate: log on every step and show it in the progress bar.
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)
        
        # Assume 'batch' is already on the correct device.
        with autocast(enabled=self.amp):
            #log.error(batch[0].shape)
            loss, _ = self.diffusion(batch)
        # Scale loss for gradient accumulation.
        loss = loss / self.gradient_accumulate_every
        self.scaler.scale(loss).backward()
        #log.info(f"Step {self.step}, Global Step {self.global_step}")
        if (self.step + 1) % self.gradient_accumulate_every == 0: # accumulated gradient already now 
            if self.max_grad_norm is not None:
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.max_grad_norm)
            #log.error("stepping")
            self.scaler.step(opt)
            self.scaler.update()
            opt.zero_grad()
            self.lr_schedulers().step()
            
            # IN orig code, EMA model only updates after grad_accum is done but here I can't do that so i only execute
            # after the grad has accumulated`
            #log.info(f"Step {self.global_step}: loss = {loss.item()}") this actually increments good 
            # EMA update after step_start_ema and every update_ema_every steps.
            if self.step >= self.step_start_ema and (self.step % self.update_ema_every == 0):
                self.ema.update_model_average(self.ema_model, self.diffusion)
            # Save checkpoints and sample images at intervals.
        if self.step != 0 and self.step % self.save_and_sample_every == 0:
            self.ema_model.eval()
            with torch.no_grad():
                milestone = self.step
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)
                # Sample using the EMA model (assumes self.diffusion.sample exists).
                all_videos_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_videos_list = torch.cat(all_videos_list, dim=0)
                log.info(all_videos_list.shape)
            milestone = self.global_step
            volume_folder = self.results_folder / 'volumes'
            volume_folder.mkdir(exist_ok=True, parents=True)
            volume_path = str(volume_folder / f'{milestone}.nii')
            # Save the volume using the custom function.
            volume_tensor_to_nifti(all_videos_list, volume_path)
            ckpt_folder = self.results_folder / 'checkpoints'
            self.ema_model.train() # sets back to train model 

        self.log("train/loss", loss * self.gradient_accumulate_every, prog_bar=True)
        self.step += 1
        return loss
    def validation_step(self, batch: Any) -> torch.Tensor:
        #log.error(batch[0].shape)
        loss, _ = self.diffusion(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Any) -> torch.Tensor:
        loss, _ = self.diffusion(batch)
        self.log("test/loss", loss, prog_bar=True)
        return loss
        