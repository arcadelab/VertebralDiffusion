from typing import Any, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from PIL import Image
import numpy as np

from .vq_gan_3d import VQGAN3D, Discriminator, weights_init, LPIPS
from ..utils import pylogger
from .losses import DiceLoss2D

log = pylogger.get_pylogger(__name__)


class VQGANModule(LightningModule):
    """LighnintModule for training a VQGAN.

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
        vqgan: VQGAN3D,
        discriminator: Discriminator,
        optimizers: list[torch.optim.Optimizer],
        schedulers: Optional[list[torch.optim.lr_scheduler.LRScheduler]] = None,
        disc_factor: float = 1.0,
        disc_start: int = 50_000,
        rec_loss_factor: float = 1.0,
        perceptual_loss_factor: float = 1.0,
        segmentation_loss_factor: float = 1.0,
    ):
        """
        Args:
            vqgan: VQGAN model
            discriminator: Discriminator model
            optimizers: The VQ optimizer and the discriminator optimizer, partially initialized
            schedulers: list of schedulers for each optimizer, or None.
            disc_factor: The discriminator loss factor.
            disc_start: The number of steps to wait before starting to train the discriminator.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.vqgan = vqgan
        self.discriminator = discriminator
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval()
        self.dice_loss = DiceLoss2D(skip_bg=False)
        self.opt_vq, self.opt_disc = self.configure_optimizers()

        # manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        opt_vq, opt_disc = self.hparams.optimizers
        opt_vq = opt_vq(params=self.vqgan.parameters())
        opt_disc = opt_disc(params=self.discriminator.parameters())

        if self.hparams.schedulers is not None:
            sched_vq, sched_disc = self.hparams.schedulers
            sched_vq = sched_vq(optimizer=opt_vq)
            sched_disc = sched_disc(optimizer=opt_disc)
            return [opt_vq, opt_disc], [sched_vq, sched_disc]

        return [opt_vq, opt_disc]

    def forward(self, x: torch.Tensor):
        # TODO: implement forward pass
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self,
        batch: Any,
        batch_idx: int,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs: dict[str, torch.Tensor]
        targets: dict[str, torch.Tensor]
        inputs, targets = batch

        imgs = inputs["image"]  # intensity augmentations applied
        target_imgs = targets["image"]  # no intensity augmentations

        decoded_images, decoded_segs, _, q_loss = self.vqgan(imgs)

        disc_real = self.discriminator(target_imgs)
        disc_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(
            self.hparams.disc_factor, self.global_step, self.hparams.disc_start
        )

        perceptual_loss = self.perceptual_loss(target_imgs, decoded_images)
        rec_loss = torch.abs(target_imgs - decoded_images)
        perceptual_rec_loss = (
            self.hparams.perceptual_loss_factor * perceptual_loss
            + self.hparams.rec_loss_factor * rec_loss
        )
        perceptual_rec_loss = perceptual_rec_loss.mean()
        g_loss = -torch.mean(disc_fake)

        if decoded_segs is not None:
            seg_loss = self.hparams.segmentation_loss_factor * self.dice_loss(
                decoded_segs, targets["segs"]
            )
        else:
            seg_loss = 0.0

        if self.training:
            # Requires grad for the VQGAN model
            lam = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
            vq_loss = (
                perceptual_rec_loss + seg_loss + q_loss + disc_factor * lam * g_loss
            )
        else:
            vq_loss = 0.0

        d_loss_real = torch.mean(F.relu(1.0 - disc_real))
        d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        self.log(f"{mode}/perceptual_rec_loss", perceptual_rec_loss, on_step=True)
        self.log(f"{mode}/seg_loss", seg_loss, on_step=True)
        self.log(f"{mode}/q_loss", q_loss, on_step=True)
        self.log(f"{mode}/vq_loss", vq_loss, on_step=True)
        self.log(f"{mode}/gan_loss", gan_loss, on_step=True)

        return vq_loss, gan_loss, decoded_images, target_imgs

    def training_step(self, batch: Any, batch_idx: int):
        opt_vq, opt_disc = self.optimizers()
        vq_loss, gan_loss, _, _ = self.model_step(batch, batch_idx, "train")

        opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)

        opt_disc.zero_grad()
        gan_loss.backward()

        opt_vq.step()
        opt_disc.step()

    def validation_step(self, batch: Any, batch_idx: int):
        vq_loss, gan_loss, decoded_images, target_imgs = self.model_step(
            batch, batch_idx, "val"
        )

    def test_step(self, batch: Any, batch_idx: int):
        vq_loss, gan_loss, decoded_images, target_imgs = self.model_step(
            batch, batch_idx, "test"
        )

    def log_images(self, batch, **kwargs):
        inputs, targets = batch
        imgs = inputs["image"]
        target_imgs = targets["image"]
        decoded_images, decoded_segs, _, _ = self.vqgan(imgs)

        input_images = imgs.detach().cpu()
        target_images = target_imgs.detach().cpu()
        decoded_images = decoded_images.detach().cpu()
        target_segs = targets["segs"].detach().cpu()

        images = torch.cat([target_images, input_images, decoded_images], dim=3)
        return images, decoded_segs, target_segs
