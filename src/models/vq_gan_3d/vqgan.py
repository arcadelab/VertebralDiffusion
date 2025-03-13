"""Adapted from https://github.com/SongweiGe/TATS"""

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_3d.lpips import LPIPS
from vq_gan_3d.codebook import Codebook
from monai.metrics.regression import SSIMMetric, PSNRMetric


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


class VQGAN(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim
        self.automatic_optimization = False
        self.n_codes = cfg.model.n_codes

        self.encoder = Encoder(
            cfg.model.n_hiddens,
            cfg.model.downsample,
            cfg.dataset.image_channels,
            cfg.model.norm_type,
            cfg.model.padding_type,
            cfg.model.num_groups,
            ch_muls=cfg.model.ch_muls,
        )
        use_larger_decoder = False
        if "larger_decoder" in kwargs.keys():
            use_larger_decoder = kwargs["larger_decoder"]
        elif "larger_decoder" in cfg.model.keys():
            use_larger_decoder = cfg.model.larger_decoder
        self.decoder = Decoder(
            cfg.model.n_hiddens,
            cfg.model.downsample,
            cfg.dataset.image_channels,
            cfg.model.norm_type,
            cfg.model.num_groups,
            ch_muls=cfg.model.ch_muls,
            larger_decoder= use_larger_decoder,
        )
        self.enc_out_ch = self.encoder.out_channels

        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch,
            cfg.model.embedding_dim,
            1,
            padding_type=cfg.model.padding_type,
        )
        self.post_vq_conv = SamePadConv3d(cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(
            cfg.model.n_codes,
            cfg.model.embedding_dim,
            no_random_restart=cfg.model.no_random_restart,
            restart_thres=cfg.model.restart_thres,
        )
        # self.classifier = nn.Sequential(nn.Conv3d(self.enc_out_ch, 1,kernel_size=4,stride=2,padding=2), nn.ReLU(),
        #                                 nn.Linear(self.enc_out_ch*self.enc_out_ch//4, 1), nn.Sigmoid()
        #                                 )
        self.unique_indices = set()

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels,
            cfg.model.disc_channels,
            cfg.model.disc_layers,
            norm_layer=nn.BatchNorm2d,
        )
        self.video_discriminator = NLayerDiscriminator3D(
            cfg.dataset.image_channels,
            cfg.model.disc_channels,
            cfg.model.disc_layers,
            norm_layer=nn.BatchNorm3d,
        )

        if cfg.model.disc_loss_type == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == "hinge":
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = cfg.model.image_gan_weight
        self.video_gan_weight = cfg.model.video_gan_weight

        self.perceptual_weight = cfg.model.perceptual_weight

        self.l1_weight = cfg.model.l1_weight
        self.hc_weight = cfg.model.hc_weight
        self.lc_weight = cfg.model.lc_weight
        self.ssim_metric = SSIMMetric(spatial_dims=3, data_range=5)
        self.psnr_metric = PSNRMetric(max_val=5)
        self.save_hyperparameters()

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output["embeddings"], vq_output["encodings"]
            else:
                return vq_output["encodings"]
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output["encodings"]
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False):
        B, C, T, H, W = x.shape
        # print(f"x.shape: {x.shape}, x.min(): {x.min()}, x.max(): {x.max()}")
        # print(f"Mean : {torch.mean(x)}, Std : {torch.std(x)}, Max : {torch.max(x)}, Min : {torch.min(x)}")
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        self.unique_indices.update(
            torch.unique(vq_output["encodings"].detach()).tolist()
        )

        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        assert (
            x_recon.shape == x.shape
        ), f"SHAPE MISMATCH, x_recon.shape: {x_recon.shape}, x.shape: {x.shape}"
        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        lc_mask = torch.where(
            torch.logical_and(x.detach() > 1.5, x.detach() < 2.75),
            torch.ones_like(x),
            torch.zeros_like(x).to(x.device),
        ).float()
        hc_mask = torch.where(
            x.detach() > 2.75, torch.ones_like(x), torch.zeros_like(x).to(x.device)
        ).float()
        recon_loss += F.l1_loss(x_recon * lc_mask, x * lc_mask) * self.lc_weight
        recon_loss += F.l1_loss(x_recon * hc_mask, x * hc_mask) * self.hc_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).to(x.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"

            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = (
                    self.perceptual_model(frames, frames_recon).mean()
                    * self.perceptual_weight
                )

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
            logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_video_loss = -torch.mean(logits_video_fake)
            g_loss = (
                self.image_gan_weight * g_image_loss
                + self.video_gan_weight * g_video_loss
            )
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start
            )
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            video_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(frames)
                for i in range(len(pred_image_fake) - 1):
                    image_gan_feat_loss += (
                        feat_weights
                        * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach())
                        * (self.image_gan_weight > 0)
                    )
            if self.video_gan_weight > 0:
                logits_video_real, pred_video_real = self.video_discriminator(x)
                for i in range(len(pred_video_fake) - 1):
                    video_gan_feat_loss += (
                        feat_weights
                        * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach())
                        * (self.video_gan_weight > 0)
                    )
            gan_feat_loss = (
                disc_factor
                * self.gan_feat_weight
                * (image_gan_feat_loss + video_gan_feat_loss)
            )

            self.log(
                "train/g_image_loss",
                g_image_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/g_video_loss",
                g_video_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/image_gan_feat_loss",
                image_gan_feat_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/video_gan_feat_loss",
                video_gan_feat_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/perceptual_loss",
                perceptual_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/recon_loss",
                recon_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/commitment_loss",
                vq_output["commitment_loss"],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/perplexity",
                vq_output["perplexity"],
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/codebook_util_rate", len(self.unique_indices) / self.n_codes
            )
            return (
                recon_loss,
                x_recon,
                vq_output,
                aeloss,
                perceptual_loss,
                gan_feat_loss,
            )

        if optimizer_idx == 1:
            # Train discriminator
            disc_factor = adopt_weight(
                self.global_step, threshold=self.cfg.model.discriminator_iter_start
            )
            if disc_factor < 1e-4:
                return 0
            logits_image_real, _ = self.image_discriminator(frames.detach())
            logits_video_real, _ = self.video_discriminator(x.detach())

            logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
            logits_video_fake, _ = self.video_discriminator(x_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
            discloss = 10 * disc_factor * (
                self.image_gan_weight * d_image_loss
                + self.video_gan_weight * d_video_loss
            )

            self.log(
                "train/logits_image_real",
                logits_image_real.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/logits_image_fake",
                logits_image_fake.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/logits_video_real",
                logits_video_real.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/logits_video_fake",
                logits_video_fake.mean().detach(),
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/d_image_loss",
                d_image_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/d_video_loss",
                d_video_loss,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train/discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            return discloss

        perceptual_loss = (
            self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
        )
        return recon_loss, x_recon, vq_output, perceptual_loss

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        opt1.zero_grad()
        opt2.zero_grad()

        x = batch["ct"].as_tensor()
        recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss = (
            self.forward(x, 0)
        )
        commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        self.manual_backward(loss)
        opt1.step()

        self.ssim_metric(x_recon, x)
        self.psnr_metric(x_recon, x)
        ssim = self.ssim_metric.aggregate()
        psnr = self.psnr_metric.aggregate()
        self.log(
            "train/ssim",
            ssim,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/psnr",
            psnr,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        discloss = self.forward(x, 1)
        if discloss < 1e-4:
            return loss
        loss = discloss
        self.manual_backward(loss)
        opt2.step()
        return loss

    def on_train_epoch_end(
        self,
    ) -> None:
        self.ssim_metric.reset()
        self.psnr_metric.reset()
        self.unique_indices = set()

    def validation_step(self, batch, batch_idx):
        x = batch["ct"].as_tensor()  # TODO: batch['stft']
        recon_loss, x_recon, vq_output, perceptual_loss = self.forward(x)
        self.log("val/recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log("val/perceptual_loss", perceptual_loss, prog_bar=True, sync_dist=True)
        self.log(
            "val/perplexity", vq_output["perplexity"], prog_bar=True, sync_dist=True
        )
        self.log(
            "val/commitment_loss",
            vq_output["commitment_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        self.ssim_metric(x_recon, x)
        self.psnr_metric(x_recon, x)
        ssim = self.ssim_metric.aggregate()
        psnr = self.psnr_metric.aggregate()
        self.log(
            "val/ssim",
            ssim,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/psnr",
            psnr,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(
        self,
    ):
        self.ssim_metric.reset()
        self.psnr_metric.reset()
        self.unique_indices = set()

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.pre_vq_conv.parameters())
            + list(self.post_vq_conv.parameters())
            + list(self.codebook.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
            eps=1e-4,
        )
        opt_disc = torch.optim.Adam(
            list(self.image_discriminator.parameters())
            + list(self.video_discriminator.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
            eps=1e-4,
        )
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch["ct"].as_tensor()
        x = x.to(self.device)
        frames, frames_rec, _, _ = self(x, log_image=True)
        print(f"frames.shape: {frames.shape}")
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch["ct"].as_tensor()
        _, _, x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log


def Normalize(in_channels, norm_type="group", num_groups=32):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-3, affine=True
        )
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(
        self,
        n_hiddens,
        downsample,
        image_channel=3,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
        ch_muls=[],
    ):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type
        )

        if len(ch_muls) == 0:
            ch_muls = [2**i for i in range(max_ds + 1)]
        else:
            ch_muls = [1] + ch_muls

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * ch_muls[i]
            out_channels = n_hiddens * ch_muls[i + 1]
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type
            )
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups), SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        n_hiddens,
        upsample,
        image_channel,
        norm_type="group",
        num_groups=32,
        ch_muls=[],
        larger_decoder=False,
    ):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        if len(ch_muls) == 0:
            ch_muls = [2**i for i in range(max_us + 1)]
            if larger_decoder:
                ch_muls = ch_muls[1:] + [ch_muls[-1]]
        else:
            ch_muls = [1] + ch_muls
        ch_muls.reverse()

        in_channels = n_hiddens * ch_muls[0]
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups), SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = n_hiddens * ch_muls[i]
            out_channels = n_hiddens * ch_muls[i + 1]
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
        padding_type="replicate",
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type
        )
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


# Does not support dilation. I think stride here must be 1 in order for this to be same convolution
class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.SyncBatchNorm,
        use_sigmoid=False,
        getIntermFeat=True,
    ):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4  # kernel size
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.SyncBatchNorm,
        use_sigmoid=False,
        getIntermFeat=True,
    ):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [
                [
                    nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf),
                    nn.LeakyReLU(0.2, True),
                ]
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [
            [
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, "model" + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _