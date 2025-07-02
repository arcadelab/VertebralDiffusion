import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
from pathlib import Path
from typing import Optional

from .diffusion import GaussianDiffusion, extract, cosine_beta_schedule
from ..vqgan_module import VQGAN3D
from ..vqgan_seg_module import VQGAN3D_Seg

class GaussianDiffusionMetaStackedVQGANPL(GaussianDiffusion):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        use_cls_embed=False,
        cls_embed_kwags={},
        channels=3,
        timesteps=1000,
        loss_type="l1",
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
        vqgan_seg_ckpt=None,
        use_indices=False,
        staked_vqgan=False,
    ):
        super().__init__(
            denoise_fn=denoise_fn,
            image_size=image_size,
            num_frames=num_frames,
            text_use_bert_cls=text_use_bert_cls,
            channels=channels,
            timesteps=timesteps,
            loss_type=loss_type,
            use_dynamic_thres=use_dynamic_thres,
            dynamic_thres_percentile=dynamic_thres_percentile,
            vqgan_ckpt=vqgan_ckpt,
        )
        
        self.device = "cpu"
        self.staked_vqgan = staked_vqgan
        self.use_indices = use_indices
        self.use_cls_embed = use_cls_embed
        
        if self.use_cls_embed:
            self.cls_embed = ClassEmbeding(**cls_embed_kwags)
            
        if vqgan_ckpt:
            self.vqgan = VQGAN_INDICE.load_from_checkpoint(vqgan_ckpt)
            self.vqgan.eval()
            self.vqgan.requires_grad_(False)
        else:
            self.vqgan = None
            
        self.vqgan_seg_ckpt = vqgan_seg_ckpt
        self.configure_vqgan_seg()

    def configure_vqgan_seg(self):
        if self.vqgan_seg_ckpt:
            print(f"using {self.vqgan_seg_ckpt}")
            self.vqgan_seg = VQGAN_SEG.load_from_checkpoint(self.vqgan_seg_ckpt)
            self.vqgan_seg.eval()
            self.vqgan_seg.requires_grad_(False)
            self.vqgan_seg.to(self.device)
        else:
            self.vqgan_seg = None

    def cads_linear_schedule(self, t, tau1, tau2):
        """ CADS annealing schedule function """
        if t <= tau1:
            return 1.0
        if t >= tau2:
            return 0.0
        gamma = (tau2-t)/(tau2-tau1)
        return gamma

    def add_noise(self, y, gamma, noise_scale, psi, rescale=False, zero_cond=None):
        """ CADS adding noise to the condition
        Arguments:
        y: Input conditioning
        gamma: Noise level w.r.t t
        noise_scale (float): Noise scale
        psi (float): Rescaling factor
        rescale (bool): Rescale the condition
        """
        y_mean, y_std = torch.mean(y), torch.std(y)
        if zero_cond is not None:
            noise = zero_cond
        else:
            noise = torch.randn_like(y)
        y = torch.sqrt(gamma) * y + noise_scale * torch.sqrt(1-gamma) * noise
        if rescale:
            y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
            if not torch.isnan(y_scaled).any():
                y = psi * y_scaled + (1 - psi) * y
            else:
                print("Warning: NaN encountered in rescaling")
        return y

    def cad(self, cond, t, t1=0.6, t2=0.8, noise_scale=0.35, psi=0.5, rescale=True):
        t = max(min(t / self.num_timesteps, 1.0), 0.0)
        gamma = self.cads_linear_schedule(t, t1, t2)
        cond = self.add_noise(cond, gamma, noise_scale, psi, rescale)
        return cond

    def p_sample_loop(self, shape, cond=None, cond_scale=1.0, progress=True, use_cad=False):
        with torch.no_grad():
            device = self.betas.device
            b = shape[0]
            img = torch.randn(shape, device=device)

            if progress:
                for i in tqdm(
                    reversed(range(0, self.num_timesteps)),
                    desc="sampling loop time step",
                    total=self.num_timesteps,
                ):
                    if cond is not None and use_cad:
                        cond_feed = self.cad(cond, i)
                    else:
                        cond_feed = cond
                    img = self.p_sample(
                        img,
                        torch.full((b,), i, device=device, dtype=torch.long),
                        cond=cond_feed,
                        cond_scale=cond_scale,
                    )
            else:
                for i in reversed(range(0, self.num_timesteps)):
                    if cond is not None and use_cad:
                        cond_feed = self.cad(cond, i)
                    else:
                        cond_feed = cond
                    img = self.p_sample(
                        img,
                        torch.full((b,), i, device=device, dtype=torch.long),
                        cond=cond_feed,
                        cond_scale=cond_scale,
                    )

            return img

    def sample(self, cond=None, cond_scale=1.0, batch_size=16, overlap=0.25, progress=True, use_cad=False):
        with torch.no_grad():
            if self.use_cls_embed and cond is not None:
                cond = self.cls_embed(cond)
            device = next(self.denoise_fn.parameters()).device

            if is_list_str(cond):
                cond = bert_embed(tokenize(cond)).to(device)

            batch_size = cond.shape[0] if exists(cond) else batch_size
            image_size = self.image_size
            channels = self.channels
            num_frames = self.num_frames
            
            _sample = self.p_sample_loop(
                (batch_size, channels, num_frames, image_size, image_size),
                cond=cond,
                cond_scale=cond_scale,
                progress=progress,
                use_cad=use_cad
            )

            if isinstance(self.vqgan, VQGAN_INDICE):
                # denormalize
                if self.vqgan_seg is not None:
                    dim_ct = self.denoise_fn.channels // 2
                    indices_sample = _sample[:, :dim_ct]
                    seg_indices_sample = _sample[:, dim_ct:]
                    seg_indices_sample = (
                        ((seg_indices_sample + 1.0) / 2.0)
                        * (
                            self.vqgan_seg.codebook.embeddings.max()
                            - self.vqgan_seg.codebook.embeddings.min()
                        )
                    ) + self.vqgan_seg.codebook.embeddings.min()
                    N, C, D, H, W = seg_indices_sample.shape
                    seg_indices_sample = (
                        seg_indices_sample.contiguous().permute(0, 1, 3, 4, 2).view(N, C, H, W, D)
                    )
                else:
                    dim_ct = self.denoise_fn.channels
                    indices_sample = _sample[:, :dim_ct]
                    seg_indices_sample = None
                    
                indices_sample = (
                    ((indices_sample + 1.0) / 2.0)
                    * (self.vqgan.codebook.embeddings.max() - self.vqgan.codebook.embeddings.min())
                ) + self.vqgan.codebook.embeddings.min()

                N, C, D, H, W = indices_sample.shape
                indices_sample = (
                    indices_sample.contiguous().permute(0, 1, 3, 4, 2).view(N, C, H, W, D)
                )
                
                ct_sample, seg_sample = self.indices_to_img(
                    indices_sample,
                    seg_indices_sample,
                    self.vqgan,
                    self.vqgan_seg,
                    (128, 128, 128),
                    (80, 80, 80),
                    (128, 128, 128),
                    16,
                    overlap=overlap,
                    quantize=True,
                    progress=progress,
                )
                
                N, C, H, W, D = ct_sample.shape
                ct_sample = ct_sample.contiguous().permute(0, 1, 4, 2, 3).view(N, C, D, H, W)
                if self.vqgan_seg is not None:
                    seg_sample = seg_sample.contiguous().permute(0, 1, 4, 2, 3).view(N, C, D, H, W)
                else:
                    seg_sample = None
            else:
                raise NotImplementedError("Not implemented, must have vqgan")

            return ct_sample, seg_sample

    def forward(self, x, *args, **kwargs):
        if isinstance(self.vqgan, VQGAN_INDICE):
            with torch.no_grad():
                if not self.use_indices:
                    raise NotImplementedError("Must use indices")
                else:
                    x_ct = x[:, 0]
                    if self.vqgan_seg is not None:
                        x_seg = x[:, 1]
                        x_seg = self.vqgan_seg.codebook.dictionary_lookup(x_seg)
                        N, D, H, W, C = x_seg.shape
                        x_seg = x_seg.contiguous().permute(0, 4, 1, 2, 3).view(N, C, D, H, W)
                        x_seg = (
                            (x_seg - self.vqgan_seg.codebook.embeddings.min())
                            / (
                                self.vqgan_seg.codebook.embeddings.max()
                                - self.vqgan_seg.codebook.embeddings.min()
                            )
                        ) * 2.0 - 1.0
                        
                    x_ct = self.vqgan.codebook.dictionary_lookup(x_ct)
                    N, D, H, W, C = x_ct.shape
                    x_ct = x_ct.contiguous().permute(0, 4, 1, 2, 3).view(N, C, D, H, W)
                    x_ct = (
                        (x_ct - self.vqgan.codebook.embeddings.min())
                        / (
                            self.vqgan.codebook.embeddings.max()
                            - self.vqgan.codebook.embeddings.min()
                        )
                    ) * 2.0 - 1.0
                    
                    if self.vqgan_seg is not None:
                        x = torch.cat((x_ct, x_seg), dim=1)
                    else:
                        x = x_ct
        else:
            x = normalize_img(x)

        b, device, img_size = x.shape[0], x.device, self.image_size
        check_shape(x, "b c f h w", c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        if self.use_cls_embed:
            kwargs["cond"] = self.cls_embed(kwargs["cond"])

        return self.p_losses(x, t, *args, **kwargs) 