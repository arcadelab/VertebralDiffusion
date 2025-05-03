from typing import Any, Dict, Optional, Tuple
import os
from glob import glob
from collections import defaultdict
from pathlib import Path 
import random
import torchio as tio
from torchio import SubjectsDataset, SubjectsLoader
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
import numpy as np
from src.utils.pylogger import get_pylogger
import warnings
warnings.filterwarnings(
    "ignore",
    message="Using TorchIO images without a torchio.SubjectsLoader in PyTorch >= 2.3 might have unexpected consequences, e.g., the collated batches will be instances of torchio.Subject with 5D images.",
    category=UserWarning,
    module="torchio.data.image"
)
warnings.filterwarnings("ignore", message="Output shape")

warnings.filterwarnings(
    "ignore",
    message="Using TorchIO images without a torchio.SubjectsLoader in PyTorch >= 2.3 might have unexpected consequences, e.g., the collated batches will be instances of torchio.Subject with 5D images.",
    category=UserWarning,
    module="torchio.data.image"
)

from ..augmentations import RandomUniformRotation


log = get_pylogger(__name__)

class NiftiDataset(Dataset):
    """A simple Dataset for loading NIfTI images from a directory."""
    def __init__(self, data_files: dict, dim: int, train: bool = False) -> None:
        """
        Args:
            data_dir (str): Path to directory containing NIfTI files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dim = dim
        self.train = train
        # Find files with .nii or .nii.gz extension.
        pairs = []
        for ct_path, vert_list in data_files.items():
            # vert_list is something like [v1, v2, v3…]
            for vert_path in vert_list:
                # for each vertebra v1, v2, … make a tuple (vert_path, ct_path)
                pairs.append((vert_path, ct_path))
        self.pairs = pairs
        #self.nifti_files = data_files
        if not self.pairs:
            raise ValueError(f"No NIfTI files found in {data_files}")

    def __len__(self) -> int:
        return len(self.pairs)

    def data_aug(self):
        rotation = RandomUniformRotation()
        train_transform = tio.Compose([
        rotation,
        tio.Resample(target=(1, 1, 1)), # don't use 0.5
        tio.Resize((self.dim, self.dim, self.dim)), # shouldn't do but wait till bigger gpu
        #tio.CropOrPad(
        #   target_shape = (self.dim, self.dim, self.dim),
        ##    padding_mode=-1024
        #    ),
        tio.RescaleIntensity(out_min_max = (-1, 1)),
        ])
        
        val_transform = tio.Compose([
            #tio.Resample(target=(1, 1, 0.5))
           rotation,
           tio.Resample(target=(1, 1, 1)),
           tio.Resize((self.dim, self.dim, self.dim)), # shouldn't do but wait till bigger gpu 
           #tio.CropOrPad(
           # target_shape = (self.dim, self.dim, self.dim),
           # padding_mode= -1024
        #),
            tio.RescaleIntensity(out_min_max = (-1, 1)),
        ])
        return train_transform, val_transform
    

    def __getitem__(self, index: int) -> torch.Tensor:
        vert_path, ct_path = self.pairs[index]
        # Load the image using nibabel
        img_nib = nib.load(vert_path)
        #log.error(vert_path)
        img = img_nib.get_fdata()
        #log.error(ct_path)
        whole_CT_nib = nib.load(ct_path)
        whole_CT = whole_CT_nib.get_fdata()
        train_aug, val_aug = self.data_aug() # reinstantiates the rotation every single time get item is called 
        # Convert the image to a torch tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = img.clone().detach().to(torch.float32).unsqueeze(0)

        # Convert the image to a torch tensor
        whole_CT = torch.tensor(whole_CT, dtype=torch.float32)
        whole_CT = whole_CT.clone().detach().to(torch.float32).unsqueeze(0)

        if self.train:
            img = train_aug(img)
        else:
            img = val_aug(img)
       #log.error(img.shape)
        #log.error(whole_CT.shape)
        return {
            "vertebrae": img,      
            "whole_CT" : whole_CT        
        }


class NiftiDataModuleVertAndCT(LightningDataModule):
    """A barebones LightningDataModule for loading NIfTI images."""
    def __init__(
        self,
        vertebrae_dir: str = "data/nifti/",
        whole_CT_dir: str = "data/whole_CT/",
        train_val_test_split: Tuple[int, int, int] = (70, 10, 20),
        batch_size: int = 4,
        dim: int = 128, 
        level: str = "L",
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.save_hyperparameters(logger=False)
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.level = level

    def prepare_data(self) -> None:
        # Data is assumed to be available locally.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Scans data_dir for NIfTI files and splits them into:
        - train (80%)
        - validation (10%)
        - test (10%)
        """
        # Convert the data directory to a Path object.
        #print(Path(self.hparams.data_dir))
        data_dir = Path(str(self.hparams.vertebrae_dir))
        CT_dir = Path(str(self.hparams.whole_CT_dir))
        
        # Find all NIfTI files (supporting both .nii and .nii.gz extensions).
        nifti_files = sorted(list(data_dir.rglob("*.nii")) + list(data_dir.rglob("*.nii.gz")))
        whole_CT_files = sorted(list(CT_dir.rglob("*.nii")) + list(CT_dir.rglob("*.nii.gz")))
        nifti_files = sorted([f for f in nifti_files if self.level in f.name])

        # 1) map each case-ID to its whole CT
        ct_by_case = {
            ct.relative_to(CT_dir).parts[0]: ct
            for ct in whole_CT_files
        }
        verts_by_case = defaultdict(list)
        for v in nifti_files:
            case_id = v.relative_to(data_dir).parts[0]
            verts_by_case[case_id].append(v)
        #log.error(list(verts_by_case.keys())[0])
        #log.error(list(ct_by_case.keys())[0])
        #assert(1==2), len(set(verts_by_case.keys()).intersection(set(ct_by_case.keys())))
        ct_to_verts = {}
        for case_id, verts in verts_by_case.items():
            #log.error(case_id)
            #log.error(verts)
            ct = ct_by_case.get(case_id)
            if ct:
                ct_to_verts[ct] = verts
            else:
                log.error(f"CT not found for case ID {case_id}.")
        #log.error(nifti_files)
        #log.error(self.level)
        if not nifti_files:
            raise FileNotFoundError(f"No NIfTI files found in {data_dir}")
        
        # Shuffle the file list to ensure randomness.
        #random.shuffle(nifti_files)
        #n = len(nifti_files)
        #train_count = int(0.9 * n)
        #val_count = int(0.1 * n)
        #log.error(train_count)
        #log.debug(val_count)
        #log.error(val_count)
        #log.error(train_count)
        # The test set will be the remainder.

        all_cts = list(ct_to_verts.keys())
        random.shuffle(all_cts)
        n = len(all_cts)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        train_cts = all_cts[:n_train]
        val_cts   = all_cts[n_train:n_train+n_val]
        test_cts  = all_cts[n_train+n_val:]

        train_pairs = { ct: ct_to_verts[ct] for ct in train_cts }
        val_pairs   = { ct: ct_to_verts[ct] for ct in val_cts   }
        test_pairs  = { ct: ct_to_verts[ct] for ct in test_cts  }
        
        if stage is None or stage == "fit":
            #train_files = nifti_files[:train_count]
            #val_files = nifti_files[train_count:train_count + val_count]
            self.data_train = NiftiDataset(train_pairs, self.dim, train=True)
            self.data_val = NiftiDataset(val_pairs, self.dim, train=False)
        
        if stage is None or stage == "test":
            #test_files = nifti_files[train_count + val_count:]
            self.data_test = NiftiDataset(test_pairs, self.dim, train=False)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = NiftiDataModuleVertAndCT()

