from typing import Any, Dict, Optional, Tuple
import os
from glob import glob
from pathlib import Path 
import random
import torchio as tio
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule


class NiftiDataset(Dataset):
    """A simple Dataset for loading NIfTI images from a directory."""
    def __init__(self, data_files: list, train: bool = False) -> None:
        """
        Args:
            data_dir (str): Path to directory containing NIfTI files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
        # Find files with .nii or .nii.gz extension.
        self.nifti_files = data_files
        if not self.nifti_files:
            raise ValueError(f"No NIfTI files found in {data_files}")

    def __len__(self) -> int:
        return len(self.nifti_files)
    

    #may want to add this idk: 
     #random noise
        #tio.RandomNoise(
        #    mean=0.0,
        #    std=(0, 0.1),
        #   p=0.5
        #)

#   tio.RandomMotion(
     #       degrees = 10,
     #       translation=0,
     #   )
    def data_aug(self):
        train_transform = tio.Compose([
        #tio.RandomAffine(
        #    scales=(0.9, 1.1),       
        #    degrees=1,             
        #    translation=0,           
        #    p=0.75                  
        #),
        tio.RandomMotion(
            degrees = 10,
            translation=0,
        )
        ])
        size_transform = tio.Resize((128, 128, 128))
        pad_transform = tio.CropOrPad(
            target_shape = (128, 128, 128),
            padding_mode='constant'
        ) # constant_values =-1024

        return train_transform, size_transform, pad_transform
    

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.nifti_files[index]
        print(file_path)
        # Load the image using nibabel
        img_nib = nib.load(file_path)
        img = img_nib.get_fdata()
        train_aug, resize, pad_transform = self.data_aug()

        # Convert the image to a torch tensor
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        if self.train:
            img = train_aug(img)
        #img = resize(img)
        img = pad_transform(img) #resize(img)#pad_transform(img)
        #print(img.shape)

        #augmented_img_np = img.squeeze(0).numpy()  # remove channel dim for saving
        #augmented_img_nib = nib.Nifti1Image(augmented_img_np, affine=img_nib.affine)

        # Define save path
        #augmented_path = file_path.replace('.nii', '_augmented.nii')
        #augmented_path = Path(file_path).with_name(Path(file_path).stem + '_augmented.nii')
        #nib.save(augmented_img_nib, augmented_path)
        return img


class NiftiDataModule(LightningDataModule):
    """A barebones LightningDataModule for loading NIfTI images."""
    def __init__(
        self,
        data_dir: str = "data/nifti/",
        train_val_test_split: Tuple[int, int, int] = (70, 10, 20),
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train = None
        self.data_val = None
        self.data_test = None

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
        data_dir = Path(str(self.hparams.data_dir))
        
        # Find all NIfTI files (supporting both .nii and .nii.gz extensions).
        nifti_files = sorted(list(data_dir.rglob("*.nii")) + list(data_dir.rglob("*.nii.gz")))
        if not nifti_files:
            raise FileNotFoundError(f"No NIfTI files found in {data_dir}")
        
        # Shuffle the file list to ensure randomness.
        random.shuffle(nifti_files)
        n = len(nifti_files)
        train_count = int(0.8 * n)
        val_count = int(0.1 * n)
        print(train_count, val_count)
        # The test set will be the remainder.
        
        if stage is None or stage == "fit":
            train_files = nifti_files[:train_count]
            val_files = nifti_files[train_count:train_count + val_count]
            self.data_train = NiftiDataset(train_files, train=True)
            self.data_val = NiftiDataset(val_files, train=False)
        
        if stage is None or stage == "test":
            test_files = nifti_files[train_count + val_count:]
            self.data_test = NiftiDataset(test_files, train=False)

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
    _ = NiftiDataModule()

