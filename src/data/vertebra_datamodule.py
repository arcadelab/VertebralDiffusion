from typing import Any, Dict, Optional, Tuple
import os
from glob import glob

import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule


class NiftiDataset(Dataset):
    """A simple Dataset for loading NIfTI images from a directory."""
    def __init__(self, data_dir: str, transform: Optional[Any] = None) -> None:
        """
        Args:
            data_dir (str): Path to directory containing NIfTI files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        # Find files with .nii or .nii.gz extension.
        self.nifti_files = sorted(glob(os.path.join(data_dir, "*.nii*")))
        if not self.nifti_files:
            raise ValueError(f"No NIfTI files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.nifti_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        file_path = self.nifti_files[index]
        # Load the image using nibabel
        img = nib.load(file_path).get_fdata()
        # Convert the image to a torch tensor
        img = torch.tensor(img, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
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
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transform = transform
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        # Data is assumed to be available locally.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = NiftiDataset(self.hparams.data_dir, transform=self.transform)
        total_len = len(dataset)
        train_len, val_len, test_len = self.hparams.train_val_test_split
        if train_len + val_len + test_len != total_len:
            raise ValueError(
                f"Sum of train, val, and test lengths ({train_len + val_len + test_len}) does not match dataset size ({total_len})."
            )
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

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

