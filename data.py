import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as v2
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,))
        ])
    
    def prepare_data(self):
        datasets.MNIST(root='data', train=True, download=True)
        datasets.MNIST(root='data', train=False, download=True)

    def setup(self, stage: str = None) -> None:
        if stage in ('fit', None):
            self.train_dataset = datasets.MNIST(root='data', train=True, transform=self.transform)
            self.val_dataset = datasets.MNIST(root='data', train=False, transform=self.transform)

        if stage in ('test', None):
            self.test_dataset = datasets.MNIST(root='data', train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self) -> None:
        datasets.CIFAR10(root='data', train=True, download=True)
        datasets.CIFAR10(root='data', train=False, download=True)

    def setup(self, stage: str = None) -> None:
        match stage:
            case "fit" | None:
                self.train_dataset = datasets.CIFAR10(root='data', train=True, transform=self.transform)
                self.val_dataset = datasets.CIFAR10(root='data', train=False, transform=self.transform)
            case "test" | None:
                self.test_dataset = datasets.CIFAR10(root='data', train=False, transform=self.transform)
            case _:
                raise ValueError(f"Stage {stage} not recognized")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
