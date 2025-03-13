import pytorch_lightning as pl
from pytorch_lightning import Trainer

from data import CIFAR10DataModule
from model import Autoencoder

def main():
    pl.seed_everything(42)

    cifar10_dm = CIFAR10DataModule(batch_size=64, num_workers=8)
    
    autoencoder = Autoencoder(latent_dim=196)

    trainer = Trainer(
        max_epochs=10,
        accelerator="cpu",
        log_every_n_steps=10,
        default_root_dir="checkpoints"
    )

    trainer.fit(autoencoder, cifar10_dm)

if __name__ == "__main__":
    main()
