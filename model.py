import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics.image as TF


class Autoencoder(pl.LightningModule):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.psnr = TF.PeakSignalNoiseRatio()
        self.ssim = TF.StructuralSimilarityIndexMeasure()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)  # Latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Tanh()  # Normalize to [-1, 1]
        )

    def forward(self, x) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, _ = batch
        x_hat = self(x)
        metrics = self.compute_metrics(x, x_hat)

        mse_loss = metrics['mse_loss']
        ssim_loss = 1 - metrics['ssim']
        psnr_loss = 1 / (metrics['psnr'] + 1e-8)

        loss = mse_loss + 1 * ssim_loss + 0.5 * psnr_loss

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        with torch.no_grad():
            x, _ = batch
            x_hat = self(x)

        metrics = self.compute_metrics(x, x_hat)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, prog_bar=True)

        return metrics['mse_loss']
    
    def compute_metrics(self, x, x_hat) -> dict:
        mse_loss = self.mse_loss(x_hat, x)
        mae_loss = self.mae_loss(x_hat, x)
        psnr = self.psnr(x_hat, x)

        if x.shape[1] == 1:
            ssim = self.ssim(x_hat, x)
        else:
            ssim = torch.stack([
                self.ssim(x_hat[:, i:i+1], x[:, i:i+1])
                for i in range(x.shape[1])
            ]).mean() # Average over channels

        return {
            "mse_loss": mse_loss,
            "mae_loss": mae_loss,
            "psnr": psnr,
            "ssim": ssim
        }
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    