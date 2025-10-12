import os
import sys
import warnings
import math

# Use plot lib gui for training
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

# Add parent directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

# Suppress Pydantic frozen field warning from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import torch
import wandb
import torchvision
from torchvision.transforms import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from safetensors.torch import save_file
import random
import string

from models import InspoResNetVAE, log_example_images
from helpers import save_checkpoint, load_checkpoint

# Model save path
MODEL_SAVE_PATH = os.path.join(project_dir, 'models/safetensors/vae/')
CHECKPOINT_PATH = os.path.join(project_dir, 'checkpoints/vae/')
LOG_IMAGES_EVERY = 1  # Log example images every N epochs
UPDATE_CURRENT_CHECKPOINT_EVERY = 25  # Update current.pt every N epochs
CREATE_PERMANENT_CHECKPOINT_EVERY = 100  # Create epoch_N.pt every N epochs

# Load data once (outside train function for sweep efficiency)
train_ds = MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_ds = MNIST(root='./data', train=False, download=True, transform=ToTensor())

# Hyperparameter defaults
default_config = dict(
    latent_shape=(1,7,7),
    base_channels=16,
    blocks_per_level=2,
    groups=1,
    dropout=0.4,
    activation='GELU',
    use_skips=True,
    use_bn=True,

    batch_size=128,
    learning_rate=2e-3,
    weight_decay=5e-5,
    epochs=7,

    beta_final=1.0,
    warmup_epochs=1,
    ema=0.97,
)

def train(config=None, project="vae_conv_testing", checkpoint_path=None):
    # Get run ID from checkpoint if resuming
    if checkpoint_path:
        resume_id = torch.load(checkpoint_path, map_location='cpu').get('wandb_run_id')
    else:
        resume_id = None

    # Use default config if none provided for new runs
    if config is None and not resume_id:
        config = default_config

    # Initialize or resume wandb run
    with wandb.init(project=project, config=config, id=resume_id, resume='must' if resume_id else False) as run:
        if resume_id and config is not None:
            wandb.config.update(config, allow_val_change=True)
        config = wandb.config

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        # Get activation function from config
        activation_map = {
            'GELU': nn.GELU,
            'ReLU': nn.ReLU,
            'SiLU': nn.SiLU,
            'LeakyReLU': nn.LeakyReLU,
        }
        act_fn = activation_map.get(config.activation, nn.GELU)

        print('Using device:', device)

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, num_workers=2, shuffle=True, persistent_workers=True)
        val_loader = DataLoader(test_ds, batch_size=config.batch_size, num_workers=2, shuffle=False, persistent_workers=True)

        # Parse latent_shape (supports both old latent_dim and new latent_shape)
        if hasattr(config, 'latent_shape'):
            # Convert to tuple if it's a list or int (from wandb YAML)
            if isinstance(config.latent_shape, (list, tuple)):
                latent_shape = tuple(config.latent_shape)
            elif isinstance(config.latent_shape, int):
                latent_shape = (config.latent_shape,)
            else:
                latent_shape = config.latent_shape
        elif hasattr(config, 'latent_dim'):
            latent_shape = (config.latent_dim,)
        else:
            latent_shape = (1,7,7)  # fallback default

        # Calculate total latent dimensions
        latent_dim_total = math.prod(latent_shape)

        # Create model
        model = InspoResNetVAE(
            latent_shape=latent_shape,
            act=act_fn,
            use_skips=config.use_skips,
            use_bn=config.use_bn,
            base_channels=config.base_channels,
            blocks_per_level=config.blocks_per_level,
            groups=config.groups,
            dropout=config.dropout
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Build config dict
        config_update = {
            "total_params": total_params,
            "latent_dim": latent_dim_total,
            "model_class": model.__class__.__name__
        }
        if hasattr(model, 'config'): config_update["model_config"] = model.config

        wandb.config.update(config_update)
        print(f"Training {model.__class__.__name__} with {total_params:,} parameters")

        # Training setup
        recon_criterion = F.binary_cross_entropy_with_logits
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

        # Load checkpoint if provided
        start_epoch = 0
        if checkpoint_path is not None:
            ckpt = load_checkpoint(checkpoint_path, model, optimizer, ema=ema, device=device)
            start_epoch = ckpt['epoch']
            print(f"Resuming from epoch {start_epoch}")

        spatial = len(model.latent_shape) > 1 if hasattr(model, 'latent_shape') else False
        global_step = 0

        for epoch in range(start_epoch, config.epochs):
            model.train()

            beta = config.beta_final * min((epoch + 1) / config.warmup_epochs, 1.0)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()

                z, x_hat, mu, log_var, z_hat = model(data)

                recon_loss = recon_criterion(x_hat, data, reduction="sum") / data.size(0)

                kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
                kl = kl.view(kl.size(0), -1).sum(dim=1).mean()

                loss = recon_loss + beta * kl

                loss.backward()
                optimizer.step()
                ema.update()

                wandb.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "val_mode": False,
                    "train_loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl.item(),
                    "beta": beta,
                    "beta*kl": (beta * kl).item(),
                })
                global_step += 1
                pbar.set_postfix(Loss=f"{loss.item():.4f}", Recon=f"{recon_loss.item():.4f}", KLw=f"{(beta*kl).item():.5f}")

            # Validation with EMA
            model.eval()
            with ema.average_parameters():
                val_loss = val_recon = val_kl = 0.0
                mu_stats = []
                mse_total = mae_total = ssim_total = psnr_total = 0.0
                num_samples = 0

                with torch.no_grad():
                    for data, _ in val_loader:
                        data = data.to(device)
                        z, x_hat, mu, log_var, z_hat = model(data)

                        recon = recon_criterion(x_hat, data)
                        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon + beta * kl

                        val_loss += loss.item()
                        val_recon += recon.item()
                        val_kl += kl.item()
                        mu_stats.append(mu)

                        # Reconstruction quality metrics (on sigmoid output)
                        x_hat_sigmoid = torch.sigmoid(x_hat)

                        # MSE (Mean Squared Error)
                        mse = F.mse_loss(x_hat_sigmoid, data, reduction='sum')
                        mse_total += mse.item()

                        # MAE (Mean Absolute Error)
                        mae = F.l1_loss(x_hat_sigmoid, data, reduction='sum')
                        mae_total += mae.item()

                        # PSNR (Peak Signal-to-Noise Ratio) - higher is better
                        # PSNR = 10 * log10(MAX^2 / MSE), MAX=1 for normalized images
                        psnr = 10 * torch.log10(1.0 / (F.mse_loss(x_hat_sigmoid, data) + 1e-8))
                        psnr_total += psnr.item() * data.size(0)

                        # Simple SSIM approximation (variance-based similarity)
                        # For proper SSIM, would need sliding window computation
                        data_flat = data.view(data.size(0), -1)
                        recon_flat = x_hat_sigmoid.view(data.size(0), -1)

                        # Pearson correlation coefficient (ranges -1 to 1, closer to 1 is better)
                        data_mean = data_flat.mean(dim=1, keepdim=True)
                        recon_mean = recon_flat.mean(dim=1, keepdim=True)
                        data_centered = data_flat - data_mean
                        recon_centered = recon_flat - recon_mean
                        correlation = (data_centered * recon_centered).sum(dim=1) / (
                            torch.sqrt((data_centered**2).sum(dim=1)) * torch.sqrt((recon_centered**2).sum(dim=1)) + 1e-8
                        )
                        ssim_total += correlation.sum().item()

                        num_samples += data.size(0)

                n = len(val_loader)
                val_loss /= n
                val_recon /= n
                val_kl /= n

                # Average reconstruction metrics
                mse_avg = mse_total / num_samples
                mae_avg = mae_total / num_samples
                psnr_avg = psnr_total / num_samples
                ssim_avg = ssim_total / num_samples

                mu_all = torch.cat(mu_stats, dim=0)
                mu_mean = mu_all.mean().item()
                mu_std = mu_all.std().item()

                wandb.log({
                    "epoch": epoch + 1,
                    "val_mode": True,
                    "val_loss": val_loss,  # lower is better (overall objective)
                    "val_recon_loss": val_recon,  # lower is better (BCE reconstruction)
                    "val_kl_loss": val_kl,  # moderate is best (too low = underfitting, too high = posterior collapse)
                    "mu_mean": mu_mean,  # should be near 0 (well-regularized)
                    "mu_std": mu_std,  # should be near 1 (well-regularized)
                    "val_mse": mse_avg,  # lower is better (pixel error)
                    "val_mae": mae_avg,  # lower is better (absolute pixel error)
                    "val_psnr": psnr_avg,  # higher is better (>25-30 dB is good)
                    "val_correlation": ssim_avg,  # higher is better (closer to 1.0 = perfect)
                })

                # Log example images using EMA weights
                if (epoch + 1) % LOG_IMAGES_EVERY == 0:
                    log_example_images(model, val_loader.dataset, epoch + 1, spatial=spatial, n=5)

            # Checkpoint saving
            if (epoch + 1) % UPDATE_CURRENT_CHECKPOINT_EVERY == 0:
                run_name = wandb.run.name if (hasattr(wandb, 'run') and wandb.run) else f"{model.__class__.__name__}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"
                run_checkpoint_dir = os.path.join(CHECKPOINT_PATH, run_name)
                os.makedirs(run_checkpoint_dir, exist_ok=True)

                save_checkpoint(os.path.join(run_checkpoint_dir, 'current.pt'), model, optimizer, epoch + 1, ema=ema, wandb_run_id=wandb.run.id)

                if (epoch + 1) % CREATE_PERMANENT_CHECKPOINT_EVERY == 0:
                    save_checkpoint(os.path.join(run_checkpoint_dir, f'epoch_{epoch + 1}.pt'), model, optimizer, epoch + 1, ema=ema, wandb_run_id=wandb.run.id)

        # Save model after training completes
        try:
            # Organize by sweep or standalone
            if hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None:
                sweep_dir = wandb.run.sweep_id
            else:
                sweep_dir = "standalone"

            save_dir = os.path.join(MODEL_SAVE_PATH, sweep_dir)
            os.makedirs(save_dir, exist_ok=True)

            save_name = f"{wandb.run.name}.safetensors"
            save_path = os.path.join(save_dir, save_name)

            # Save model state dict
            save_file(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

            # Log the model path to wandb
            wandb.config.update({"model_save_path": save_path})
        except Exception as e:
            print(f"Failed to save model: {e}")
            # Continue anyway - don't fail the run just because save failed

if __name__ == '__main__':
    # checkpoint_path = r'../checkpoints/vae/fearless-deluge-32/current.pt'
    train(
        # checkpoint_path=checkpoint_path
    )
