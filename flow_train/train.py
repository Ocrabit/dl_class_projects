import os
import sys
import random
import string

# Add parent directory to path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import torch
import wandb
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from safetensors.torch import save_file

from helpers import integrate_path, rk4_step, warp_time, save_checkpoint, load_checkpoint

# Paths
MODEL_SAVE_PATH = os.path.join(project_dir, 'models/safetensors/flow/')
CHECKPOINT_PATH = os.path.join(project_dir, 'checkpoints/flow/')
UPDATE_CURRENT_CHECKPOINT_EVERY = 50
CREATE_PERMANENT_CHECKPOINT_EVERY = 200
VALIDATE_EVERY = 5  # Run validation every N epochs

# Default config
default_config = dict(
    learning_rate=1e-3,
    weight_decay=0.0,
    epochs=50,
    batch_size=128,
    n_steps=100,
    use_time_warp=False,
    reflow_every=0,  # 0 = no reflow, >0 = reflow every N steps
)


def train(model, train_loader, val_loader, config=None, project='flow_experimentation', checkpoint_path=None, pretrained_model=None):
    # Get run ID from checkpoint if resuming
    if checkpoint_path:
        resume_id = torch.load(checkpoint_path, map_location='cpu').get('wandb_run_id')
    else:
        resume_id = None

    # Use default config if none provided
    if config is None and not resume_id:
        config = default_config

    # Initialize wandb
    with wandb.init(project=project, config=config, id=resume_id, resume='must' if resume_id else False) as run:
        if resume_id and config is not None:
            wandb.config.update(config, allow_val_change=True)
        config = wandb.config

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)

        # Move pretrained model to device if provided
        if pretrained_model is not None:
            pretrained_model = pretrained_model.to(device)
            pretrained_model.eval()

        print(f'Using device: {device}')

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        wandb.config.update({
            "total_params": total_params,
            "model_class": model.__class__.__name__
        })
        print(f"Training {model.__class__.__name__} with {total_params:,} parameters")

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

        # Load checkpoint if provided
        start_epoch = 0
        if checkpoint_path is not None:
            ckpt = load_checkpoint(checkpoint_path, model, optimizer, ema=ema, device=device)
            start_epoch = ckpt['epoch']
            print(f"Resuming from epoch {start_epoch}")

        # Check if spatial
        sample_batch = next(iter(train_loader))[0]
        spatial = sample_batch.dim() > 2
        print(f"Latent data shape: {sample_batch.shape}, spatial: {spatial}")

        warp_fn = warp_time if config.use_time_warp else None
        global_step = 0

        for epoch in range(start_epoch, config.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)

            for batch_idx, (data, _) in enumerate(pbar):
                optimizer.zero_grad()

                target_x = data.to(device)
                sampled_x = torch.randn_like(target_x)
                B = sampled_x.size(0)

                # Reflow if configured - use pretrained model if provided, otherwise use current model
                if config.reflow_every > 0 and global_step % config.reflow_every == 0 and global_step > 0:
                    reflow_model = pretrained_model if pretrained_model is not None else model
                    with torch.no_grad():
                        target_x = integrate_path(reflow_model, sampled_x, step_fn=rk4_step, n_steps=20, latent_2d=spatial)

                t = torch.rand(B, 1, device=device, dtype=target_x.dtype)
                if warp_fn:
                    t = warp_fn(t)

                if spatial:
                    t_expanded = t.view(B, 1, 1, 1)
                    interpolated_x = sampled_x * (1 - t_expanded) + target_x * t_expanded
                else:
                    interpolated_x = sampled_x * (1 - t) + target_x * t

                line_directions = target_x - sampled_x
                drift = model(interpolated_x, t)
                loss = criterion(drift, line_directions)

                loss.backward()
                optimizer.step()
                ema.update()

                with torch.no_grad():
                    cos_sim = F.cosine_similarity(drift, line_directions, dim=1).mean()

                wandb.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": loss.item(),
                    "cos_sim": cos_sim.item(),
                    "drift_norm": drift.norm(dim=1).mean().item(),
                })
                global_step += 1
                pbar.set_postfix(Loss=f"{loss.item():.4f}", CosSim=f"{cos_sim.item():.3f}")

            # Validation with EMA
            if (epoch + 1) % VALIDATE_EVERY == 0:
                model.eval()
                with ema.average_parameters():
                    val_loss = 0.0
                    val_cos_sim = 0.0
                    n_batches = 0

                    with torch.no_grad():
                        for data, _ in val_loader:
                            target_x = data.to(device)
                            sampled_x = torch.randn_like(target_x)
                            B = sampled_x.size(0)

                            t = torch.rand(B, 1, device=device, dtype=target_x.dtype)
                            if warp_fn:
                                t = warp_fn(t)

                            if spatial:
                                t_expanded = t.view(B, 1, 1, 1)
                                interpolated_x = sampled_x * (1 - t_expanded) + target_x * t_expanded
                            else:
                                interpolated_x = sampled_x * (1 - t) + target_x * t

                            line_directions = target_x - sampled_x
                            drift = model(interpolated_x, t)
                            loss = criterion(drift, line_directions)
                            cos_sim = F.cosine_similarity(drift, line_directions, dim=1).mean()

                            val_loss += loss.item()
                            val_cos_sim += cos_sim.item()
                            n_batches += 1

                    wandb.log({
                        "epoch": epoch + 1,
                        "val_loss": val_loss / n_batches,
                        "val_cos_sim": val_cos_sim / n_batches,
                    })

            # Checkpoint saving
            if (epoch + 1) % UPDATE_CURRENT_CHECKPOINT_EVERY == 0:
                run_name = wandb.run.name if (hasattr(wandb, 'run') and wandb.run) else f"{model.__class__.__name__}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=4))}"
                run_checkpoint_dir = os.path.join(CHECKPOINT_PATH, run_name)
                os.makedirs(run_checkpoint_dir, exist_ok=True)

                save_checkpoint(os.path.join(run_checkpoint_dir, 'current.pt'), model, optimizer, epoch + 1, ema=ema, wandb_run_id=wandb.run.id)

                if (epoch + 1) % CREATE_PERMANENT_CHECKPOINT_EVERY == 0:
                    save_checkpoint(os.path.join(run_checkpoint_dir, f'epoch_{epoch + 1}.pt'), model, optimizer, epoch + 1, ema=ema, wandb_run_id=wandb.run.id)

        # Save final model
        try:
            if hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id is not None:
                sweep_dir = wandb.run.sweep_id
            else:
                sweep_dir = "standalone"

            save_dir = os.path.join(MODEL_SAVE_PATH, sweep_dir)
            os.makedirs(save_dir, exist_ok=True)

            save_name = f"{wandb.run.name}.safetensors"
            save_path = os.path.join(save_dir, save_name)

            save_file(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

            wandb.config.update({"model_save_path": save_path})
        except Exception as e:
            print(f"Failed to save model: {e}")

    return model


if __name__ == '__main__':
    # Example usage
    from models import ConvFlowNet
    from helpers import load_encoded_dataset

    train_ds, test_ds = load_encoded_dataset('data/InspoResNetVAEEncoderSpatial_l7x7/MNIST')
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    flow_model = ConvFlowNet(latent_ch=1, hidden=32, depth=3, grow=True)

    train(flow_model, train_loader, val_loader)