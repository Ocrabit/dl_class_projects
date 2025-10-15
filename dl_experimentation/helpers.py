import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os
from numpy.lib.format import open_memmap
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn
from safetensors.torch import save_file, load_file
from typing import Optional, Dict, Any

class EncodedDataset(Dataset):
    def __init__(self, dataset_path, split="train", add_channel=True, mmap=True):
        """
        General dataset loader for encoded .npy latent datasets.

        Args:
            dataset_path (str): Path to dataset folder, e.g. '../data/ResNetVAEEncoderSpatial/MNIST'
            split (str): 'train' or 'test'
            add_channel (bool): Whether to add a channel dim (for (7,7) → (1,7,7))
            mmap (bool): Whether to memory-map large files instead of loading into RAM

        Path Structure
        <encoder_name>/
         └── <dataset_name>/
              ├── train/
              │    ├── data.npy
              │    └── target.npy
              └── test/
                   ├── data.npy
                   └── target.npy
        """
        self.dataset_path = dataset_path
        self.split = split
        self.add_channel = add_channel

        # Expected structure: <dataset_path>/<split>/data.npy + target.npy
        base = os.path.join(dataset_path, split)
        self.data_path = os.path.join(base, "data.npy")
        self.target_path = os.path.join(base, "target.npy")

        if not os.path.exists(self.data_path) or not os.path.exists(self.target_path):
            raise FileNotFoundError(f"Expected data.npy and target.npy in: {base}")

        mmap_mode = "r" if mmap else None
        self.data = np.load(self.data_path, mmap_mode=mmap_mode)
        self.targets = np.load(self.target_path, mmap_mode=mmap_mode)

        assert len(self.data) == len(self.targets), "Data and targets must be same length"

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.add_channel and x.ndim == 2:
            x = np.expand_dims(x, 0)  # (1, H, W)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y

def load_encoded_dataset(base_dir, add_channel=True, mmap=True):
    """
    Automatically loads both train and test splits from an encoded dataset.

    Args:
        base_dir (str): Base directory path, e.g. 'data/ResNetVAEEncoderSpatial/MNIST'
        add_channel (bool): Whether to add (1, H, W) dimension
        mmap (bool): Use memory-mapped loading (recommended for large datasets)

    Returns:
        (train_dataset, test_dataset)
    """
    train_ds = EncodedDataset(base_dir, split="train", add_channel=add_channel, mmap=mmap)
    test_ds = EncodedDataset(base_dir, split="test",  add_channel=add_channel, mmap=mmap)
    return train_ds, test_ds

@torch.no_grad()
def get_latent_dtype_dim(encoder, loader, device=None):
    encoder.eval()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    x, _ = next(iter(loader))
    x = x.to(device, dtype=torch.float32)
    mu, _ = encoder(x)
    return mu.dtype, tuple(mu.squeeze(1).shape[1:])

@torch.no_grad()
def encode_dataset(model, train_dataloader, test_dataloader, dir_='data/', device=None, suffix: str = '', ema=None):
    # Some cheap checks
    assert hasattr(model, "encoder"), "Model must have an encoder attribute"
    assert model.encoder is not None, "Model encoder cannot be None"

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    encoder = model.encoder
    encoder.eval()

    model_dir = model.encoder.__class__.__name__ + (suffix if suffix[0] == '_' else '_' + suffix)

    # Use EMA weights if provided
    context = ema.average_parameters() if ema is not None else torch.no_grad()

    with context:
        for loader, mode in [(train_dataloader, 'train'), (test_dataloader, 'test')]:
            print(f"Processing {mode} split...")
            dataset_dir = loader.dataset.__class__.__name__
            base_dir = os.path.join(dir_, model_dir, dataset_dir, mode)
            os.makedirs(base_dir, exist_ok=True)

            latent_torch_dtype, latent_img_dim = get_latent_dtype_dim(encoder, loader)
            latent_torch_dtype = str(latent_torch_dtype).replace("torch.", "")
            dtype_target = str(loader.dataset.targets.dtype).replace("torch.", "")
            N = loader.dataset.data.shape[0]

            data_mm = open_memmap(os.path.join(base_dir, "data.npy"),   mode="w+", dtype=latent_torch_dtype, shape=(N, *latent_img_dim))
            target_mm = open_memmap(os.path.join(base_dir, "target.npy"), mode="w+", dtype=dtype_target, shape=(N,))

            i = 0
            for data, target in tqdm(loader, desc=f"{dataset_dir} {mode}"):
                data = data.to(device)

                mu, _ = encoder(data)
                mu = mu.squeeze(1)

                b = mu.size(0)
                data_mm[i:i+b] = mu.detach().cpu().numpy()
                target_mm[i:i+b] = target.detach().cpu().numpy()
                i += b

            # Finish write
            data_mm.flush()
            target_mm.flush()

# Flow Helpers
@torch.no_grad()
def fwd_euler_step(model, current_points, current_t, dt):
    velocity = model(current_points, current_t)
    return current_points + velocity * dt

@torch.no_grad()
def rk4_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3,       t + dt)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

@torch.no_grad()
def integrate_path(model, initial_points, step_fn=rk4_step, n_steps=100, warp_fn=None, latent_2d=False):
    p = next(model.parameters())
    device, model_dtype = p.device, p.dtype

    current_points = initial_points.to(device=device, dtype=model_dtype).clone()
    model.eval()

    ts = torch.linspace(0, 1, n_steps, device=device, dtype=model_dtype)
    if warp_fn: ts = warp_fn(ts)
    if latent_2d: t_batch = torch.empty((current_points.shape[0], 1), device=device, dtype=model_dtype)

    for i in range(len(ts) - 1):
        t, dt = ts[i], ts[i + 1] - ts[i]
        if latent_2d: t = t_batch.fill_(t.item())

        current_points = step_fn(model, current_points, t, dt)

    return current_points

@torch.no_grad()
def warp_time(t, dt=None, s=.5):
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s < 0 or s > 1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4 * (1 - s) * t ** 3 + 6 * (s - 1) * t ** 2 + (3 - 2 * s) * t
    if dt:  # warped time-step requested; use derivative
        return tw, dt * 12 * (1 - s) * t ** 2 + 12 * (s - 1) * t + (3 - 2 * s)
    return tw

@torch.no_grad()
def plot_latent_space(model, dataloader, n_samples=2000, use_3d=False,
                      reducer="tsne", device=None, point_size=4, opacity=0.7, template="plotly_dark"):
    """Visualize (V)AE latent space with Plotly.
    - Flattens spatial latents to vectors.
    - Options: `pca`/`tsne` down to 2D/3D.
    """
    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    zs, labels = [], []
    total = 0

    for x_batch, y_batch in dataloader:
        if total >= n_samples:
            break

        x_batch = x_batch.to(device)

        # 1) encode -> get mu if available
        enc_out = model.encoder(x_batch)
        if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 2:
            mu, logvar = enc_out[:2]
            z = mu
        else:
            z = enc_out  # AE or custom encoder returning a single tensor

        # 2) flatten if spatial
        if z.dim() > 2:  # [B, C, H, W] -> [B, C*H*W]
            z = z.view(z.size(0), -1)

        zs.append(z.detach().cpu())
        labels.append(y_batch.detach().cpu())
        total += z.size(0)

    if not zs:
        print("No batches found—check dataloader or n_samples.")
        if was_training: model.train()
        return

    zs = torch.cat(zs, dim=0)[:n_samples].numpy()
    labels = torch.cat(labels, dim=0)[:n_samples].numpy()
    if labels.ndim > 1:
        labels = np.squeeze(labels)

    out_dim = 3 if use_3d else 2

    # 3) dimensionality reduction
    title_suffix = ""
    if zs.shape[1] > out_dim:
        if reducer.lower() == "tsne":
            from sklearn.manifold import TSNE
            zs_dr = TSNE(n_components=out_dim, perplexity=30, learning_rate=200).fit_transform(zs)
            title_suffix = " (t-SNE)"
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=out_dim)
            zs_dr = pca.fit_transform(zs)
            ev = getattr(pca, "explained_variance_ratio_", None)
            if ev is not None:
                title_suffix = f" (PCA, explained var={ev.sum():.2f})"
    else:
        # already 2D/3D
        zs_dr = zs
        # if asking 3D but dims==2, pad a zero z for nicer 3D view
        if use_3d and zs_dr.shape[1] == 2:
            zs_dr = np.concatenate([zs_dr, np.zeros((zs_dr.shape[0], 1))], axis=1)
            title_suffix = " (padded z=0)"

    if use_3d:
        fig = px.scatter_3d(
            x=zs_dr[:, 0],
            y=zs_dr[:, 1],
            z=zs_dr[:, 2],
            color=labels.astype(str),  # categorical coloring
            opacity=opacity,
            template=template
        )
        fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
        fig.update_layout(
            width=900, height=650,
            scene=dict(
                xaxis_title="z1", yaxis_title="z2", zaxis_title="z3",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            title=f"Latent Space{title_suffix}"
        )
    else:
        fig = px.scatter(
            x=zs_dr[:, 0],
            y=zs_dr[:, 1],
            color=labels.astype(str),
            opacity=opacity,
            template=template
        )
        fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
        fig.update_layout(
            width=850, height=650,
            xaxis_title="z1", yaxis_title="z2",
            margin=dict(l=0, r=0, t=50, b=0),
            title=f"Latent Space{title_suffix}"
        )

    fig.show()
    if was_training:
        model.train()

def plot_generated(gen_xhat, nrow=5):
    gen_xhat = gen_xhat.detach().cpu()

    # assume shape (N, 28, 28)
    n_samples = gen_xhat.size(0)
    ncol = nrow
    nrow = (n_samples + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*2, nrow*2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_samples:
            ax.imshow(gen_xhat[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def create_model_suffix(specifications):
    """Create suffix from specs dict (extracts latent_shape or latent_dim). E.g. {"latent_shape": (1,7,7), "base_channels": 16, "activation": "GELU"} -> "l7x7_ba16_acGELU" """
    # Extract latent shape/dim from specifications
    if 'latent_shape' in specifications:
        latent_shape = specifications['latent_shape']
    elif 'latent_dim' in specifications:
        latent_shape = (specifications['latent_dim'],)
    else:
        raise ValueError("specifications must contain either 'latent_shape' or 'latent_dim'")

    # Format latent string
    if len(latent_shape) == 3:
        latent_str = 'x'.join(map(str, latent_shape[1:]))  # last 2 dims
    elif len(latent_shape) == 1:
        latent_str = str(latent_shape[0])
    else:  # len==2
        latent_str = 'x'.join(map(str, latent_shape))

    # Create spec string (excluding latent_shape/latent_dim since we already used it)
    spec_items = {}
    for k, v in specifications.items():
        if k in ['latent_shape', 'latent_dim']:
            continue

        # Handle any value that's a class or has "<class" in string
        if isinstance(v, type):  # e.g., nn.GELU class
            v = v.__name__
        elif isinstance(v, str) and "<class" in v:  # e.g., "<class 'torch.nn.modules.activation.GELU'>"
            v = v.split(".")[-1].rstrip("'>")
        elif hasattr(v, '__class__') and not isinstance(v, (str, int, float, bool)):  # instance
            v = v.__class__.__name__

        spec_items[k] = v

    spec_str = '_'.join([f"{k[:2]}{v}" for k, v in spec_items.items()])
    return f"l{latent_str}_{spec_str}"


def comparative_generate_samples(flow1, flow2, vae, latent_shape: tuple, n_samples: int, n_steps=100, latent_2d=False) -> torch.Tensor:
    # For shape think (1, latent_dim, latent_dim) or (latent_dim,)
    device = next(flow1.parameters()).device
    flow1.eval(); flow2.eval(); vae.eval()
    z0 = torch.randn(n_samples, *latent_shape, device=device)

    z1 = integrate_path(flow1, z0.clone(), n_steps=n_steps, step_fn=rk4_step, latent_2d=True)
    z2 = integrate_path(flow2, z0.clone(), n_steps=n_steps, step_fn=rk4_step, latent_2d=True)

    x1 = torch.sigmoid(vae.decoder(z1)).view(-1, 28, 28)
    x2 = torch.sigmoid(vae.decoder(z2)).view(-1, 28, 28)
    return x1, x2

class BasicModel(nn.Module):
    def __init__(self, vae, flow_model, latent_shape: tuple):
        super().__init__()

        self.latent_shape = latent_shape

        self.vae = vae
        self.flow_model = flow_model

        self.device = 'mps'

    def generate_samples(self, n_samples: int, n_steps=100, step_fn=rk4_step) -> torch.Tensor:
        self.flow_model.eval();
        self.vae.eval()
        z0 = torch.randn(n_samples, *self.latent_shape, device=self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps, step_fn=step_fn, latent_2d=True)
        return F.sigmoid(self.vae.decoder(z1)).squeeze(1)
        # return F.sigmoid(self.decode(z1)).view(-1, 28, 28)

    def to(self, device):
        self.device = device
        self.vae.to(self.device)
        self.flow_model.to(self.device)
        return self


# ============================================================================
# Checkpoint Helpers
# ============================================================================

def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ema: Optional[Any] = None,
    **extra_state
) -> None:
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()

    checkpoint.update(extra_state)

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path} (epoch {epoch})")


def load_checkpoint(
    load_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load checkpoint and resume from saved epoch. Returns checkpoint dict."""
    if device is None:
        device = next(model.parameters()).device

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    print(f"Checkpoint loaded: {load_path} (epoch {checkpoint['epoch']})")
    return checkpoint
