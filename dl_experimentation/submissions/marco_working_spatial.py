## Sample submission file for the VAE + Flow leaderboard challenge
## Author: Scott H. Hawley, Oct 6 2025
import torch
import torch.nn as nn
from safetensors.torch import load_file
import gdown
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
from math import log2

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_skip=True, use_bn=True, act=nn.GELU, dropout=0.4, groups=1):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn, groups=groups)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn, groups=groups)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.use_skip, self.act = use_skip, act

    def forward(self, x):
        if self.use_skip: x0 = x
        out = self.act()(self.bn1(self.conv1(x)))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.bn2(self.conv2(out))
        if self.use_skip: out = out + x0
        return self.act()(out)

class InspoResNetVAEEncoder(nn.Module):
    """this makes a 1D vector of length latent_dim"""

    def __init__(self, in_channels, latent_dim=3, base_channels=32, blocks_per_level=2, use_skips=True, use_bn=True,
                 act=nn.GELU, groups=1, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        channels = [base_channels, base_channels * 2, base_channels * 4]  # len(channels) = num levels
        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act, dropout=dropout, groups=groups) for _ in range(blocks_per_level)]) for ch in
             channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, 2 * latent_dim)
        self.act = act
        self.groups = groups
        self.dropout = dropout

    def forward(self, x):
        x = self.act()(self.bn1(self.conv1(x)))
        for i in range(len(self.levels)):
            if i > 0:  # shrink down
                x = F.avg_pool2d(x, 2)
                x = self.transitions[i - 1](x)
            for block in self.levels[i]:
                x = block(x)
        x = self.global_avg_pool(x)
        x = self.fc(x.flatten(start_dim=1))
        mean, logvar = x.chunk(2, dim=1)  # mean and log variance
        return mean, logvar

class InspoResNetVAEEncoderSpatial(nn.Module):
    """does dynamic ish downsampling with some adaptive pooling at the end, change input size from 28 to higher for other datasets"""

    def __init__(self, in_channels, latent_shape=(1,7,7), base_channels=32, blocks_per_level=2, use_skips=True,
                 use_bn=True, act=nn.GELU, groups=1, dropout=0.4, input_size=28):
        super().__init__()
        # Take in latent shape
        assert len(latent_shape) == 3

        latent_ch, latent_h, latent_w = latent_shape
        n_downsamples = int(log2(input_size / latent_h))

        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()

        channels = [base_channels * (2**i) for i in range(n_downsamples + 1)]

        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act, dropout=dropout, groups=groups) for _ in
                            range(blocks_per_level)]) for ch in channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((latent_h, latent_w))  # adapt to shape
        self.channel_proj = nn.Conv2d(in_channels=channels[-1], out_channels=2 * latent_shape[0], kernel_size=1)  # for vae (mu, logvar)
        self.act = act

    def forward(self, x):
        x = self.act()(self.bn1(self.conv1(x)))
        for i in range(len(self.levels)):
            if i > 0:  # shrink down
                x = F.avg_pool2d(x, 2)
                x = self.transitions[i - 1](x)
            for block in self.levels[i]:
                x = block(x)
        x = self.adaptive_pool(x)
        x = self.channel_proj(x)

        mean, logvar = x.chunk(2, dim=1)  # mean and log variance
        return mean, logvar

class InspoResNetVAEDecoderSpatial(nn.Module):
    """this is just the mirror image of InspoResNetVAEEnoderSpatial"""

    def __init__(self, out_channels, latent_shape=(1, 7, 7), base_channels=32, blocks_per_level=2, use_skips=True,
                 use_bn=True, act=nn.GELU, groups=1, dropout=0.4, input_size=28):
        super().__init__()
        # Take in latent shape
        assert len(latent_shape) == 3

        latent_ch, latent_h, latent_w = latent_shape
        n_downsamples = int(log2(input_size / latent_h))
        self.size_before_adaptive = input_size // (2 ** n_downsamples)

        channels = [base_channels * (2**i) for i in range(n_downsamples + 1)][::-1]  # reversed from encoder
        self.channels = channels
        self.channel_proj = nn.Conv2d(in_channels=latent_ch, out_channels=channels[0], kernel_size=1)  # for vae (mu, logvar)

        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act, dropout=dropout, groups=groups) for _ in
                            range(blocks_per_level)]) for ch in channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.act = act

    def forward(self, z):
        x = self.channel_proj(z)
        x = F.interpolate(x, size=(self.size_before_adaptive, self.size_before_adaptive),
                      mode='bilinear', align_corners=False)
        for i in range(len(self.levels)):
            for block in self.levels[i]:
                x = block(x)
            if i < len(self.levels) - 1:  # not last level
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self.transitions[i](x)
        return self.final_conv(x)

class InspoResNetVAEDecoder(nn.Module):
    """this is just the mirror image of ResNetVAEEnoder""" # honestly same as scotss init but with literally parameter changes, at least spatial was diff haha

    def __init__(self, out_channels, latent_dim=3, base_channels=32, blocks_per_level=2, use_skips=True, use_bn=True,
                 act=nn.GELU, groups=1, dropout=0.4):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4][::-1]  # reversed from encoder
        self.channels = channels
        self.start_dim = 7  # starting spatial dimension
        self.fc = nn.Linear(latent_dim, channels[0] * self.start_dim * self.start_dim)  # 128 * 16  # starting size
        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act, dropout=dropout, groups=groups) for _ in range(blocks_per_level)]) for ch in
             channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.act = act
        self.groups = groups
        self.dropout = dropout

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels[0], self.start_dim, self.start_dim)  # project to spatial
        for i in range(len(self.levels)):
            for block in self.levels[i]:
                x = block(x)
            if i < len(self.levels) - 1:  # not last level
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self.transitions[i](x)
        return self.final_conv(x)

class InspoResNetVAE(nn.Module):
    """Inspo version of a Main VAE class"""
    def __init__(self, latent_shape=(3,), act=nn.GELU, use_skips=True, use_bn=True, base_channels=32, blocks_per_level=3, groups=1, dropout=0.4):
        super().__init__()
        if isinstance(latent_shape, int):
            latent_shape = (latent_shape,)

        self.channels = 1  # Keep 1 for MNIST (grey scale)

        if len(latent_shape) == 1:
            use_spatial = False
            latent_dim = latent_shape[0]
        elif len(latent_shape) == 2:
            use_spatial = True
            latent_dim = latent_shape[0] * latent_shape[1]
            latent_shape = (self.channels,) + latent_shape  # Add dim 1 for channel mnist
        elif len(latent_shape) == 3:
            use_spatial = True
            latent_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]
        else:
            raise NotImplementedError("Have not thought about this yet")

        self.latent_dim = latent_dim
        self.latent_shape = latent_shape
        self.act = act
        self.use_skips = use_skips
        self.use_bn = use_bn
        self.base_channels = base_channels
        self.blocks_per_level = blocks_per_level
        self.groups = groups
        self.dropout = dropout

        if use_spatial:
            self.encoder = InspoResNetVAEEncoderSpatial(
                in_channels=self.channels, latent_shape=self.latent_shape, base_channels=self.base_channels,
                blocks_per_level=self.blocks_per_level,
                use_skips=self.use_skips, use_bn=self.use_bn, act=self.act, groups=self.groups, dropout=self.dropout)
            self.decoder = InspoResNetVAEDecoderSpatial(
                out_channels=self.channels, latent_shape=self.latent_shape, base_channels=self.base_channels,
                blocks_per_level=self.blocks_per_level,
                use_skips=self.use_skips, use_bn=self.use_bn, act=self.act, groups=self.groups, dropout=self.dropout)
        else:
            self.encoder = InspoResNetVAEEncoder(
                in_channels=self.channels, latent_dim=self.latent_dim, base_channels=self.base_channels,
                blocks_per_level=self.blocks_per_level,
                use_skips=self.use_skips, use_bn=self.use_bn, act=self.act, groups=self.groups, dropout=self.dropout)
            self.decoder = InspoResNetVAEDecoder(
                out_channels=self.channels, latent_dim=self.latent_dim, base_channels=self.base_channels,
                blocks_per_level=self.blocks_per_level,
                use_skips=self.use_skips, use_bn=self.use_bn, act=self.act, groups=self.groups, dropout=self.dropout)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = torch.cat([mu, log_var], dim=1)
        z_hat = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        x_hat = self.decoder(z_hat)
        return z, x_hat, mu, log_var, z_hat

    @property
    def config(self):
        """Return model configuration as dict for logging"""
        return {
            "latent_dim": self.latent_dim,
            "latent_shape": str(self.latent_shape),
            "act": self.act.__name__ if hasattr(self.act, '__name__') else str(self.act),
            "use_skips": self.use_skips,
            "use_bn": self.use_bn,
            "base_channels": self.base_channels,
            "blocks_per_level": self.blocks_per_level,
            "groups": self.groups,
            "dropout": self.dropout,
            "channels": self.channels,
            "model_class": self.__class__.__name__
        }

@torch.no_grad()
def test_inference(model, test_ds, idx=None, return_fig=False, in_train=False):
    device = next(model.parameters()).device
    model.eval()
    if idx is None: idx = torch.randint(len(test_ds), (1,))[0]
    if isinstance(idx, int): idx = [idx]
    elif isinstance(idx, range): idx = list(idx)
    x_batch = torch.stack([test_ds[i][0] for i in idx]).to(device)  # images
    y_batch = torch.tensor([test_ds[i][1] for i in idx]).to(device) # labels
    result = model.forward(x_batch)
    z, recon = result[:2]
    recon = torch.sigmoid(recon.view(len(idx), 28, 28))
    fig, axs = plt.subplots(2, len(idx), figsize=(3*len(idx), 4))
    if len(idx) == 1: axs = axs.reshape(2, 1)
    for i in range(len(idx)):
        axs[0,i].imshow(x_batch[i].view(28,28).cpu(), cmap='gray')
        axs[1,i].imshow(recon[i].cpu(), cmap='gray')
        if i == 0:
            axs[0,0].set_ylabel('Input', fontsize=12)
            axs[1,0].set_ylabel('Reconstruction', fontsize=12)
    model.train()
    if return_fig: return fig

    if not in_train: plt.show()  # Second security

@torch.no_grad()
def test_inference_spatial(model, test_ds, idx=None, return_fig=False, in_train=False):
    device = next(model.parameters()).device
    model.eval()
    if idx is None: idx = torch.randint(len(test_ds), (1,))[0]
    if isinstance(idx, int): idx = [idx]
    elif isinstance(idx, range): idx = list(idx)

    x_batch = torch.stack([test_ds[i][0] for i in idx]).to(device)  # images
    y_batch = torch.tensor([test_ds[i][1] for i in idx]).to(device) # labels
    result = model.forward(x_batch)
    z, recon, mu, log_var, z_hat = result[:5]
    recon = torch.sigmoid(recon.view(len(idx), 28, 28))

    # Normalize mu for visualization
    mu=mu.squeeze()
    mu_flat = mu.view(len(idx),-1)
    mu_min = mu_flat.min(dim=1, keepdim=True)[0]
    mu_max = mu_flat.max(dim=1, keepdim=True)[0]
    mu_norm = ((mu_flat - mu_min) / (mu_max - mu_min + 1e-8)).view_as(mu)

    fig, axs = plt.subplots(3, len(idx), figsize=(3*len(idx), 6))
    if len(idx) == 1: axs = axs.reshape(3, 1)

    for i in range(len(idx)):
        axs[0,i].imshow(x_batch[i].view(28,28).cpu(), cmap='gray')
        axs[1,i].imshow(mu_norm[i].cpu(), cmap='viridis')  # middle row: mu
        axs[2,i].imshow(recon[i].cpu(), cmap='gray')
        if i == 0:
            axs[0,0].set_ylabel('Input', fontsize=12)
            axs[1,0].set_ylabel('Latent z_μ', fontsize=12)
            axs[2,0].set_ylabel('Reconstruction', fontsize=12)

    model.train()
    if return_fig: return fig
    if not in_train: plt.show()  # Second security

def log_example_images(model, test_ds, epoch, spatial=True, n=5):
    if wandb.run is None:
        return
    fig = (test_inference_spatial if spatial else test_inference)(
        model, test_ds, idx=range(n), return_fig=True, in_train=True
    )
    wandb.log({"reconstructions": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act=nn.SiLU, use_skip=True, groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1)  # 1x1 conv for channel mixing
        self.act = act
        self.use_skip = use_skip and (in_ch == out_ch)
        # nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.use_skip: x0 = x
        out = self.act()(self.conv1(x))
        out = self.conv2(out)
        if self.use_skip: out = out + x0
        return self.act()(out)

class ConvFlowNet(nn.Module):
    def __init__(self, latent_ch, hidden=32, depth=3, grow=True, time_embed_dim=32, act=nn.SiLU, use_skip=True, groups=1):
        super().__init__()

        # Store config
        self.latent_ch = latent_ch
        self.hidden = hidden
        self.depth = depth
        self.grow = grow
        self.time_embed_dim = time_embed_dim
        self.act = act
        self.use_skip = use_skip
        self.groups = groups

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            act(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Input conv now takes latent + time embedding channels
        self.in_conv = nn.Conv2d(latent_ch + time_embed_dim, hidden, 3, padding=1)
        channels = [hidden * (2**i) for i in range(depth)] if grow else [hidden] * depth

        self.blocks = nn.Sequential(
            *[ConvBlock(in_ch, out_ch, act=act, use_skip=use_skip, groups=groups)
              for in_ch, out_ch in zip(channels[:-1], channels[1:])]
        )

        self.out_conv = nn.Conv2d(channels[-1], latent_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z, t):
        # z: [B,C,H,W], t: scalar, [B], or [B,1]
        B, C, H, W = z.shape
        t = t.expand(B, 1)  # Handle all cases -> [B, 1]
        t_embed = self.time_embed(t)  # [B, time_embed_dim]

        # Broadcast time embedding to spatial dimensions
        t_img = t_embed.view(B, -1, 1, 1).expand(B, -1, H, W)  # [B, time_embed_dim, H, W]

        h = F.silu(self.in_conv(torch.cat([z, t_img], dim=1)))
        h = self.blocks(h)
        return self.out_conv(h)

    @property
    def config(self):
        """Return model configuration as dict for logging"""
        return {
            "latent_ch": self.latent_ch,
            "hidden": self.hidden,
            "depth": self.depth,
            "grow": self.grow,
            "time_embed_dim": self.time_embed_dim,
            "use_skip": self.use_skip,
            "groups": self.groups,
            "act": self.act.__name__ if hasattr(self.act, '__name__') else str(self.act),
            "model_class": self.__class__.__name__
        }



@torch.no_grad()
def rk4_step(f, y, t, dt):
    # f: callable (y, t) -> dy/dt  (aka model forward haha)
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

@torch.no_grad()  # from blog :)
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


class SubmissionInterface(nn.Module):
    """All teams must implement this for automated evaluation.
    When you subclass/implement these methods, replace the NotImplementedError."""

    def __init__(self):
        super().__init__()

        # --- REQUIRED INFO:
        self.specifications = dict(
            latent_shape=(1, 7, 7),
            base_channels=32,
            blocks_per_level=1,
            groups=4,
            dropout=0.294905,
            act=torch.nn.SiLU
        )
        self.spatial = True
        self.latent_shape = (1, 7, 7)
        torch.mps.empty_cache()

        self.info = {
            'team': 'marco_test_spat_work',
            'names': 'Marco',
        }
        # ----

        # keep support for full auto-initialization:
        self.device = 'mps'
        self.load_vae()
        self.load_flow_model()

    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
            downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.vae = InspoResNetVAE(**self.specifications)
        vae_weights_file = 'downloaded_vae.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1KoYShNvcdIQXVA66-MjI4CUv-wl6VzES/view?usp=drive_link"
        gdown.download(safetensors_link, vae_weights_file, quiet=False, fuzzy=True)
        self.vae.load_state_dict(load_file(vae_weights_file))
        # self.vae.load_state_dict(load_file(r'../models/safetensors/vae/sp_vae.safetensors'))

    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.flow_model = ConvFlowNet(latent_ch=1, hidden=64, depth=3,
                                        grow=True, time_embed_dim=16, use_skip=True, groups=4)
        flow_weights_file = 'downloaded_flow.safetensors'
        safetensors_link = "https://drive.google.com/file/d/17qyDKU5PgNuIW0OE8HzZUekqcfbOgpXz/view?usp=drive_link"
        gdown.download(safetensors_link, flow_weights_file, quiet=False, fuzzy=True)
        self.flow_model.load_state_dict(load_file(flow_weights_file))
        # self.flow_model.load_state_dict(load_file(r'../models/safetensors/flow/sp_flow_7x7.safetensors'))

    def generate_samples(self, n_samples:int, n_steps=10) -> torch.Tensor:
        self.flow_model.eval(); self.vae.eval()
        z0 = torch.randn(n_samples, *self.latent_shape, device=self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps, latent_2d=True)
        return F.sigmoid(self.decode(z1)).squeeze(1)
        # return F.sigmoid(self.decode(z1)).view(-1, 28, 28)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # if your vae has linear layers, flatten first
        # if your vae has conv layers, comment out next line
        # images = images.view(images.size(0), -1)
        with torch.no_grad():
            z = self.vae.encoder(images.to(self.device))
            # mu = z[:, :self.latent_dim]  # return only first half (mu)
            if isinstance(z, (tuple, list)):
                mu, _ = z
            else:
                mu = z
            return mu

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(latents)

    def to(self, device):
        self.device = device
        self.vae.to(self.device)
        self.flow_model.to(self.device)
        return self

    # Sample usage:
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# mysub = SubmissionInterface().to(device) # loads vae and flow models
# xhat_gen = mysub.generate_samples(n_samples=10, n_steps=100)
