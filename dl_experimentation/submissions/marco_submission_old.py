## Sample submission file for the VAE + Flow leaderboard challenge
## Author: Scott H. Hawley, Oct 6 2025
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import gdown
import os


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

class InspoResNetVAEDecoder(nn.Module):
    """this is just the mirror image of ResNetVAEEnoder"""

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
    """Main VAE class"""
    def __init__(self, latent_dim=3, act=nn.GELU, use_skips=True, use_bn=True, base_channels=32, blocks_per_level=3, groups=1, dropout=0.4):
        super().__init__()

        self.latent_dim = latent_dim
        self.act = act
        self.use_skips = use_skips
        self.use_bn = use_bn
        self.base_channels = base_channels
        self.blocks_per_level = blocks_per_level
        self.groups = groups
        self.dropout = dropout

        self.channels = 1  # Keep 1 for MNIST (grey scale)

        self.encoder = InspoResNetVAEEncoder(
                        in_channels=self.channels, latent_dim=self.latent_dim, base_channels=self.base_channels, blocks_per_level=self.blocks_per_level,
                        use_skips=self.use_skips, use_bn=self.use_bn, act=self.act, groups=self.groups, dropout=self.dropout)
        self.decoder = InspoResNetVAEDecoder(
                        out_channels=self.channels, latent_dim=self.latent_dim, base_channels=self.base_channels, blocks_per_level=self.blocks_per_level,
                        use_skips=self.use_skips, use_bn=self.use_bn, act=self.act, groups=self.groups, dropout=self.dropout)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = torch.cat([mu, log_var], dim=1)
        z_hat = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        x_hat = self.decoder(z_hat)
        return z, x_hat, mu, log_var, z_hat


class FlatVelocityNet(nn.Module):
    def __init__(self, latent_dim, h_dim=64):
        super().__init__()
        self.fc_in = nn.Linear(latent_dim + 1, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, latent_dim)

    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions
        x = torch.cat([x, t], dim=1)
        x = act(self.fc_in(x))
        x = act(self.fc2(x))
        x = act(self.fc3(x))
        return self.fc_out(x)


@torch.no_grad()
def rk4_step(f, y, t, dt):
    # f: callable (y, t) -> dy/dt  (aka model forward haha)
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def integrate_path(model, initial_points, step_fn=rk4_step, n_steps=100,
                   save_trajectories=False, warp_fn=None):
    """this 'sampling' routine is primarily used for visualization."""
    device = next(model.parameters()).device
    current_points = initial_points.clone()
    ts = torch.linspace(0, 1, n_steps).to(device)
    if warp_fn: ts = warp_fn(ts)
    if save_trajectories: trajectories = [current_points]
    for i in range(len(ts) - 1):
        current_points = step_fn(model, current_points, ts[i], ts[i + 1] - ts[i])
        if save_trajectories: trajectories.append(current_points)
    if save_trajectories: return current_points, torch.stack(trajectories).cpu()
    return current_points


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


class SubmissionInterface(nn.Module):
    """All teams must implement this for automated evaluation.
    When you subclass/implement these methods, replace the NotImplementedError."""

    def __init__(self):
        super().__init__()

        # --- REQUIRED INFO:
        self.info = {
            'team': 'marco',
            'names': 'Marco',
        }
        self.latent_dim = 13
        self.specifications = {
            "base_channels": 16,
            "blocks_per_level": 2,
            "groups": 1
        }
        # ----

        # keep support for full auto-initialization:
        self.device = 'cpu'  # prob should change back to cpu for him
        self.load_vae()
        self.load_flow_model()

    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
            downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.vae = InspoResNetVAE(latent_dim=self.latent_dim, **self.specifications)
        vae_weights_file = 'downloaded_vae.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1CKSOQeVd7bvNafLojD5ftANVh8kQm5Pe/view?usp=drive_link"
        gdown.download(safetensors_link, vae_weights_file, quiet=False, fuzzy=True)
        self.vae.load_state_dict(load_file(vae_weights_file))
        # self.vae.load_state_dict(torch.load(f'../models/vae_{self.latent_dim}.pth', map_location=self.device))

    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.flow_model = FlatVelocityNet(latent_dim=self.latent_dim)
        flow_weights_file = 'downloaded_flow.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1C16z1HH71f84B5m82rk0HyAHgqhSjOtS/view?usp=drive_link"
        gdown.download(safetensors_link, flow_weights_file, quiet=False, fuzzy=True)
        self.flow_model.load_state_dict(load_file(flow_weights_file))
        # self.flow_model.load_state_dict(torch.load(f'../models/flow_{self.latent_dim}.pth', map_location=self.device))

    def generate_samples(self, n_samples: int, n_steps=15) -> torch.Tensor:
        z0 = torch.randn([n_samples, self.latent_dim]).to(self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps, step_fn=rk4_step)
        gen_xhat = F.sigmoid(self.decode(z1).view(-1, 28, 28))
        return gen_xhat

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
