## Sample submission file for the VAE + Flow leaderboard challenge
## Author: Scott H. Hawley, Oct 6 2025
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import gdown
import os


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_skip=True, use_bn=True, act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.use_skip, self.act = use_skip, act

    def forward(self, x):
        if self.use_skip: x0 = x
        out = self.act()(self.bn1(self.conv1(x)))
        out = F.dropout(out, 0.4, training=self.training)
        out = self.bn2(self.conv2(out))
        if self.use_skip: out = out + x0
        return self.act()(out)

class ResNetVAEEncoderSpatial(nn.Module):
    "this shrinks down to a wee image for its latents, e.g. for MNIST: 1x28x28 -> 1x7x7 for two downsampling operations"

    def __init__(self, in_channels, latent_channels=1, base_channels=32, blocks_per_level=4, use_skips=True,
                 use_bn=True, act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act) for _ in range(blocks_per_level)]) for ch in
             channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.channel_proj = nn.Conv2d(in_channels=channels[-1], out_channels=2 * latent_channels,
                                      kernel_size=1)  # 1x1 conv
        self.act = act

    def forward(self, x):
        x = self.act()(self.bn1(self.conv1(x)))
        for i in range(len(self.levels)):
            if i > 0:  # shrink down
                x = F.avg_pool2d(x, 2)
                x = self.transitions[i - 1](x)
            for block in self.levels[i]:
                x = block(x)
        x = self.channel_proj(x)
        mean, logvar = x.chunk(2, dim=1)  # mean and log variance
        return mean, logvar


class ResNetVAEDecoderSpatial(nn.Module):
    """this is just the mirror image of ResNetVAEEnoderSpatial"""

    def __init__(self, out_channels, latent_channels=1, base_channels=32, blocks_per_level=4, use_skips=True,
                 use_bn=True, act=nn.GELU):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4][::-1]  # reversed from encoder
        self.channels = channels
        self.channel_proj = nn.Conv2d(in_channels=latent_channels, out_channels=channels[0], kernel_size=1)  # 1x1 conv
        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act) for _ in range(blocks_per_level)]) for ch in
             channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.act = act

    def forward(self, z):
        x = self.channel_proj(z)
        for i in range(len(self.levels)):
            for block in self.levels[i]:
                x = block(x)
            if i < len(self.levels) - 1:  # not last level
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self.transitions[i](x)
        return self.final_conv(x)


class ResNetVAE(nn.Module):
    """Main VAE class"""

    def __init__(self, data_channels=1, latent_dim=3, act=nn.GELU):
        super().__init__()
        self.encoder = ResNetVAEEncoderSpatial(data_channels, act=act)
        self.decoder = ResNetVAEDecoderSpatial(data_channels, act=act)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = torch.cat([mu, log_var], dim=1)  # this is unnecessary/redundant but our other Lesson code expects z
        z_hat = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        x_hat = self.decoder(z_hat)
        return z, x_hat, mu, log_var, z_hat


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = act()
        # nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')  # test later
        nn.init.zeros_(self.conv.bias)  # start zero?


    def forward(self, x):
        return self.act(self.conv(x))

class ConvFlowNet(nn.Module):
    def __init__(self, latent_ch, hidden=32, depth=3, grow=True):
        super().__init__()
        self.in_conv  = nn.Conv2d(latent_ch + 1, hidden, 3, padding=1)
        channels = [hidden * (2**i) for i in range(depth)] if grow else [hidden] * depth

        self.blocks = nn.Sequential(
            *[ConvBlock(in_ch, out_ch)
              for in_ch, out_ch in zip(channels[:-1], channels[1:])]
        )

        self.out_conv = nn.Conv2d(channels[-1], latent_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z, t):
        # z: [B,C,H,W], t: [B] or [B,1]
        B, C, H, W = z.shape
        t_img = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        h = F.silu(self.in_conv(torch.cat([z, t_img], dim=1)))
        h = self.blocks(h)
        return self.out_conv(h)


@torch.no_grad()
def fwd_euler_step(model, current_points, current_t, dt):
    velocity = model(current_points, current_t)
    return current_points + velocity * dt

@torch.no_grad()
def rk4_step(f, y, t, dt):
    # f: callable (y, t) -> dy/dt  (aka model forward haha)
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# @torch.no_grad()
# def integrate_path(model, initial_points, step_fn=rk4_step, n_steps=100,
#                    save_trajectories=False, warp_fn=None):
#     """this 'sampling' routine is primarily used for visualization."""
#     device = next(model.parameters()).device
#     current_points = initial_points.clone()
#     ts = torch.linspace(0, 1, n_steps).to(device)
#     if warp_fn: ts = warp_fn(ts)
#     if save_trajectories: trajectories = [current_points]
#     for i in range(len(ts) - 1):
#         current_points = step_fn(model, current_points, ts[i], ts[i + 1] - ts[i])
#         if save_trajectories: trajectories.append(current_points)
#     if save_trajectories: return current_points, torch.stack(trajectories).cpu()
#     return current_points

@torch.no_grad()
def integrate_path(model, initial_points, step_fn=rk4_step, n_steps=100, warp_fn=None):
    """Fast spatial integrator for [B, C, H, W] shapes."""
    p = next(model.parameters())
    device, model_dtype = p.device, p.dtype

    y = initial_points.to(device=device, dtype=model_dtype).clone()
    model.eval()

    ts = torch.linspace(0, 1, n_steps, device=device, dtype=model_dtype)
    if warp_fn: ts = warp_fn(ts)

    t_batch = torch.empty((y.shape[0], 1), device=device, dtype=model_dtype)

    for i in range(len(ts) - 1):
        t, dt = ts[i], ts[i + 1] - ts[i]
        t_batch.fill_(t.item())
        y = step_fn(model, y, t_batch, dt)

    return y


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
            'team': 'marco', # _sp_7x7_flow
            'names': 'Marco',
        }
        self.latent_dim = 7  # TODO: we could just (re)measure this on the fly
        self.latent_shape = (1,7,7)
        # ----

        # keep support for full auto-initialization:
        self.device = 'mps'
        self.load_vae()
        self.load_flow_model()

    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
            downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.vae = ResNetVAE()
        vae_weights_file = 'downloaded_vae.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1dwifwb3dL4U6d2SLhEKLMEmezdJZfkhB/view?usp=drive_link"
        gdown.download(safetensors_link, vae_weights_file, quiet=False, fuzzy=True)
        self.vae.load_state_dict(load_file(vae_weights_file))
        # self.vae.load_state_dict(load_file(r'../models/safetensors/vae/sp_vae.safetensors'))

    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.flow_model = ConvFlowNet(1)
        flow_weights_file = 'downloaded_flow.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1BnOg-aQDJUF2FC9wjUP-F0VinAZgvynO/view?usp=drive_link"
        gdown.download(safetensors_link, flow_weights_file, quiet=False, fuzzy=True)
        self.flow_model.load_state_dict(load_file(flow_weights_file))
        # self.flow_model.load_state_dict(load_file(r'../models/safetensors/flow/sp_flow_7x7.safetensors'))

    def generate_samples(self, n_samples:int, n_steps=10) -> torch.Tensor:
        self.flow_model.eval(); self.vae.eval()
        z0 = torch.randn(n_samples, *self.latent_shape, device=self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps, step_fn=rk4_step)
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
