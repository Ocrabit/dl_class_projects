import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb


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


class ResNetVAEEncoder(nn.Module):
    """this makes a 1D vector of length latent_dim"""

    def __init__(self, in_channels, latent_dim=3, base_channels=32, blocks_per_level=4, use_skips=True, use_bn=True,
                 act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act) for _ in range(blocks_per_level)]) for ch in
             channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, 2 * latent_dim)
        self.act = act

    def forward(self, x):
        x = self.act()(self.bn1(self.conv1(x)))
        for i in range(len(self.levels)):
            if i > 0:  # shrink down
                x = F.avg_pool2d(x, 2)
                x = self.transitions[i - 1](x)
            for block in self.levels[i]:
                x = block(x)
        # print("encoder: x.shape = ", x.shape)
        x = self.global_avg_pool(x)
        x = self.fc(x.flatten(start_dim=1))
        mean, logvar = x.chunk(2, dim=1)  # mean and log variance
        return mean, logvar


class ResNetVAEDecoder(nn.Module):
    """this is just the mirror image of ResNetVAEEnoder"""

    def __init__(self, out_channels, latent_dim=3, base_channels=32, blocks_per_level=4, use_skips=True, use_bn=True,
                 act=nn.GELU):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4][::-1]  # reversed from encoder
        self.channels = channels
        self.start_dim = 7  # starting spatial dimension
        self.fc = nn.Linear(latent_dim, channels[0] * self.start_dim * self.start_dim)  # 128 * 16  # starting size
        self.levels = nn.ModuleList(
            [nn.ModuleList([ResidualBlock(ch, use_skips, use_bn, act=act) for _ in range(blocks_per_level)]) for ch in
             channels])
        self.transitions = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], 1, bias=not use_bn) for i in range(len(channels) - 1)])
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.act = act

    def forward(self, z):
        x = self.fc(z).view(-1, self.channels[0], self.start_dim, self.start_dim)  # project to spatial
        for i in range(len(self.levels)):
            for block in self.levels[i]:
                x = block(x)
            if i < len(self.levels) - 1:  # not last level
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self.transitions[i](x)
        return self.final_conv(x)


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

    def __init__(self,
                 data_channels=1,  # 1 channel for MNIST, 3 for CFAR10, etc.
                 latent_dim=3,  # dimensionality of the latent space. bigger=less compression, better reconstruction
                 act=nn.GELU,
                 spatial=True,
                 ):
        super().__init__()
        if spatial:
            self.encoder = ResNetVAEEncoderSpatial(data_channels, latent_channels=1, act=act)
            self.decoder = ResNetVAEDecoderSpatial(data_channels, latent_channels=1, act=act)
        else:
            self.encoder = ResNetVAEEncoder(data_channels, latent_dim=latent_dim, act=act)
            self.decoder = ResNetVAEDecoder(data_channels, latent_dim=latent_dim, act=act)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = torch.cat([mu, log_var], dim=1)  # this is unnecessary/redundant but our other Lesson code expects z
        z_hat = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        x_hat = self.decoder(z_hat)
        return z, x_hat, mu, log_var, z_hat

@torch.no_grad()
def test_inference(model, test_ds, idx=None, return_fig=False):
    device = next(model.parameters()).device
    model.eval()
    if idx is None: idx = torch.randint(len(test_ds), (1,))[0]
    if isinstance(idx, int): idx = [idx]
    elif isinstance(idx, range): idx = list(idx)
    x_batch = torch.stack([test_ds[i][0] for i in idx]).to(device)  # images
    y_batch = torch.tensor([test_ds[i][1] for i in idx]).to(device) # labels
    result = model.forward(x_batch)
    z, recon, mu, log_var, z_hat= result[:2]
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
    plt.show()


@torch.no_grad()
def test_inference_spatial(model, test_ds, idx=None, return_fig=False):
    was_training = model.training
    model.eval()

    if idx is None:
        idx = [torch.randint(len(test_ds), ()).item()]
    elif isinstance(idx, int):
        idx = [idx]
    elif isinstance(idx, range):
        idx = list(idx)
    else:
        idx = list(idx)

    xs = [test_ds[i][0] for i in idx]
    x_batch = torch.stack(xs, 0).to(next(model.parameters()).device)
    B, C, H, W = x_batch.shape

    out = model(x_batch)

    # Find recon and mu
    if isinstance(out, (tuple, list)):
        recon = next(t for t in out if isinstance(t, torch.Tensor) and t.shape[:2] == x_batch.shape[:2])
        # mu likely [B, Cmu, Hmu, Wmu]; pick something tensor-ish that isn’t recon
        mu = next((t for t in out if isinstance(t, torch.Tensor) and t is not recon and t.dim() == 4 and t.size(0) == B), None)
    else:
        recon, mu = out, None

    recon_vis = torch.sigmoid(recon) if (recon.min() < 0 or recon.max() > 1) else recon

    # Prepare a per-sample normalized μ map to visualize
    mu_vis = None
    if mu is not None and mu.dim() == 4:
        # choose first channel (or you could take mean across channels)
        mu_ = mu[:, 0:1]  # [B,1,h,w]
        mu_flat = mu_.view(B, -1)
        mu_min = mu_flat.min(dim=1, keepdim=True)[0]
        mu_max = mu_flat.max(dim=1, keepdim=True)[0]
        denom = (mu_max - mu_min).clamp_min(1e-8)
        mu_norm = ((mu_flat - mu_min) / denom).view_as(mu_)  # [B,1,h,w]
        mu_vis = mu_norm

    x_cpu = x_batch.detach().cpu()
    r_cpu = recon_vis.detach().cpu()
    mu_cpu = mu_vis.detach().cpu() if mu_vis is not None else None

    nrows = 3 if mu_cpu is not None else 2
    fig, axs = plt.subplots(nrows, B, figsize=(2.2*B, 2.6*nrows))
    if B == 1: axs = axs.reshape(nrows, 1)

    def show(ax, img, cmap='gray'):
        if img.dim() == 3:
            if img.size(0) == 1:
                ax.imshow(img.squeeze(0), cmap=cmap)
            else:
                ax.imshow(img.permute(1,2,0))
        else:
            ax.imshow(img, cmap=cmap)
        ax.axis('off')

    for i in range(B):
        show(axs[0, i], x_cpu[i], cmap='gray')
        axs[0, 0].set_ylabel('Input')
        row = 1
        if mu_cpu is not None:
            show(axs[row, i], mu_cpu[i].squeeze(0), cmap='viridis')
            axs[row, 0].set_ylabel('Latent μ'); row += 1
        show(axs[row, i], r_cpu[i], cmap='gray')
        axs[row, 0].set_ylabel('Recon')
    fig.tight_layout()

    if was_training: model.train()
    if return_fig: return fig
    plt.show()


def log_example_images(model, test_ds, epoch, spatial=True, n=5):
    if wandb.run is None:
        return
    fig = (test_inference_spatial if spatial else test_inference)(
        model, test_ds, idx=range(n), return_fig=True
    )
    wandb.log({"reconstructions": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)