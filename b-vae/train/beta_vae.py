# beta_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Small conv blocks used below
def ConvBlock(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )

def DeconvBlock(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class BetaVAE(nn.Module):
    """
    Beta-VAE with two latent heads: style and content.
    Encoder produces shared features -> two separate mu/logvar heads.
    Decoder accepts concatenated (z_style, z_content).
    """
    def __init__(self, img_size=256, style_dim=128, content_dim=128, base_channels=64):
        super().__init__()
        self.img_size = img_size
        self.style_dim = style_dim
        self.content_dim = content_dim
        self.base = base_channels

        # Encoder: down to 8x8 (for 256 input) or compute adaptively
        self.enc = nn.Sequential(
            ConvBlock(3, self.base),      # 128x128
            ConvBlock(self.base, self.base*2),  # 64x64
            ConvBlock(self.base*2, self.base*4),# 32x32
            ConvBlock(self.base*4, self.base*8),# 16x16
            ConvBlock(self.base*8, self.base*8),# 8x8
        )
        # flatten dim:
        self._feat_h = 8
        self._feat_w = 8
        self._feat_ch = self.base*8
        flat_dim = self._feat_ch * self._feat_h * self._feat_w

        # style heads
        self.fc_mu_s = nn.Linear(flat_dim, style_dim)
        self.fc_lv_s = nn.Linear(flat_dim, style_dim)
        # content heads
        self.fc_mu_c = nn.Linear(flat_dim, content_dim)
        self.fc_lv_c = nn.Linear(flat_dim, content_dim)

        # decoder: project fused latent back to feature map then upsample
        fused_dim = style_dim + content_dim
        self.fc_dec = nn.Linear(fused_dim, flat_dim)
        self.dec = nn.Sequential(
            DeconvBlock(self._feat_ch, self.base*4),  # 16x16
            DeconvBlock(self.base*4, self.base*2),    # 32x32
            DeconvBlock(self.base*2, self.base),      # 64x64
            DeconvBlock(self.base, self.base//2),     # 128x128
            nn.ConvTranspose2d(self.base//2, 3, kernel_size=4, stride=2, padding=1),  # 256x256
            nn.Sigmoid()  # outputs [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        b = h.shape[0]
        flat = h.view(b, -1)
        mu_s = self.fc_mu_s(flat); lv_s = self.fc_lv_s(flat)
        mu_c = self.fc_mu_c(flat); lv_c = self.fc_lv_c(flat)
        return mu_s, lv_s, mu_c, lv_c

    def reparam(self, mu, lv):
        std = (0.5 * lv).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_from_latent(self, z_s, z_c):
        z = torch.cat([z_s, z_c], dim=1)
        h = self.fc_dec(z)
        h = h.view(-1, self._feat_ch, self._feat_h, self._feat_w)
        out = self.dec(h)
        return out

    def forward(self, x):
        mu_s, lv_s, mu_c, lv_c = self._forward_stats(x)
        z_s = self.reparam(mu_s, lv_s)
        z_c = self.reparam(mu_c, lv_c)
        recon = self.decode_from_latent(z_s, z_c)
        return recon, mu_s, lv_s, mu_c, lv_c

    def _forward_stats(self, x):
        h = self.enc(x)
        flat = h.view(h.shape[0], -1)
        mu_s = self.fc_mu_s(flat); lv_s = self.fc_lv_s(flat)
        mu_c = self.fc_mu_c(flat); lv_c = self.fc_lv_c(flat)
        return mu_s, lv_s, mu_c, lv_c

    # helpers to just get encodings / traverse
    def encode_style(self, x):
        mu_s, lv_s, _, _ = self._forward_stats(x)
        return mu_s  # deterministic mean (for traversal use)
    def encode_content(self, x):
        _, _, mu_c, lv_c = self._forward_stats(x)
        return mu_c
    def decode_from_means(self, mu_s, mu_c):
        return self.decode_from_latent(mu_s, mu_c)
