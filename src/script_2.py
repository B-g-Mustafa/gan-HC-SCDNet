# HC-SCDNet: Core Architecture Implementation
# Following the architecture diagram provided

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

# ====================================
# STAGE 1: DISENTANGLED ENCODING (β-VAE Module)
# ====================================

class BetaVAEEncoder(nn.Module):
    """
    β-VAE Encoder for disentangled style and content representation learning
    Based on the architecture diagram: Stage 1 - Disentangled Encoding
    """
    def __init__(self, 
                 input_dim: int = 3,
                 style_latent_dim: int = 512,
                 content_latent_dim: int = 512,
                 hidden_dims: List[int] = [64, 128, 256, 512]):
        super(BetaVAEEncoder, self).__init__()
        
        self.style_latent_dim = style_latent_dim
        self.content_latent_dim = content_latent_dim
        
        # Shared convolutional layers (as shown in diagram)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(in_dim, h_dim, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = h_dim
            
        self.shared_encoder = nn.Sequential(*layers)
        
        # Calculate the flattened dimension after convolutions
        # Assuming input size 512x512, after 4 layers with stride 2: 512/2^4 = 32
        self.flatten_dim = hidden_dims[-1] * 32 * 32
        
        # Style-specific branches (Texture, Color, Brushstrokes)
        self.style_encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        
        # Content-specific branches (Shapes, Objects, Composition)  
        self.content_encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        
        # Latent distribution parameters
        self.style_mu = nn.Linear(512, style_latent_dim)
        self.style_logvar = nn.Linear(512, style_latent_dim)
        
        self.content_mu = nn.Linear(512, content_latent_dim)
        self.content_logvar = nn.Linear(512, content_latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through β-VAE encoder
        Args:
            x: Input image tensor [B, C, H, W]
        Returns:
            Dict containing style and content distributions
        """
        # Shared feature extraction
        shared_features = self.shared_encoder(x)  # [B, 512, 32, 32]
        shared_flat = shared_features.view(shared_features.size(0), -1)  # [B, 512*32*32]
        
        # Style encoding (Texture, Color, Brushstrokes)
        style_features = self.style_encoder(shared_flat)
        style_mu = self.style_mu(style_features)
        style_logvar = self.style_logvar(style_features)
        style_z = self.reparameterize(style_mu, style_logvar)
        
        # Content encoding (Shapes, Objects, Composition)
        content_features = self.content_encoder(shared_flat)
        content_mu = self.content_mu(content_features)
        content_logvar = self.content_logvar(content_features)
        content_z = self.reparameterize(content_mu, content_logvar)
        
        return {
            'style_z': style_z,
            'content_z': content_z,
            'style_mu': style_mu,
            'style_logvar': style_logvar,
            'content_mu': content_mu,
            'content_logvar': content_logvar,
            'shared_features': shared_features
        }

# ====================================
# STAGE 2: CONTROLLABLE FUSION (Lightweight Diffusion Module)
# ====================================

class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning diffusion on β-VAE latents"""
    def __init__(self, dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)
        
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)

class ResNetBlock(nn.Module):
    """ResNet block with group normalization"""
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        temb = self.temb_proj(F.silu(temb))
        h = h + temb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class LightweightDiffusionUNet(nn.Module):
    """
    Lightweight U-Net for diffusion process with β-VAE conditioning
    Following the Stage 2 architecture from the diagram
    """
    def __init__(self, 
                 in_channels: int = 4,
                 model_channels: int = 320,
                 out_channels: int = 4,
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple = (4, 2, 1),
                 channel_mult: Tuple = (1, 2, 4, 4),
                 style_dim: int = 512,
                 content_dim: int = 512):
        super().__init__()
        
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Style and content conditioning projections
        self.style_proj = nn.Linear(style_dim, model_channels)
        self.content_proj = nn.Linear(content_dim, model_channels)
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Downsampling blocks
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResNetBlock(ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                
                # Add cross-attention at specified resolutions
                if ds in attention_resolutions:
                    layers.append(CrossAttentionBlock(ch, style_dim + content_dim))
                    
                self.input_blocks.append(nn.Sequential(*layers))
                
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResNetBlock(ch, ch, time_embed_dim),
            CrossAttentionBlock(ch, style_dim + content_dim),
            ResNetBlock(ch, ch, time_embed_dim),
        )
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [ResNetBlock(ch + ch, mult * model_channels, time_embed_dim)]
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    layers.append(CrossAttentionBlock(ch, style_dim + content_dim))
                    
                if level and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                    
                self.output_blocks.append(nn.Sequential(*layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )
    
    def forward(self, 
                x: torch.Tensor, 
                timesteps: torch.Tensor, 
                style_z: torch.Tensor, 
                content_z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through lightweight diffusion U-Net
        Args:
            x: Noisy latent [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            style_z: Style latent from β-VAE [B, style_dim]
            content_z: Content latent from β-VAE [B, content_dim]
        """
        # Time embedding
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        
        # Conditioning context (concatenate style and content)
        conditioning = torch.cat([style_z, content_z], dim=-1).unsqueeze(1)  # [B, 1, style_dim + content_dim]
        
        # Forward through U-Net with skip connections
        hs = []
        h = x
        
        # Downsampling path
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, ResNetBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, CrossAttentionBlock):
                        # Reshape for cross-attention
                        b, c, h_dim, w_dim = h.shape
                        h_flat = h.view(b, c, h_dim * w_dim).transpose(1, 2)  # [B, H*W, C]
                        h_flat = layer(h_flat, conditioning)
                        h = h_flat.transpose(1, 2).view(b, c, h_dim, w_dim)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle block
        for layer in self.middle_block:
            if isinstance(layer, ResNetBlock):
                h = layer(h, emb)
            elif isinstance(layer, CrossAttentionBlock):
                b, c, h_dim, w_dim = h.shape
                h_flat = h.view(b, c, h_dim * w_dim).transpose(1, 2)
                h_flat = layer(h_flat, conditioning)
                h = h_flat.transpose(1, 2).view(b, c, h_dim, w_dim)
        
        # Upsampling path with skip connections
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResNetBlock):
                    h = layer(h, emb)
                elif isinstance(layer, CrossAttentionBlock):
                    b, c, h_dim, w_dim = h.shape
                    h_flat = h.view(b, c, h_dim * w_dim).transpose(1, 2)
                    h_flat = layer(h_flat, conditioning)
                    h = h_flat.transpose(1, 2).view(b, c, h_dim, w_dim)
                else:
                    h = layer(h)
        
        return self.out(h)

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

print("✅ Stage 1 & 2 implementation complete:")
print("   - BetaVAEEncoder for disentangled style/content encoding")
print("   - LightweightDiffusionUNet with cross-attention conditioning")
print("   - ResNet blocks and attention mechanisms")
print("Next: Stage 3 implementation...")