
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from beta_vae_encoder import BetaVAEEncoder
from lightweight_diffusion import LightweightDiffusionUNet
from quality_enhancement import QualityEnhancementModule
from diffusers import DDPMScheduler, AutoencoderKL
import math

class HCSCDNet(nn.Module):
    """
    HC-SCDNet: Hybrid Controllable Style-Content Disentanglement Network

    A novel neural style transfer architecture combining:
    1. β-VAE for disentangled style-content representation learning
    2. Lightweight diffusion model for controllable generation  
    3. Adversarial refinement for quality enhancement

    Following the complete architecture diagram provided.
    """

    def __init__(self,
                 style_latent_dim: int = 512,
                 content_latent_dim: int = 512,
                 diffusion_steps: int = 10,
                 beta: float = 4.0,
                 device: str = 'cuda'):
        super().__init__()

        self.device = device
        self.beta = beta
        self.diffusion_steps = diffusion_steps

        # Stage 1: Disentangled Encoding (β-VAE Module)
        self.beta_vae_encoder = BetaVAEEncoder(
            input_dim=3,
            style_latent_dim=style_latent_dim,
            content_latent_dim=content_latent_dim
        )

        # Pre-trained VAE decoder from Stable Diffusion (for latent space)
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            subfolder="vae"
        )

        # Stage 2: Controllable Fusion (Lightweight Diffusion Module) 
        self.diffusion_unet = LightweightDiffusionUNet(
            in_channels=4,  # Latent space channels
            style_dim=style_latent_dim,
            content_dim=content_latent_dim
        )

        # DDPM scheduler for diffusion process
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )

        # Stage 3: Quality Enhancement (Adversarial Refinement)
        self.quality_enhancer = QualityEnhancementModule(
            input_channels=3,
            perceptual_weight=1.0,
            style_weight=250.0,
            adversarial_weight=0.1
        )

        # Freeze VAE parameters (we only fine-tune our modules)
        for param in self.vae.parameters():
            param.requires_grad = False

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space"""
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * 0.18215  # Stable Diffusion scaling factor
        return latent

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image space"""
        with torch.no_grad():
            latent = latent / 0.18215
            image = self.vae.decode(latent).sample
        return image

    def compute_beta_vae_loss(self, 
                            style_mu: torch.Tensor, 
                            style_logvar: torch.Tensor,
                            content_mu: torch.Tensor, 
                            content_logvar: torch.Tensor,
                            reconstruction: torch.Tensor,
                            target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute β-VAE loss with disentanglement regularization
        Following the mathematical formulation from the diagram
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, target)

        # KL divergence for style latents
        style_kl = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
        style_kl = style_kl / (style_mu.size(0) * style_mu.size(1))  # Normalize by batch and latent dim

        # KL divergence for content latents  
        content_kl = -0.5 * torch.sum(1 + content_logvar - content_mu.pow(2) - content_logvar.exp())
        content_kl = content_kl / (content_mu.size(0) * content_mu.size(1))

        # Mutual information minimization between style and content
        # Simplified implementation using correlation penalty
        style_norm = F.normalize(style_mu, dim=-1)
        content_norm = F.normalize(content_mu, dim=-1)
        mi_loss = torch.abs(torch.sum(style_norm * content_norm, dim=-1)).mean()

        # Total β-VAE loss
        total_loss = recon_loss + self.beta * (style_kl + content_kl) + 0.1 * mi_loss

        return {
            'reconstruction': recon_loss,
            'style_kl': style_kl,
            'content_kl': content_kl,
            'mi_penalty': mi_loss,
            'total': total_loss
        }

    def diffusion_forward_process(self, 
                                latent: torch.Tensor, 
                                timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to latent according to diffusion schedule"""
        noise = torch.randn_like(latent)
        noisy_latent = self.scheduler.add_noise(latent, noise, torch.tensor([timestep]))
        return noisy_latent, noise

    def diffusion_reverse_process(self,
                                noisy_latent: torch.Tensor,
                                timestep: int,
                                style_z: torch.Tensor,
                                content_z: torch.Tensor,
                                style_control: float = 1.0,
                                content_control: float = 1.0) -> torch.Tensor:
        """
        Single reverse diffusion step with controllable conditioning
        """
        # Predict noise using our lightweight U-Net
        predicted_noise = self.diffusion_unet(
            noisy_latent,
            torch.tensor([timestep]).to(self.device),
            style_z,
            content_z,
            style_control,
            content_control
        )

        # Remove predicted noise (simplified DDIM-like step)
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        alpha_t_prev = self.scheduler.alphas_cumprod[max(0, timestep - self.diffusion_steps)]

        predicted_x0 = (noisy_latent - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        denoised = torch.sqrt(alpha_t_prev) * predicted_x0 + torch.sqrt(1 - alpha_t_prev) * predicted_noise

        return denoised

    def forward(self,
              content_image: torch.Tensor,
              style_image: torch.Tensor,
              style_control: float = 1.0,
              content_control: float = 1.0,
              num_inference_steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for controllable style transfer

        Args:
            content_image: Input content image [B, 3, H, W]
            style_image: Style reference image [B, 3, H, W]  
            style_control: Style intensity control [0, 1]
            content_control: Content preservation control [0, 1]
            num_inference_steps: Number of diffusion steps

        Returns:
            Dictionary containing generated image and intermediate results
        """
        batch_size = content_image.size(0)

        # ====================================
        # STAGE 1: DISENTANGLED ENCODING
        # ====================================

        # Extract disentangled representations
        content_encoding = self.beta_vae_encoder(content_image)
        style_encoding = self.beta_vae_encoder(style_image)

        # Get style and content latents
        content_z = content_encoding['content_z'] 
        style_z = style_encoding['style_z']

        # ====================================
        # STAGE 2: CONTROLLABLE FUSION
        # ====================================

        # Encode content to latent space for diffusion
        content_latent = self.encode_to_latent(content_image)

        # Initialize diffusion process
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Start from noisy content latent
        current_latent = torch.randn_like(content_latent)

        # Progressive denoising with controllable conditioning
        for i, timestep in enumerate(timesteps):
            current_latent = self.diffusion_reverse_process(
                current_latent,
                timestep.item(),
                style_z,
                content_z,
                style_control,
                content_control
            )

        # Decode to image space
        generated_image = self.decode_from_latent(current_latent)

        # ====================================
        # STAGE 3: QUALITY ENHANCEMENT
        # ====================================

        # Apply quality enhancement (during training)
        if self.training:
            quality_losses = self.quality_enhancer.compute_generator_loss(
                generated_image,
                content_image,
                style_image
            )

            # Compute β-VAE losses
            vae_losses = self.compute_beta_vae_loss(
                content_encoding['content_mu'],
                content_encoding['content_logvar'],
                style_encoding['style_mu'],
                style_encoding['style_logvar'],
                generated_image,
                content_image  # Reconstruction target
            )

            return {
                'generated_image': generated_image,
                'style_z': style_z,
                'content_z': content_z,
                'vae_losses': vae_losses,
                'quality_losses': quality_losses,
                'content_latent': content_latent,
                'style_encoding': style_encoding,
                'content_encoding': content_encoding
            }
        else:
            return {
                'generated_image': generated_image,
                'style_z': style_z, 
                'content_z': content_z
            }

    def get_controllable_generation(self,
                                  content_image: torch.Tensor,
                                  style_image: torch.Tensor,
                                  style_controls: list = [0.0, 0.5, 1.0],
                                  content_controls: list = [0.0, 0.5, 1.0]) -> torch.Tensor:
        """
        Generate a grid of results with different control settings
        Useful for demonstrating controllability
        """
        results = []

        for style_ctrl in style_controls:
            row = []
            for content_ctrl in content_controls:
                output = self.forward(
                    content_image,
                    style_image, 
                    style_control=style_ctrl,
                    content_control=content_ctrl
                )
                row.append(output['generated_image'])
            results.append(torch.cat(row, dim=-1))  # Concatenate horizontally

        return torch.cat(results, dim=-2)  # Concatenate vertically

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters in each module"""
        counts = {
            'beta_vae_encoder': sum(p.numel() for p in self.beta_vae_encoder.parameters() if p.requires_grad),
            'diffusion_unet': sum(p.numel() for p in self.diffusion_unet.parameters() if p.requires_grad),
            'quality_enhancer': sum(p.numel() for p in self.quality_enhancer.parameters() if p.requires_grad),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'total_all': sum(p.numel() for p in self.parameters())
        }
        return counts
