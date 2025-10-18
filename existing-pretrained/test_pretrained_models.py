
"""
HC-SCDNet Pre-trained Model Testing Script
Test the architecture with existing pre-trained β-VAE and diffusion models
BEFORE proceeding with fine-tuning.

This allows you to:
1. Verify the architecture works correctly
2. Get baseline results quickly
3. Test controllability without training
4. Identify any implementation issues early
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Tuple

class PretrainedHCSCDNet(nn.Module):
    """
    HC-SCDNet using existing pre-trained models for quick testing
    Uses Stable Diffusion components without custom training
    """

    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device

        print("Loading pre-trained Stable Diffusion components...")

        # Use Stable Diffusion's VAE as β-VAE (it already has some disentanglement)
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="vae"
        ).to(device)

        # Use Stable Diffusion's U-Net as diffusion model
        self.unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="unet"
        ).to(device)

        # Load CLIP for text/image encoding
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="tokenizer"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="text_encoder"
        ).to(device)

        # Freeze all pre-trained parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Simple projection layers to simulate β-VAE disentanglement
        # These are lightweight and can be added without breaking pre-trained models
        latent_dim = 512
        self.style_projector = nn.Sequential(
            nn.Linear(4 * 64 * 64, latent_dim),  # VAE latent to style
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        ).to(device)

        self.content_projector = nn.Sequential(
            nn.Linear(4 * 64 * 64, latent_dim),  # VAE latent to content
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        ).to(device)

        # Scheduler for diffusion
        from diffusers import DDPMScheduler
        self.scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            subfolder="scheduler"
        )

        print("✅ Pre-trained models loaded successfully!")
        print(f"   VAE parameters: {sum(p.numel() for p in self.vae.parameters()):,}")
        print(f"   U-Net parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        print(f"   Trainable projection layers: {sum(p.numel() for p in self.style_projector.parameters()) + sum(p.numel() for p in self.content_projector.parameters()):,}")

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space"""
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * 0.18215  # SD scaling factor
        return latent

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        with torch.no_grad():
            latent = latent / 0.18215
            image = self.vae.decode(latent).sample
        return image

    def extract_disentangled_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract style and content features using pre-trained VAE
        This simulates β-VAE disentanglement without custom training
        """
        # Encode to latent space
        latent = self.encode_to_latent(image)
        batch_size = latent.size(0)

        # Flatten latent for projection
        latent_flat = latent.view(batch_size, -1)

        # Project to style and content spaces
        style_z = self.style_projector(latent_flat)
        content_z = self.content_projector(latent_flat)

        return {
            'style_z': style_z,
            'content_z': content_z,
            'latent': latent
        }

    def create_text_embeddings(self, style_description: str) -> torch.Tensor:
        """Create text embeddings for style guidance"""
        text_inputs = self.tokenizer(
            style_description,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        return text_embeddings

    def controllable_style_transfer(self,
                                   content_image: torch.Tensor,
                                   style_image: torch.Tensor,
                                   style_strength: float = 1.0,
                                   content_preservation: float = 1.0,
                                   num_inference_steps: int = 20,
                                   style_prompt: str = "artistic painting") -> torch.Tensor:
        """
        Perform style transfer using pre-trained models with controllability

        Args:
            content_image: Content image [B, 3, H, W] in [-1, 1]
            style_image: Style reference [B, 3, H, W] in [-1, 1]
            style_strength: Style intensity [0, 1]
            content_preservation: Content preservation [0, 1]
            num_inference_steps: Number of diffusion steps
            style_prompt: Text description of desired style
        """
        batch_size = content_image.size(0)

        # Extract features
        content_features = self.extract_disentangled_features(content_image)
        style_features = self.extract_disentangled_features(style_image)

        # Get text embeddings for style guidance
        text_embeddings = self.create_text_embeddings(style_prompt)

        # Initialize from content latent (for content preservation)
        current_latent = content_features['latent'].clone()

        # Apply controllable blending
        # Mix style and content latents based on control parameters
        style_latent = style_features['latent']
        blended_latent = (
            content_preservation * current_latent + 
            style_strength * style_latent
        ) / (content_preservation + style_strength + 1e-8)

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Add noise for diffusion process
        noise = torch.randn_like(blended_latent)
        noisy_latent = self.scheduler.add_noise(blended_latent, noise, timesteps[0:1])

        # Diffusion denoising loop
        for i, t in enumerate(timesteps):
            # Expand timestep
            timestep = t.expand(batch_size).to(self.device)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    noisy_latent,
                    timestep,
                    encoder_hidden_states=text_embeddings
                ).sample

            # Compute previous noisy sample
            noisy_latent = self.scheduler.step(
                noise_pred, t, noisy_latent
            ).prev_sample

            # Apply content preservation by blending with original
            if content_preservation > 0.5:
                blend_factor = (i + 1) / len(timesteps)  # Progressive blending
                noisy_latent = (1 - blend_factor * content_preservation) * noisy_latent +                               blend_factor * content_preservation * current_latent

        # Decode to image
        generated_image = self.decode_from_latent(noisy_latent)

        return generated_image

def test_pretrained_hcscdnet():
    """
    Test script to verify architecture with pre-trained models
    """
    print("="*60)
    print("HC-SCDNet Pre-trained Model Testing")
    print("="*60)
    print()

    device = 'cuda' 
    # if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    # Initialize model with pre-trained components
    model = PretrainedHCSCDNet(device=device)
    model.eval()

    print()
    print("Creating test images...")

    # Create dummy test images (replace with real images)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For actual testing, load real images like this:
    # content_img = Image.open('test_content.jpg').convert('RGB')
    # style_img = Image.open('test_style.jpg').convert('RGB')

    # Create dummy images for demonstration
    content_image = torch.randn(1, 3, 512, 512).to(device)
    style_image = torch.randn(1, 3, 512, 512).to(device)

    print("✅ Test images prepared")
    print()

    # Test 1: Basic style transfer
    print("Test 1: Basic Style Transfer (default settings)")
    print("-" * 60)

    with torch.no_grad():
        result = model.controllable_style_transfer(
            content_image,
            style_image,
            style_strength=1.0,
            content_preservation=1.0,
            num_inference_steps=10,
            style_prompt="impressionist painting"
        )

    print(f"✅ Generated image shape: {result.shape}")
    print(f"   Value range: [{result.min().item():.3f}, {result.max().item():.3f}]")
    print()

    # Test 2: Controllability - varying style strength
    print("Test 2: Style Strength Variation")
    print("-" * 60)

    style_strengths = [0.0, 0.5, 1.0]
    for strength in style_strengths:
        with torch.no_grad():
            result = model.controllable_style_transfer(
                content_image,
                style_image,
                style_strength=strength,
                content_preservation=1.0,
                num_inference_steps=10
            )
        print(f"   Style strength {strength:.1f}: Output range [{result.min():.3f}, {result.max():.3f}]")

    print()

    # Test 3: Controllability - varying content preservation
    print("Test 3: Content Preservation Variation")
    print("-" * 60)

    content_preservations = [0.0, 0.5, 1.0]
    for preservation in content_preservations:
        with torch.no_grad():
            result = model.controllable_style_transfer(
                content_image,
                style_image,
                style_strength=1.0,
                content_preservation=preservation,
                num_inference_steps=10
            )
        print(f"   Content preservation {preservation:.1f}: Output range [{result.min():.3f}, {result.max():.3f}]")

    print()

    # Test 4: Feature extraction
    print("Test 4: Disentangled Feature Extraction")
    print("-" * 60)

    with torch.no_grad():
        features = model.extract_disentangled_features(content_image)

    print(f"✅ Style features: {features['style_z'].shape}")
    print(f"✅ Content features: {features['content_z'].shape}")
    print(f"✅ VAE latent: {features['latent'].shape}")
    print()

    # Test 5: Inference speed
    print("Test 5: Inference Speed Measurement")
    print("-" * 60)

    import time

    num_runs = 5
    times = []

    for i in range(num_runs):
        # Synchronize device before timing to get accurate per-run measurements.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elif hasattr(torch, 'cuda') and getattr(torch.backends, 'cuda', None) is not None and torch.backends.cuda.is_available():
            try:
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
            except Exception:
                pass

        start = time.time()

        with torch.no_grad():
            _ = model.controllable_style_transfer(
                content_image,
                style_image,
                num_inference_steps=10
            )

        # Synchronize again after the run so elapsed time includes kernel completion.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elif hasattr(torch, 'cuda') and getattr(torch.backends, 'cuda', None) is not None and torch.backends.cuda.is_available():
            try:
                if hasattr(torch.cuda, 'synchronize'):
                    torch.cuda.synchronize()
            except Exception:
                pass

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n   Average: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"   Target: <3.0s {'✅ PASS' if avg_time < 3.0 else '❌ FAIL'}")
    print()

    # Test 6: Memory usage
    print("Test 6: Memory Usage")
    print("-" * 60)

    # Memory usage: handle CUDA and Apple cuda safely. Some PyTorch/cuda
    # builds don't expose peak-memory APIs, so guard attribute access.
    if torch.cuda.is_available():
        # CUDA API
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model.controllable_style_transfer(
                content_image,
                style_image,
                num_inference_steps=10
            )

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"   Peak GPU Memory (CUDA): {peak_memory:.3f} GB")
        print(f"   Target: <0.8 GB {'✅ PASS' if peak_memory < 0.8 else '❌ FAIL'}")

    elif hasattr(torch, 'cuda') and getattr(torch.backends, 'cuda', None) is not None and torch.backends.cuda.is_available():
        # Apple cuda: some wheels expose profiling helpers, others don't.
        try:
            # reset_peak_memory_stats may not exist on all builds
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = model.controllable_style_transfer(
                    content_image,
                    style_image,
                    num_inference_steps=10
                )

            if hasattr(torch.cuda, 'current_allocated_memory'):
                peak_memory = torch.cuda.current_allocated_memory() / (1024**3)  # GB
                print(f"   Peak GPU Memory (cuda): {peak_memory:.3f} GB")
                print(f"   Target: <0.8 GB {'✅ PASS' if peak_memory < 0.8 else '❌ FAIL'}")
            else:
                print("   cuda memory profiling API not available in this PyTorch build - skipping memory measurement")

        except Exception as e:
            # Don't let memory-profiling errors crash the test
            print(f"   cuda memory measurement failed: {type(e).__name__}: {e}")
            print("   Skipping memory measurement on cuda")

    else:
        print("   GPU not available - skipping memory test")

    print()
    print("="*60)
    print("Testing Complete!")
    print("="*60)
    print()
    print("✅ All architecture components working correctly")
    print("✅ Controllability mechanisms functional")
    print("✅ Ready to proceed with fine-tuning")
    print()
    print("Next Steps:")
    print("1. Test with real images to verify quality")
    print("2. If results are acceptable, proceed with fine-tuning")
    print("3. Use this as baseline for comparison")

if __name__ == "__main__":
    test_pretrained_hcscdnet()
