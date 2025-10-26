# =============================================================================
# HC-SCDNet: Style Transfer Fine-tuning for Stable Diffusion 3.5 Large
# Fine-tune MM-DiT with LoRA for controllable style-content transfer
# =============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import lpips
from PIL import Image
import os
import sys
import traceback
from pathlib import Path
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Model settings
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    output_dir = "./sd35_style_transfer_lora"
    
    # LoRA settings
    lora_rank = 32
    lora_alpha = 32
    lora_dropout = 0.0
    lora_target_modules = [
        "attn.to_q",
        "attn.to_k", 
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2"
    ]
    
    # Training settings
    num_epochs = 10
    batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 4e-4
    lr_scheduler = "cosine"
    lr_warmup_steps = 100
    max_grad_norm = 1.0
    
    # Loss weights
    lambda_content = 1.0
    lambda_style = 10.0
    lambda_perceptual = 0.5
    
    # Image settings
    resolution = 512
    img2img_strength = 0.6  # How much to denoise (higher = more style)
    num_inference_steps = 20
    
    # Data paths
    content_dir = "../b-vae/data/coco_split/train"
    style_dir = "../b-vae/data/wikiart_split/train"
    
    # Logging
    log_with = "wandb"
    project_name = "hc-scdnet-sd35"
    save_steps = 500
    
    # Hardware
    mixed_precision = "bf16"
    gradient_checkpointing = True
    
    # Seed
    seed = 42

cfg = Config()

# =============================================================================
# Dataset
# =============================================================================

class StyleTransferDataset(Dataset):
    """Dataset for unpaired style transfer training."""
    
    def __init__(self, content_dir, style_dir, resolution=512):
        self.content_paths = list(Path(content_dir).glob("*.jpg")) + \
                            list(Path(content_dir).glob("*.png"))
        self.style_paths = list(Path(style_dir).glob("*.jpg")) + \
                          list(Path(style_dir).glob("*.png"))
        
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return max(len(self.content_paths), len(self.style_paths))
    
    def __getitem__(self, idx):
        # Random pairing for unpaired training
        content_idx = idx % len(self.content_paths)
        style_idx = np.random.randint(0, len(self.style_paths))
        
        content_img = Image.open(self.content_paths[content_idx]).convert("RGB")
        style_img = Image.open(self.style_paths[style_idx]).convert("RGB")
        
        content_tensor = self.transform(content_img)
        style_tensor = self.transform(style_img)
        
        return {
            "content": content_tensor,
            "style": style_tensor,
            "content_path": str(self.content_paths[content_idx]),
            "style_path": str(self.style_paths[style_idx])
        }

# =============================================================================
# Loss Functions
# =============================================================================

class VGGPerceptualLoss(torch.nn.Module):
    """VGG-based perceptual loss for content preservation."""
    
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.content_layers = ['21']  # relu4_2
        self.style_layers = ['0', '5', '10', '19', '28']  # relu1_1 to relu5_1
        
        self.vgg_layers = torch.nn.ModuleDict()
        for i, layer in enumerate(vgg):
            layer_name = str(i)
            self.vgg_layers[layer_name] = layer
            if layer_name in self.style_layers:
                self.vgg_layers[layer_name].eval()
        
        for param in self.parameters():
            param.requires_grad = False
    
    def extract_features(self, x):
        """Extract VGG features at multiple layers."""
        features = {}
        for name, layer in self.vgg_layers.items():
            x = layer(x)
            if name in self.content_layers or name in self.style_layers:
                features[name] = x
        return features
    
    def gram_matrix(self, x):
        """Compute Gram matrix for style representation."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, generated, content, style):
        """Compute combined content and style loss."""
        gen_features = self.extract_features(generated)
        content_features = self.extract_features(content)
        style_features = self.extract_features(style)
        
        # Content loss
        content_loss = 0
        for layer in self.content_layers:
            content_loss += F.mse_loss(gen_features[layer], 
                                       content_features[layer])
        
        # Style loss (Gram matrix)
        style_loss = 0
        for layer in self.style_layers:
            gen_gram = self.gram_matrix(gen_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            style_loss += F.mse_loss(gen_gram, style_gram)
        
        return content_loss, style_loss

# =============================================================================
# Training Functions
# =============================================================================

def setup_lora_model(transformer):
    """Configure and apply LoRA to MM-DiT transformer."""
    
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
    )
    
    transformer = get_peft_model(transformer, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in transformer.parameters() 
                          if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())
    
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}")
    
    return transformer

def encode_with_vae(vae, images):
    """Encode images to latent space using VAE."""
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents

def decode_with_vae(vae, latents):
    """Decode latents to pixel space using VAE."""
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        images = vae.decode(latents).sample
    return images

def add_noise_for_img2img(scheduler, latents, strength, noise):
    """Add noise to latents for img2img (preserves some content structure)."""
    # Calculate how many timesteps to denoise based on strength
    init_timestep = int(scheduler.config.num_train_timesteps * strength)
    init_timestep = min(init_timestep, scheduler.config.num_train_timesteps)
    
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device=latents.device)
    
    # Add noise according to noise magnitude at timestep
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    
    return noisy_latents, timesteps

def train_step(batch, pipeline, optimizer, perceptual_loss, lpips_fn, 
               accelerator, global_step):
    """Single training step."""
    
    content_imgs = batch["content"]
    style_imgs = batch["style"]
    
    # Encode to latents
    content_latents = encode_with_vae(pipeline.vae, content_imgs)
    
    # Add noise for img2img
    noise = torch.randn_like(content_latents)
    noisy_latents, timesteps = add_noise_for_img2img(
        pipeline.scheduler, 
        content_latents, 
        cfg.img2img_strength, 
        noise
    )
    
    # Forward pass through transformer (this is where LoRA is applied)
    # In practice, you'd inject style conditioning here via attention
    # For now, we use text conditioning as a proxy
    prompt = "transfer the artistic style"
    
    with accelerator.autocast():
        # Get text embeddings
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(accelerator.device)
        
        text_embeddings = pipeline.text_encoder(text_inputs.input_ids)[0]
        
        # Predict noise with transformer
        model_pred = pipeline.transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Decode to images
        generated_latents = noisy_latents - model_pred
        generated_imgs = decode_with_vae(pipeline.vae, generated_latents)
    
    # Compute losses
    content_loss, style_loss = perceptual_loss(
        generated_imgs, 
        content_imgs, 
        style_imgs
    )
    
    # LPIPS perceptual loss
    lpips_loss = lpips_fn(generated_imgs, content_imgs).mean()
    
    # Combined loss
    total_loss = (cfg.lambda_content * content_loss + 
                  cfg.lambda_style * style_loss + 
                  cfg.lambda_perceptual * lpips_loss)
    
    # Backward pass
    accelerator.backward(total_loss)
    
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(pipeline.transformer.parameters(), 
                                   cfg.max_grad_norm)
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Logging
    logs = {
        "loss/total": total_loss.item(),
        "loss/content": content_loss.item(),
        "loss/style": style_loss.item(),
        "loss/lpips": lpips_loss.item(),
        "lr": optimizer.param_groups[0]["lr"]
    }
    
    return logs, generated_imgs

# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.log_with,
        project_dir=cfg.output_dir,
    )
    
    # Initialize wandb
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.project_name)
        wandb.init(project=cfg.project_name)
    
    # Set seed
    set_seed(cfg.seed)
    
    # Load models (wrapped in try/except to surface errors clearly)
    print("Loading Stable Diffusion 3.5 Large...")
    sys.stdout.flush()
    try:
        # Load VAE (frozen)
        print("-> Loading VAE subfolder...")
        sys.stdout.flush()
        vae = AutoencoderKL.from_pretrained(
            cfg.model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )
        vae.requires_grad_(False)
        print("<-- VAE loaded")
        sys.stdout.flush()

        # Load text encoders (frozen)
        text_encoder = CLIPTextModel.from_pretrained(
            cfg.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16
        )
        text_encoder.requires_grad_(False)

        tokenizer = CLIPTokenizer.from_pretrained(
            cfg.model_id,
            subfolder="tokenizer"
        )
        print("<-- Tokenizer loaded")
        sys.stdout.flush()

        # Load transformer (will apply LoRA)
        print("-> Loading transformer subfolder...")
        sys.stdout.flush()
        transformer = SD3Transformer2DModel.from_pretrained(
            cfg.model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        print("<-- Transformer loaded")
        sys.stdout.flush()
    except Exception as e:
        print("Error while loading pre-trained model components:")
        traceback.print_exc()
        # Ensure the process exits with non-zero so Slurm marks it as failed
        sys.exit(2)
    
    # Apply LoRA
    print("\nApplying LoRA to MM-DiT transformer...")
    transformer = setup_lora_model(transformer)
    
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # Create pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Setup dataset
    print("\nLoading datasets...")
    dataset = StyleTransferDataset(
        cfg.content_dir,
        cfg.style_dir,
        cfg.resolution
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup loss functions
    perceptual_loss = VGGPerceptualLoss().to(accelerator.device)
    lpips_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    
    # Prepare with accelerator
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {cfg.num_epochs} epochs")
    print(f"Total batches per epoch: {len(dataloader)}")
    print(f"{'='*60}\n")
    
    global_step = 0
    
    for epoch in range(cfg.num_epochs):
        transformer.train()
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            with accelerator.accumulate(transformer):
                logs, generated_imgs = train_step(
                    batch, pipeline, optimizer, perceptual_loss, 
                    lpips_fn, accelerator, global_step
                )
                
                # Update progress bar
                progress_bar.set_description(
                    f"Epoch {epoch} | Loss: {logs['loss/total']:.4f} | "
                    f"Content: {logs['loss/content']:.4f} | "
                    f"Style: {logs['loss/style']:.4f}"
                )
                
                # Log to wandb
                if accelerator.is_main_process and global_step % 10 == 0:
                    accelerator.log(logs, step=global_step)
                
                global_step += 1
            
            # Save checkpoint
            if global_step % cfg.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    # Save LoRA weights
                    unwrapped_transformer = accelerator.unwrap_model(transformer)
                    unwrapped_transformer.save_pretrained(save_path)
                    
                    print(f"\n✓ Saved checkpoint to {save_path}\n")
    
    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(final_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Training complete! Final weights saved to {final_path}")
        print(f"{'='*60}\n")
    
    accelerator.end_training()

# =============================================================================
# Inference Script
# =============================================================================

def inference_style_transfer(content_path, style_path, lora_path, output_path):
    """Run inference with fine-tuned LoRA weights."""
    
    print("Loading model for inference...")
    
    # Load base pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16
    )
    
    # Load LoRA weights
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        lora_path
    )
    
    pipeline = pipeline.to("cuda")
    
    # Load and preprocess images
    content_img = Image.open(content_path).convert("RGB")
    style_img = Image.open(style_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((cfg.resolution, cfg.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    content_tensor = transform(content_img).unsqueeze(0).to("cuda")
    
    # Encode content to latents
    with torch.no_grad():
        content_latents = pipeline.vae.encode(content_tensor).latent_dist.sample()
        content_latents = content_latents * pipeline.vae.config.scaling_factor
    
    # Generate with img2img
    prompt = "transfer artistic style"
    
    output = pipeline(
        prompt=prompt,
        image=content_latents,
        strength=cfg.img2img_strength,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=7.5
    )
    
    # Save result
    output.images[0].save(output_path)
    print(f"✓ Stylized image saved to {output_path}")

if __name__ == "__main__":
    # Run training
    main()
    
    # Example inference
    # inference_style_transfer(
    #     "test_content.jpg",
    #     "test_style.jpg",
    #     "./sd35_style_transfer_lora/final",
    #     "output_stylized.jpg"
    # )
    #d9a74b72096b984643e4b3246a816e62be94d572 wandb
