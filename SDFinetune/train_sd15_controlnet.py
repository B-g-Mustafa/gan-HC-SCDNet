"""
Stable Diffusion 1.5 + ControlNet Fine-tuning for Style Transfer
Dataset: 60k synthetic style-transferred images with VLM captions
Training: 20 epochs with WandB monitoring
ControlNet: Conditioned on content image to guide style transfer
"""

import os
import torch
import wandb
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.models.controlnet import ControlNetOutput
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# ==================== CONFIGURATION ====================
PROJECT_NAME = "style-transfer-sd15-controlnet"
RUN_NAME = "sd15-controlnet-60k-20epochs"

# Paths
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/"
CSV_PATH = os.path.join(DATASET_ROOT, "captioned_metadata/complete_captioned_metadata.csv")
OUTPUT_DIR = "/home/msai/birul001/BIRUL001/models/sd15_controlnet"

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 3  # ControlNet needs more memory
GRADIENT_ACCUMULATION_STEPS = 5  # Effective batch size = 15
LEARNING_RATE = 1e-4
CONTROLNET_LR = 1e-5  # Lower LR for ControlNet
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
SAVE_EVERY = 5

# ControlNet + LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
CONTROLNET_CONDITIONING_SCALE = 1.0

# WandB Configuration
WANDB_PROJECT = PROJECT_NAME
WANDB_RUN_NAME = RUN_NAME
LOG_EVERY = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{'='*80}")
print(f"Stable Diffusion 1.5 + ControlNet Fine-tuning")
print(f"{'='*80}")
print(f"Device: {device}")
print(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"UNet LR: {LEARNING_RATE}, ControlNet LR: {CONTROLNET_LR}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Output: {OUTPUT_DIR}")

# ==================== DATASET ====================

class ControlNetStyleDataset(Dataset):
    """Dataset with content images as ControlNet conditioning"""
    
    def __init__(self, csv_path, dataset_root):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = dataset_root
        
        # Filter out rows with empty captions
        self.df = self.df[self.df['final_caption'].notna() & (self.df['final_caption'] != '')]
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"Dataset loaded: {len(self.df):,} samples with captions")
        if len(self.df) > 0:
            print(f"Sample caption: {self.df.iloc[0]['final_caption'][:100]}...")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row['final_caption']
        
        # Load content image (for ControlNet conditioning)
        content_path = os.path.join(self.dataset_root, row['content_path'])
        # Load synthetic styled image (target)
        synthetic_path = os.path.join(self.dataset_root, row['synthetic_path'])
        
        try:
            content_img = Image.open(content_path).convert("RGB")
            content_img = self.transform(content_img)
            
            synthetic_img = Image.open(synthetic_path).convert("RGB")
            synthetic_img = self.transform(synthetic_img)
            
            return {
                "caption": caption,
                "content_image": content_img,  # ControlNet conditioning
                "target_image": synthetic_img,  # What we want to generate
                "content_path": content_path,
                "synthetic_path": synthetic_path
            }
        except Exception as e:
            print(f"Error loading images: {e}")
            return {
                "caption": caption,
                "content_image": torch.zeros(3, 512, 512),
                "target_image": torch.zeros(3, 512, 512),
                "content_path": content_path,
                "synthetic_path": synthetic_path
            }

# ==================== MODEL SETUP ====================

def setup_model_and_optimizer():
    """Initialize SD1.5 UNet with LoRA + ControlNet"""
    
    print(f"\n{'='*80}")
    print("Loading Stable Diffusion v1.5 + ControlNet...")
    print(f"{'='*80}")
    
    # Initialize ControlNet (trained from scratch)
    controlnet = ControlNetModel.from_unet(
        UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            torch_dtype=torch.float16
        )
    )
    controlnet.to(device)
    
    # Load base SD pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Move to device
    pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    pipeline.unet.to(device)
    pipeline.controlnet.to(device)
    
    # Freeze VAE and text encoder
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    
    print("\n" + "="*80)
    print("Applying LoRA to UNet...")
    print("="*80)
    
    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    pipeline.unet.print_trainable_parameters()
    
    print("\n" + "="*80)
    print("ControlNet parameters: Trainable")
    print("="*80)
    trainable_params = sum(p.numel() for p in pipeline.controlnet.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Separate optimizers for UNet (LoRA) and ControlNet
    optimizer = torch.optim.AdamW([
        {
            'params': filter(lambda p: p.requires_grad, pipeline.unet.parameters()),
            'lr': LEARNING_RATE,
            'name': 'unet'
        },
        {
            'params': pipeline.controlnet.parameters(),
            'lr': CONTROLNET_LR,
            'name': 'controlnet'
        }
    ], betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    
    return pipeline, optimizer, noise_scheduler

# ==================== TRAINING FUNCTIONS ====================

def encode_images(vae, images):
    """Encode images to latent space"""
    with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=torch.float16)).latent_dist.sample()
        latents = latents * 0.18215
    return latents

def encode_text(tokenizer, text_encoder, prompts):
    """Encode text prompts"""
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    return text_embeddings

def train_one_epoch(pipeline, optimizer, noise_scheduler, train_loader, epoch, global_step):
    """Train for one epoch with ControlNet conditioning"""
    
    pipeline.unet.train()
    pipeline.controlnet.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        try:
            content_images = batch["content_image"]
            target_images = batch["target_image"]
            captions = batch["caption"]
            
            # Encode target (styled) images to latents
            target_latents = encode_images(pipeline.vae, target_images)
            
            # Encode text
            text_embeddings = encode_text(
                pipeline.tokenizer,
                pipeline.text_encoder,
                captions
            )
            
            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (target_latents.shape[0],),
                device=device
            ).long()
            
            # Add noise to target latents
            noise = torch.randn_like(target_latents)
            noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            # Get ControlNet conditioning from content images
            controlnet_image = content_images.to(device, dtype=torch.float16)
            
            # ControlNet forward pass
            down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=controlnet_image,
                conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                return_dict=False,
            )
            
            # UNet prediction with ControlNet residuals
            model_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            # Diffusion loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Update weights
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(pipeline.unet.parameters()) + list(pipeline.controlnet.parameters()),
                    MAX_GRAD_NORM
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log to WandB
                if global_step % LOG_EVERY == 0:
                    wandb.log({
                        "train/loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/unet_lr": optimizer.param_groups[0]['lr'],
                        "train/controlnet_lr": optimizer.param_groups[1]['lr']
                    }, step=global_step)
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({
                'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                'avg_loss': f"{epoch_loss / (step + 1):.4f}"
            })
            
        except Exception as e:
            print(f"\nError in batch {step}: {e}")
            continue
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    return avg_epoch_loss, global_step

# ==================== MAIN TRAINING ====================

def main():
    # Initialize WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": "stable-diffusion-v1.5",
            "method": "lora + controlnet",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "unet_lr": LEARNING_RATE,
            "controlnet_lr": CONTROLNET_LR,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "dataset_size": "60k samples"
        }
    )
    
    # Load dataset
    print(f"\n{'='*80}")
    print("Loading dataset...")
    print(f"{'='*80}")
    dataset = ControlNetStyleDataset(CSV_PATH, DATASET_ROOT)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Total batches per epoch: {len(train_loader):,}")
    
    # Setup model
    pipeline, optimizer, noise_scheduler = setup_model_and_optimizer()
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    global_step = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss, global_step = train_one_epoch(
            pipeline, optimizer, noise_scheduler,
            train_loader, epoch, global_step
        )
        
        print(f"\nEpoch {epoch}/{NUM_EPOCHS} - Avg Loss: {avg_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/number": epoch
        }, step=global_step)
        
        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save LoRA weights
            pipeline.unet.save_pretrained(os.path.join(checkpoint_dir, "unet_lora"))
            
            # Save ControlNet
            pipeline.controlnet.save_pretrained(os.path.join(checkpoint_dir, "controlnet"))
            
            # Save full pipeline
            pipeline.save_pretrained(
                os.path.join(OUTPUT_DIR, f"pipeline-epoch-{epoch}"),
                safe_serialization=True
            )
            
            print(f"✓ Saved checkpoint: {checkpoint_dir}")
            
            wandb.log({
                "checkpoint/epoch": epoch,
                "checkpoint/path": checkpoint_dir
            }, step=global_step)
    
    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    pipeline.unet.save_pretrained(os.path.join(final_dir, "unet_lora"))
    pipeline.controlnet.save_pretrained(os.path.join(final_dir, "controlnet"))
    pipeline.save_pretrained(os.path.join(OUTPUT_DIR, "pipeline-final"), safe_serialization=True)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Final model saved to: {final_dir}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
