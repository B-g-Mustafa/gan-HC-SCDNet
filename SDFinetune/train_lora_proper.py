"""
SD1.5 LoRA Fine-tuning Script with Proper Diffusion Loss
Trains on synthetic images with final captions
"""

import os
import csv
import traceback
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn.functional as F

# ---------------------- CONFIG & PATHS ----------------------
CSV_PATH = "/home/msai/harihara011/mustafa/data/combined.csv"
IMAGE_DIR = "/home/msai/harihara011/mustafa/data/"
OUTPUT_DIR = "/home/msai/harihara011/mustafa/output/lora_training"
LOG_FILE = "/home/msai/harihara011/mustafa/data/lora_training_log.csv"

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 4  # Reduced for memory
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 5  # Save checkpoint every N epochs

# LoRA config
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

# ---------------------- DATA PREPARATION ----------------------

class StyleTransferDataset(Dataset):
    """Dataset for style transfer images with captions"""
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        # Update image paths
        self.df['Image Path'] = self.df['Image Path'].apply(
            lambda x: os.path.join(image_dir, x.replace('data/', ''))
        )
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"Loaded {len(self.df)} samples")
        print(f"Sample row:\n{self.df.iloc[0]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row['Final Caption']
        img_path = row['Image Path']
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return {"caption": caption, "image": image}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return {"caption": caption, "image": torch.zeros(3, 512, 512)}

# ---------------------- MODEL SETUP ----------------------

def setup_model_and_optimizer():
    """Initialize model, LoRA, and optimizer"""
    print("\n=== Loading Stable Diffusion v1.5 ===")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Move components to device
    pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    pipeline.unet.to(device)
    
    # Set to eval mode (we don't train these)
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    
    # Apply LoRA to UNet
    print("\n=== Applying LoRA ===")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    pipeline.unet.print_trainable_parameters()
    
    # Optimizer - only train LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipeline.unet.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    
    return pipeline, optimizer, noise_scheduler

# ---------------------- TRAINING FUNCTIONS ----------------------

def encode_images(vae, images):
    """Encode images to latent space"""
    with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=torch.float16)).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor for SD
    return latents

def encode_text(tokenizer, text_encoder, prompts):
    """Encode text prompts to embeddings"""
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

def train_one_epoch(pipeline, optimizer, noise_scheduler, train_loader, epoch):
    """Train for one epoch"""
    pipeline.unet.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        try:
            images = batch["image"]
            captions = batch["caption"]
            
            # Encode images to latents
            latents = encode_images(pipeline.vae, images)
            
            # Encode text
            text_embeddings = encode_text(
                pipeline.tokenizer,
                pipeline.text_encoder,
                captions
            )
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise to latents
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise using UNet
            model_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Calculate loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Backward pass with gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    pipeline.unet.parameters(),
                    MAX_GRAD_NORM
                )
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})
            
        except Exception as e:
            print(f"\nError in batch {step}: {e}")
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# ---------------------- MAIN TRAINING LOOP ----------------------

def main():
    print("\n" + "="*80)
    print("SD1.5 LoRA Fine-tuning with Proper Diffusion Loss")
    print("="*80)
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = StyleTransferDataset(CSV_PATH, IMAGE_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup model
    pipeline, optimizer, noise_scheduler = setup_model_and_optimizer()
    
    # Initialize logging
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "learning_rate"])
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_loss = float("inf")
    best_epoch = -1
    
    for epoch in range(1, NUM_EPOCHS + 1):
        try:
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
            
            # Train
            train_loss = train_one_epoch(
                pipeline, optimizer, noise_scheduler, train_loader, epoch
            )
            
            # Log
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, round(train_loss, 6), LEARNING_RATE])
            
            print(f"Epoch {epoch} - Average Loss: {train_loss:.6f}")
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                save_path = os.path.join(OUTPUT_DIR, "best_model")
                pipeline.save_pretrained(save_path)
                print(f"✓ Saved best model to: {save_path}")
            
            # Periodic checkpoint
            if epoch % SAVE_EVERY == 0:
                save_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}")
                pipeline.save_pretrained(save_path)
                print(f"✓ Saved checkpoint to: {save_path}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            error_msg = f"Error at epoch {epoch}: {str(e)}"
            print(error_msg)
            with open(os.path.join(OUTPUT_DIR, "error_log.txt"), "a") as ef:
                ef.write(error_msg + "\n" + traceback.format_exc() + "\n")
    
    # Final summary
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best Epoch: {best_epoch} | Best Loss: {best_loss:.6f}")
    print(f"Logs saved to: {LOG_FILE}")
    print(f"Models saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
