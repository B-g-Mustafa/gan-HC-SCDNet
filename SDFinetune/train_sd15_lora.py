"""
Stable Diffusion 1.5 LoRA Fine-tuning for Style Transfer
Dataset: 60k synthetic style-transferred images with VLM captions
Training: 20 epochs with WandB monitoring
"""

import os
import torch
import wandb
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

# ==================== CONFIGURATION ====================
PROJECT_NAME = "style-transfer-sd15-lora"
RUN_NAME = "sd15-lora-60k-20epochs"

# Paths
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/"
CSV_PATH = os.path.join(DATASET_ROOT, "captioned_metadata/complete_captioned_metadata.csv")
OUTPUT_DIR = "/home/msai/birul001/gan-project/gan-HC-SCDNet/models/sd15_lora"

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 16
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
SAVE_EVERY = 5  # Save checkpoint every N epochs
EVAL_EVERY = 5  # Run evaluation every N epochs
TRAIN_SPLIT = 0.95  # 95% train, 5% eval (57k train, 3k eval)

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# WandB Configuration
WANDB_PROJECT = PROJECT_NAME
WANDB_RUN_NAME = RUN_NAME
LOG_EVERY = 50  # Log metrics every N steps

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{'='*80}")
print(f"Stable Diffusion 1.5 LoRA Fine-tuning")
print(f"{'='*80}")
print(f"Device: {device}")
print(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Output: {OUTPUT_DIR}")

# ==================== DATASET ====================

class StyleTransferDataset(Dataset):
    """Dataset for style-transferred images with VLM captions"""
    
    def __init__(self, csv_path, dataset_root, split='train', train_ratio=0.95):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = dataset_root
        self.split = split
        
        # Filter out rows with empty captions
        self.df = self.df[self.df['final_caption'].notna() & (self.df['final_caption'] != '')]
        
        # Split into train/eval (do not modify original CSV)
        total_samples = len(self.df)
        train_size = int(total_samples * train_ratio)
        
        if split == 'train':
            self.df = self.df.iloc[:train_size].reset_index(drop=True)
        elif split == 'eval':
            self.df = self.df.iloc[train_size:].reset_index(drop=True)
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        print(f"{split.upper()} dataset loaded: {len(self.df):,} samples")
        if len(self.df) > 0:
            print(f"  Sample caption: {self.df.iloc[0]['final_caption'][:100]}...")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row['final_caption']
        # synthetic_path is relative to dataset root
        img_path = os.path.join(self.dataset_root, row['synthetic_path'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return {"caption": caption, "image": image, "path": img_path}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            return {"caption": caption, "image": torch.zeros(3, 512, 512), "path": img_path}

# ==================== MODEL SETUP ====================

def setup_model_and_optimizer():
    """Initialize SD1.5 model with LoRA and optimizer"""
    
    print(f"\n{'='*80}")
    print("Loading Stable Diffusion v1.5...")
    print(f"{'='*80}")
    
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            variant="fp16",
            use_safetensors=True
        )
    except Exception as e:
        print(f"Error loading pipeline with fp16 variant: {e}")
        print("Trying without variant...")
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
        except Exception as e2:
            print(f"Error loading pipeline: {e2}")
            print("Trying with float32...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                safety_checker=None,
                requires_safety_checker=False
            )
    
    # Move to device
    pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    pipeline.unet.to(device)
    
    # Freeze VAE and text encoder
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    
    print("\n" + "="*80)
    print("Applying LoRA to UNet...")
    print("="*80)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    pipeline.unet.print_trainable_parameters()
    
    # Optimizer - only LoRA parameters
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

# ==================== TRAINING FUNCTIONS ====================

def encode_images(vae, images):
    """Encode images to latent space"""
    with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=torch.float16)).latent_dist.sample()
        latents = latents * 0.18215  # SD scaling factor
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

def train_one_epoch(pipeline, optimizer, noise_scheduler, train_loader, epoch, global_step):
    """Train for one epoch with proper diffusion loss"""
    
    pipeline.unet.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        try:
            images = batch["image"]
            captions = batch["caption"]
            
            # Encode to latents
            latents = encode_images(pipeline.vae, images)
            
            # Encode text
            text_embeddings = encode_text(
                pipeline.tokenizer,
                pipeline.text_encoder,
                captions
            )
            
            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Diffusion loss (MSE between predicted and actual noise)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Update weights
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(pipeline.unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log to WandB
                if global_step % LOG_EVERY == 0:
                    wandb.log({
                        "train/loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/lr": optimizer.param_groups[0]['lr']
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

@torch.no_grad()
def evaluate(pipeline, noise_scheduler, eval_loader, epoch):
    """Evaluate on evaluation set"""
    
    pipeline.unet.eval()
    eval_loss = 0.0
    num_batches = 0
    
    print(f"\n{'='*80}")
    print(f"Running Evaluation at Epoch {epoch}")
    print(f"{'='*80}")
    
    progress_bar = tqdm(eval_loader, desc=f"Evaluating")
    
    for batch in progress_bar:
        try:
            images = batch["image"]
            captions = batch["caption"]
            
            # Encode to latents
            latents = encode_images(pipeline.vae, images)
            
            # Encode text
            text_embeddings = encode_text(
                pipeline.tokenizer,
                pipeline.text_encoder,
                captions
            )
            
            # Sample timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Evaluation loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            eval_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'eval_loss': f"{loss.item():.4f}"})
            
        except Exception as e:
            print(f"\nError in evaluation batch: {e}")
            continue
    
    avg_eval_loss = eval_loss / num_batches if num_batches > 0 else float('inf')
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results at Epoch {epoch}")
    print(f"{'='*80}")
    print(f"  Average Evaluation Loss: {avg_eval_loss:.4f}")
    print(f"  Total Evaluation Batches: {num_batches}")
    print(f"{'='*80}\n")
    
    return avg_eval_loss

# ==================== MAIN TRAINING ====================

def main():
    # Print dependency versions first
    print(f"\n{'='*80}")
    print("DEPENDENCY VERSIONS CHECK")
    print(f"{'='*80}")
    try:
        import torch
        import transformers
        import diffusers
        import peft
        import wandb
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
        print(f"Diffusers: {diffusers.__version__}")
        print(f"PEFT: {peft.__version__}")
        print(f"WandB: {wandb.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error checking dependencies: {e}")
    print(f"{'='*80}")
    
    # Initialize WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": "stable-diffusion-v1.5",
            "method": "lora",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "dataset_size": "60k samples"
        }
    )
    
    # Load train and eval datasets
    print(f"\n{'='*80}")
    print("Loading datasets...")
    print(f"{'='*80}")
    
    train_dataset = StyleTransferDataset(CSV_PATH, DATASET_ROOT, split='train', train_ratio=TRAIN_SPLIT)
    eval_dataset = StyleTransferDataset(CSV_PATH, DATASET_ROOT, split='eval', train_ratio=TRAIN_SPLIT)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTraining batches per epoch: {len(train_loader):,}")
    print(f"Evaluation batches: {len(eval_loader):,}")
    
    # Setup model
    pipeline, optimizer, noise_scheduler = setup_model_and_optimizer()
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting training...")
    print(f"{'='*80}\n")
    
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        avg_train_loss, global_step = train_one_epoch(
            pipeline, optimizer, noise_scheduler,
            train_loader, epoch, global_step
        )
        
        # Print detailed epoch summary
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS} SUMMARY")
        print(f"{'='*80}")
        print(f"  Training Loss:        {avg_train_loss:.6f}")
        print(f"  Learning Rate:        {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Batch Size:           {BATCH_SIZE}")
        print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"  LoRA Rank:            {LORA_RANK}")
        print(f"  LoRA Alpha:           {LORA_ALPHA}")
        print(f"  Global Steps:         {global_step}")
        
        # Run evaluation every EVAL_EVERY epochs
        eval_loss = None
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            eval_loss = evaluate(pipeline, noise_scheduler, eval_loader, epoch)
            print(f"  Evaluation Loss:      {eval_loss:.6f}")
            
            # Track best model
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print(f"  >>> NEW BEST MODEL! (Eval Loss: {best_eval_loss:.6f})")
        
        print(f"{'='*80}\n")
        
        # Log epoch metrics to WandB
        log_dict = {
            "epoch/train_loss": avg_train_loss,
            "epoch/number": epoch,
            "epoch/learning_rate": optimizer.param_groups[0]['lr']
        }
        if eval_loss is not None:
            log_dict["epoch/eval_loss"] = eval_loss
            log_dict["epoch/best_eval_loss"] = best_eval_loss
        
        wandb.log(log_dict, step=global_step)
        
        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save LoRA weights
            pipeline.unet.save_pretrained(checkpoint_dir)
            
            # Save full pipeline for easy inference
            pipeline.save_pretrained(
                os.path.join(OUTPUT_DIR, f"pipeline-epoch-{epoch}"),
                safe_serialization=True
            )
            
            print(f"✓ Saved checkpoint: {checkpoint_dir}")
            
            # Log to WandB
            wandb.log({
                "checkpoint/epoch": epoch,
                "checkpoint/path": checkpoint_dir
            }, step=global_step)
    
    # Save final model
    final_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    pipeline.unet.save_pretrained(final_dir)
    pipeline.save_pretrained(os.path.join(OUTPUT_DIR, "pipeline-final"), safe_serialization=True)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Final model saved to: {final_dir}")
    print(f"Best Evaluation Loss: {best_eval_loss:.6f}")
    print(f"\nHyperparameters Used:")
    print(f"  Learning Rate:        {LEARNING_RATE:.2e}")
    print(f"  Batch Size:           {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs:               {NUM_EPOCHS}")
    print(f"  LoRA Rank:            {LORA_RANK}")
    print(f"  LoRA Alpha:           {LORA_ALPHA}")
    print(f"  LoRA Dropout:         {LORA_DROPOUT}")
    print(f"  Train/Eval Split:     {TRAIN_SPLIT:.2%} / {1-TRAIN_SPLIT:.2%}")
    print(f"{'='*80}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
