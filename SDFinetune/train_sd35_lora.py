"""
Stable Diffusion 3.5 Large LoRA Fine-tuning for Style Transfer
Dataset: 60k synthetic style-transferred images with VLM captions
Training: 20 epochs with WandB monitoring
Note: SD3.5 uses a different architecture (MMDiT) compared to SD1.5
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
import argparse

# ==================== CONFIGURATION ====================
PROJECT_NAME = "style-transfer-sd35-lora"
RUN_NAME = "sd35-lora-60k-20epochs-distributed"

# Paths
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/"
CSV_PATH = os.path.join(DATASET_ROOT, "captioned_metadata/complete_captioned_metadata.csv")
OUTPUT_DIR = "/home/msai/birul001/BIRUL001/models/sd35_lora"

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 4  # Optimized for A100 40GB with mixed precision
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size per GPU = 16, total = 32 across 2 GPUs
LEARNING_RATE = 5e-5  # Lower LR for larger model
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
SAVE_EVERY = 5

# LoRA Configuration
LORA_RANK = 32  # Higher rank for SD3.5
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# WandB Configuration
WANDB_PROJECT = PROJECT_NAME
WANDB_RUN_NAME = RUN_NAME
LOG_EVERY = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== DISTRIBUTED SETUP ====================

def parse_args():
    """Parse command line arguments for distributed training"""
    parser = argparse.ArgumentParser(description='SD3.5 LoRA Distributed Training')
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID (0 or 1)")
    parser.add_argument("--world_size", type=int, default=2, help="Total number of GPUs")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master_port", type=str, default="12355", help="Master port")
    return parser.parse_args()

def setup_distributed(gpu_id, world_size, master_addr, master_port):
    """Initialize distributed training for manual process setup"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(gpu_id)
    os.environ['LOCAL_RANK'] = str(gpu_id)
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        rank=gpu_id,
        world_size=world_size
    )
    
    # Set device
    torch.cuda.set_device(gpu_id)
    
    print(f"\n{'='*80}")
    print(f"🚀 DISTRIBUTED TRAINING - PROCESS {gpu_id}/{world_size-1}")
    print(f"{'='*80}")
    print(f"✓ Process Rank: {gpu_id}")
    print(f"✓ GPU Device: cuda:{gpu_id}")
    print(f"✓ World Size: {world_size}")
    print(f"✓ Master: {master_addr}:{master_port}")
    print(f"✓ Backend: NCCL")
    print(f"{'='*80}")
    
    return gpu_id

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def print_training_start(rank, world_size, is_main):
    """Print training start message with process info"""
    if is_main:
        print(f"\n{'='*80}")
        print(f"🎯 STARTING DISTRIBUTED FINE-TUNING")
        print(f"{'='*80}")
        print(f"Total GPUs: {world_size}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch per GPU: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"Total effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size}")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"GPU Memory: A100 40GB")
        print(f"{'='*80}\n")
    else:
        print(f"\n[GPU {rank}] ⚡ Worker process ready and waiting for main process...\n")

# ==================== DATASET ====================

class StyleTransferDataset(Dataset):
    """Dataset for style-transferred images with VLM captions"""
    
    def __init__(self, csv_path, dataset_root):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = dataset_root
        
        # Filter out rows with empty captions
        self.df = self.df[self.df['final_caption'].notna() & (self.df['final_caption'] != '')]
        
        # SD3.5 supports 1024x1024
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = row['final_caption']
        img_path = os.path.join(self.dataset_root, row['synthetic_path'])
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return {"caption": caption, "image": image, "path": img_path}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return {"caption": caption, "image": torch.zeros(3, 1024, 1024), "path": img_path}

def create_dataloader(dataset, batch_size, rank=0, world_size=1, is_distributed=False):
    """Create dataloader with optional distributed sampling"""
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    ), sampler

# ==================== MODEL SETUP ====================

def setup_model_and_optimizer(rank, world_size, is_distributed=False, is_main=True):
    """Initialize SD3.5 model with LoRA and optimizer"""
    
    device = f"cuda:{rank}"
    
    if is_main:
        print(f"\n{'='*80}")
        print("📦 LOADING MODEL - STABLE DIFFUSION 3.5 LARGE")
        print(f"{'='*80}")
    else:
        print(f"[GPU {rank}] 📦 Loading model on cuda:{rank}...")
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.float16,
    )
    
    # Move to specific GPU device
    pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    pipeline.text_encoder_2.to(device)
    pipeline.text_encoder_3.to(device)
    pipeline.transformer.to(device)
    
    if not is_main:
        print(f"[GPU {rank}] ✓ Model loaded on {device}")
    
    # Freeze VAE and text encoders
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.text_encoder_2.eval()
    pipeline.text_encoder_3.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    
    if is_main:
        print("\n" + "="*80)
        print("🔧 APPLYING LORA TO TRANSFORMER")
        print("="*80)
    else:
        print(f"[GPU {rank}] 🔧 Applying LoRA...")
    
    # Apply LoRA to transformer blocks
    # SD3.5 uses MMDiT transformer instead of UNet
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)
    
    if is_main:
        pipeline.transformer.print_trainable_parameters()
    
    # Wrap with DDP for distributed training
    if is_distributed:
        pipeline.transformer = DDP(
            pipeline.transformer,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
        if is_main:
            print(f"✓ Transformer wrapped with DistributedDataParallel")
        else:
            print(f"[GPU {rank}] ✓ DDP wrapper applied")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipeline.transformer.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # SD3.5 uses Flow Matching scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        subfolder="scheduler"
    )
    
    return pipeline, optimizer, noise_scheduler

# ==================== TRAINING FUNCTIONS ====================

def encode_images(vae, images, device):
    """Encode images to latent space"""
    with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=torch.float16)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents

def encode_text(pipeline, prompts, device):
    """Encode text with all three text encoders (SD3.5 uses T5, CLIP-L, CLIP-G)"""
    
    # Encode with text_encoder (CLIP-L)
    text_inputs_1 = pipeline.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Encode with text_encoder_2 (CLIP-G)
    text_inputs_2 = pipeline.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=pipeline.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Encode with text_encoder_3 (T5)
    text_inputs_3 = pipeline.tokenizer_3(
        prompts,
        padding="max_length",
        max_length=256,  # T5 max length
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        prompt_embeds_1 = pipeline.text_encoder(text_inputs_1.input_ids.to(device))[0]
        prompt_embeds_2 = pipeline.text_encoder_2(text_inputs_2.input_ids.to(device))[0]
        prompt_embeds_3 = pipeline.text_encoder_3(text_inputs_3.input_ids.to(device))[0]
    
    # Concatenate embeddings (SD3.5 concatenates all three)
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2, prompt_embeds_3], dim=-1)
    
    return prompt_embeds

def train_one_epoch(pipeline, optimizer, noise_scheduler, train_loader, epoch, global_step, rank, world_size, is_main=True):
    """Train for one epoch with flow matching loss"""
    
    device = f"cuda:{rank}"
    pipeline.transformer.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    
    # Print epoch start for each GPU
    if is_main:
        print(f"\n{'='*80}")
        print(f"🏋️  EPOCH {epoch}/{NUM_EPOCHS} - TRAINING IN PROGRESS")
        print(f"{'='*80}")
        progress_bar = tqdm(train_loader, desc=f"[MAIN GPU 0] Epoch {epoch}/{NUM_EPOCHS}")
    else:
        print(f"\n[GPU {rank}] 🏋️  Epoch {epoch}/{NUM_EPOCHS} - Fine-tuning in progress...")
        progress_bar = train_loader
    
    for step, batch in enumerate(progress_bar):
        try:
            images = batch["image"]
            captions = batch["caption"]
            
            # Encode to latents
            latents = encode_images(pipeline.vae, images, device)
            
            # Encode text (3 encoders)
            text_embeddings = encode_text(pipeline, captions, device)
            
            # Sample timesteps (SD3.5 uses flow matching)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise (flow matching interpolation)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict velocity (SD3.5 predicts velocity, not noise)
            model_pred = pipeline.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Flow matching loss
            # Target is the velocity from noise to latent
            target = latents - noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Update weights
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(pipeline.transformer.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Log to WandB (only main process)
                if is_main and global_step % LOG_EVERY == 0:
                    wandb.log({
                        "train/loss": loss.item() * GRADIENT_ACCUMULATION_STEPS,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/lr": optimizer.param_groups[0]['lr']
                    }, step=global_step)
            
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            if is_main:
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                    'avg_loss': f"{epoch_loss / (step + 1):.4f}",
                    'gpu': f"{rank}"
                })
            elif step % 100 == 0:
                print(f"[GPU {rank}] Step {step}/{len(train_loader)} - Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            
        except Exception as e:
            if is_main:
                print(f"\n[GPU {rank}] Error in batch {step}: {e}")
            continue
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    return avg_epoch_loss, global_step

# ==================== MAIN TRAINING ====================

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed training
    rank = setup_distributed(args.gpu, args.world_size, args.master_addr, args.master_port)
    is_main_process = rank == 0
    is_distributed = args.world_size > 1
    
    # Print training configuration
    print_training_start(rank, args.world_size, is_main_process)
    
    # Initialize WandB only on main process
    if is_main_process:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"{WANDB_RUN_NAME}-{args.world_size}gpu",
            config={
            "model": "stable-diffusion-3.5-large",
            "method": "lora",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size_per_gpu": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "total_effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * args.world_size,
            "learning_rate": LEARNING_RATE,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "dataset_size": "60k samples",
            "resolution": "1024x1024",
            "distributed": is_distributed,
            "world_size": args.world_size,
            "gpu_type": "NVIDIA A100 40GB"
        }
    )
        print(f"✓ WandB initialized: {WANDB_PROJECT}/{WANDB_RUN_NAME}")
    else:
        print(f"[GPU {rank}] Skipping WandB initialization (worker process)")
    
    # Load dataset
    if is_main_process:
        print(f"\n{'='*80}")
        print("📊 LOADING DATASET")
        print(f"{'='*80}")
    else:
        print(f"[GPU {rank}] 📊 Loading dataset...")
    
    dataset = StyleTransferDataset(CSV_PATH, DATASET_ROOT)
    train_loader, train_sampler = create_dataloader(
        dataset, BATCH_SIZE, rank, args.world_size, is_distributed
    )
    
    if is_main_process:
        print(f"✓ Dataset loaded: {len(dataset):,} samples")
        print(f"✓ Batches per GPU per epoch: {len(train_loader):,}")
        print(f"✓ Samples per GPU: {len(dataset) // args.world_size:,}")
        if len(dataset) > 0:
            print(f"✓ Sample caption: {dataset.df.iloc[0]['final_caption'][:80]}...")
    else:
        print(f"[GPU {rank}] ✓ Dataset ready: {len(dataset) // args.world_size:,} samples")
    
    # Setup model
    pipeline, optimizer, noise_scheduler = setup_model_and_optimizer(
        rank, args.world_size, is_distributed, is_main_process
    )
    
    # Training loop
    if is_main_process:
        print(f"\n{'='*80}")
        print("🚀 STARTING DISTRIBUTED TRAINING LOOP")
        print(f"{'='*80}\n")
    else:
        print(f"[GPU {rank}] 🚀 Ready to start training loop\n")
    
    global_step = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Set epoch for distributed sampler (important for proper shuffling)
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        avg_loss, global_step = train_one_epoch(
            pipeline, optimizer, noise_scheduler,
            train_loader, epoch, global_step, rank, args.world_size, is_main_process
        )
        
        if is_main_process:
            print(f"\n{'='*80}")
            print(f"✓ Epoch {epoch}/{NUM_EPOCHS} Complete - Avg Loss: {avg_loss:.4f}")
            print(f"{'='*80}")
        else:
            print(f"\n[GPU {rank}] ✓ Epoch {epoch}/{NUM_EPOCHS} complete - Avg Loss: {avg_loss:.4f}")
        
        # Log epoch metrics (main process only)
        if is_main_process:
            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/number": epoch
            }, step=global_step)
        
        # Save checkpoint (main process only)
        if is_main_process and (epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS):
            checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save LoRA weights (unwrap DDP if needed)
            model_to_save = pipeline.transformer.module if hasattr(pipeline.transformer, 'module') else pipeline.transformer
            model_to_save.save_pretrained(checkpoint_dir)
            
            # Save full pipeline
            pipeline.save_pretrained(
                os.path.join(OUTPUT_DIR, f"pipeline-epoch-{epoch}"),
                safe_serialization=True
            )
            
            print(f"💾 Checkpoint saved: {checkpoint_dir}")
            
            wandb.log({
                "checkpoint/epoch": epoch,
                "checkpoint/path": checkpoint_dir
            }, step=global_step)
    
    # Save final model (main process only)
    if is_main_process:
        final_dir = os.path.join(OUTPUT_DIR, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        model_to_save = pipeline.transformer.module if hasattr(pipeline.transformer, 'module') else pipeline.transformer
        model_to_save.save_pretrained(final_dir)
        pipeline.save_pretrained(os.path.join(OUTPUT_DIR, "pipeline-final"), safe_serialization=True)
        
        print(f"\n{'='*80}")
        print("🎉 TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"✓ Final model saved to: {final_dir}")
        print(f"✓ Total epochs: {NUM_EPOCHS}")
        print(f"✓ GPUs used: {args.world_size}")
        print(f"{'='*80}\n")
        
        wandb.finish()
    else:
        print(f"\n[GPU {rank}] 🎉 Training complete! Worker process finished.")
    
    # Cleanup distributed
    cleanup_distributed()
    
    if is_main_process:
        print("✓ Distributed training cleaned up successfully")

if __name__ == "__main__":
    main()
