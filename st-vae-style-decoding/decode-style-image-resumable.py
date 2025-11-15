"""
Optimized Batch-based Style Transfer Dataset Generator with Resume Support
- Ensures unique content images (no duplicates)
- Processes images in batches for GPU efficiency
- Creates organized dataset structure
- Saves CSV metadata batch by batch
- Copies original content and style images
- SUPPORTS RESUMING: Can add more samples later without duplicating content images
"""

import random
import torch
import pandas as pd
from tqdm import tqdm
import os
import shutil
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer
from pathlib import Path

# ==================== CONFIGURATION ====================
print("="*80)
print("Batch-based Style Transfer Dataset Generator (with Resume Support)")
print("="*80)

# Source directories
CONTENT_DIR = "/home/msai/birul001/BIRUL001/data/coco/train2017/"
STYLE_DIR = "/home/msai/birul001/BIRUL001/data/wikiart/split50/train"

# Output dataset structure
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/"
OUTPUT_CONTENT_DIR = os.path.join(DATASET_ROOT, "content_images")
OUTPUT_STYLE_DIR = os.path.join(DATASET_ROOT, "style_images")
OUTPUT_SYNTHETIC_DIR = os.path.join(DATASET_ROOT, "style_transferred_images")
OUTPUT_CSV_DIR = os.path.join(DATASET_ROOT, "metadata")
METADATA_PATH = os.path.join(OUTPUT_CSV_DIR, "complete_metadata.csv")
USED_CONTENT_TRACKER = os.path.join(OUTPUT_CSV_DIR, "used_content_images.txt")

# Generation parameters
TARGET_SAMPLES = 60000
STYLE_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
BATCH_SIZE = 16  # Optimal for A100 40GB (can adjust up to 32)
SAMPLES_PER_CONTENT = 5  # Each content image used 5 times with different styles

# ==================== RESUME CONFIGURATION ====================
# IMPORTANT: Set this to True when you want to add MORE samples later
# This will automatically skip content images you've already used
RESUME_MODE = False  # Set to True to add more samples without duplicates

# Model paths
VGG_PATH = '/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/vgg_r41.pth'
DEC_PATH = '/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/dec_r41.pth'
MATRIX_PATH = '/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/matrix_r41_new.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Target samples: {TARGET_SAMPLES}")
print(f"Samples per content image: {SAMPLES_PER_CONTENT}")
print(f"Resume mode: {'ENABLED' if RESUME_MODE else 'DISABLED'}")

# ==================== SETUP ====================

# Create output directories
os.makedirs(OUTPUT_CONTENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_STYLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_SYNTHETIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

print("\nCreated dataset structure:")
print(f"  - Content images: {OUTPUT_CONTENT_DIR}")
print(f"  - Style images: {OUTPUT_STYLE_DIR}")
print(f"  - Synthetic images: {OUTPUT_SYNTHETIC_DIR}")
print(f"  - Metadata CSVs: {OUTPUT_CSV_DIR}")

# Load models
print("\n" + "="*80)
print("Loading models...")
vgg = encoder4()
dec = decoder4()
matrix = MulLayer(z_dim=256)

vgg.load_state_dict(torch.load(VGG_PATH, map_location=torch.device(device)))
dec.load_state_dict(torch.load(DEC_PATH, map_location=torch.device(device)))
matrix.load_state_dict(torch.load(MATRIX_PATH, map_location=torch.device(device)))

vgg.to(device).eval()
dec.to(device).eval()
matrix.to(device).eval()
print("✓ Models loaded and set to eval mode")

# Image transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# ==================== PREPARE FILE LISTS ====================

print("\n" + "="*80)
print("Scanning directories...")

# Get all content files
content_files = sorted([
    os.path.join(CONTENT_DIR, f) 
    for f in os.listdir(CONTENT_DIR) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

# Get all style files
style_files = []
for root, dirs, files in os.walk(STYLE_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            style_files.append(os.path.join(root, f))

print(f"✓ Found {len(content_files)} content images")
print(f"✓ Found {len(style_files)} style images")

# ==================== RESUME LOGIC ====================
# RESUME TECHNIQUE:
# 1. Use a fixed random seed (42) so content_files are always shuffled the same way
# 2. Save a list of used content image paths to a tracker file
# 3. On resume, load the tracker file and filter out already-used images
# 4. This ensures we NEVER reuse the same content image across runs

# Use fixed seed for reproducible shuffling
# This means content_files will ALWAYS be in the same order across different runs
random.seed(42)
random.shuffle(content_files)

# Load previously used content images if resuming
previously_used_content = set()
content_idx_offset = 0  # Tracks how many content images were used in previous runs

if RESUME_MODE and os.path.exists(USED_CONTENT_TRACKER):
    print("\n" + "="*80)
    print("RESUME MODE: Loading previously used content images...")
    
    with open(USED_CONTENT_TRACKER, 'r') as f:
        previously_used_content = set(f.read().splitlines())
    
    print(f"✓ Found {len(previously_used_content)} previously used content images")
    
    # Calculate offset: how many images were already used
    # This helps us continue numbering from where we left off
    content_idx_offset = len(previously_used_content)
    
    # Filter out already-used content images
    # Only keep images that are NOT in the previously_used_content set
    content_files = [f for f in content_files if f not in previously_used_content]
    
    print(f"✓ Filtered to {len(content_files)} new content images")
    print(f"✓ Content indexing will start from {content_idx_offset}")
    print("="*80)

# Calculate required unique content images
required_content_images = TARGET_SAMPLES // SAMPLES_PER_CONTENT

if required_content_images > len(content_files):
    print(f"\n⚠ Warning: Need {required_content_images} unique content images but only {len(content_files)} available")
    print(f"  Adjusting target to {len(content_files) * SAMPLES_PER_CONTENT} samples")
    TARGET_SAMPLES = len(content_files) * SAMPLES_PER_CONTENT
    required_content_images = len(content_files)

print(f"\n✓ Will use {required_content_images} unique content images")
print(f"✓ Total samples to generate: {TARGET_SAMPLES}")

# Select unique content images (already filtered if resuming)
selected_content_files = content_files[:required_content_images]

print(f"✓ Selected {len(selected_content_files)} unique content images")

# ==================== BATCH PROCESSING ====================

def process_batch(batch_data):
    """Process a batch of image triplets"""
    batch_contents = []
    batch_styles = []
    batch_weights = []
    
    for item in batch_data:
        content_img = Image.open(item['content_path']).convert('RGB')
        style_img = Image.open(item['style_path']).convert('RGB')
        
        batch_contents.append(transform(content_img))
        batch_styles.append(transform(style_img))
        batch_weights.append(item['weight'])
    
    # Stack into batches
    content_batch = torch.stack(batch_contents).to(device)
    style_batch = torch.stack(batch_styles).to(device)
    weights = torch.tensor(batch_weights, device=device)
    
    # Forward pass
    with torch.no_grad():
        sF_batch = vgg(style_batch)
        cF_batch = vgg(content_batch)
        
        # Process each item in batch with its weight
        predictions = []
        for i in range(len(batch_data)):
            _, _, _, out2 = matrix(
                cF_batch['r41'][i:i+1], 
                sF_batch['r41'][i:i+1], 
                weights[i].item()
            )
            prediction = dec(out2)
            predictions.append(prediction)
        
        predictions = torch.cat(predictions, dim=0)
    
    # Convert to numpy for saving
    predictions = predictions.cpu().permute(0, 2, 3, 1).numpy()
    predictions = (predictions * 255.0).clip(0, 255).astype(np.uint8)
    
    return predictions

def copy_image(src_path, dst_dir, new_name):
    """Copy image to destination with new name"""
    dst_path = os.path.join(dst_dir, new_name)
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    return dst_path

# ==================== MAIN GENERATION LOOP ====================

print("\n" + "="*80)
print("Starting batch processing...")
print("="*80)

all_metadata = []
batch_data = []

# In resume mode, continue indexing from where we left off
# This ensures synthetic_000000.jpg, synthetic_000001.jpg don't get overwritten
if RESUME_MODE and os.path.exists(METADATA_PATH):
    # Load existing metadata to get the last global index
    existing_df = pd.read_csv(METADATA_PATH)
    # Extract the last synthetic index from the last row
    last_synthetic = existing_df['synthetic_path'].iloc[-1]
    # Extract number from "style_transferred_images/synthetic_XXXXXX.jpg"
    global_idx = int(last_synthetic.split('_')[-1].split('.')[0]) + 1
    print(f"\n⚠ RESUME MODE: Starting synthetic image indexing from {global_idx}")
else:
    global_idx = 0

batch_num = 0

# Create generation plan
# Each content image will be used SAMPLES_PER_CONTENT times with different styles
generation_plan = []
for local_idx, content_path in enumerate(selected_content_files):
    # Calculate actual content index (offset by previously used images)
    actual_content_idx = local_idx + content_idx_offset
    
    for sample_num in range(SAMPLES_PER_CONTENT):
        style_path = random.choice(style_files)
        weight = random.choice(STYLE_WEIGHTS)
        seed = random.randint(0, int(1e6))
        
        generation_plan.append({
            'content_path': content_path,
            'content_idx': actual_content_idx,  # Use offset index for consistent naming
            'style_path': style_path,
            'weight': weight,
            'seed': seed
        })

print(f"✓ Created generation plan with {len(generation_plan)} samples\n")

# Process in batches
total_batches = (len(generation_plan) + BATCH_SIZE - 1) // BATCH_SIZE

with tqdm(total=len(generation_plan), desc="Generating dataset", unit="img") as pbar:
    for i in range(0, len(generation_plan), BATCH_SIZE):
        batch_plan = generation_plan[i:i+BATCH_SIZE]
        
        try:
            # Set seeds for reproducibility
            for item in batch_plan:
                random.seed(item['seed'])
                torch.manual_seed(item['seed'])
            
            # Process batch
            predictions = process_batch(batch_plan)
            
            # Save images and metadata
            for j, (item, prediction) in enumerate(zip(batch_plan, predictions)):
                # Generate filenames
                # Content filename uses actual_content_idx for consistent numbering
                content_filename = f"content_{item['content_idx']:06d}.jpg"
                style_filename = f"style_{global_idx:06d}.jpg"
                synthetic_filename = f"synthetic_{global_idx:06d}.jpg"
                
                # Copy content image (only once per unique content)
                # If file already exists from previous run, this is a no-op
                content_dst = copy_image(
                    item['content_path'],
                    OUTPUT_CONTENT_DIR,
                    content_filename
                )
                
                # Copy style image
                style_dst = copy_image(
                    item['style_path'],
                    OUTPUT_STYLE_DIR,
                    style_filename
                )
                
                # Save synthetic image
                synthetic_path = os.path.join(OUTPUT_SYNTHETIC_DIR, synthetic_filename)
                Image.fromarray(prediction).save(synthetic_path, quality=95)
                
                # Store metadata with relative paths
                all_metadata.append({
                    'content_path': f"content_images/{content_filename}",
                    'style_path': f"style_images/{style_filename}",
                    'synthetic_path': f"style_transferred_images/{synthetic_filename}",
                    'weight': item['weight'],
                    'seed': item['seed']
                })
                
                global_idx += 1
            
            # Save batch CSV periodically
            batch_num += 1
            if batch_num % 10 == 0 or i + BATCH_SIZE >= len(generation_plan):
                batch_csv_path = os.path.join(
                    OUTPUT_CSV_DIR, 
                    f"metadata_batch_{batch_num:04d}.csv"
                )
                pd.DataFrame(all_metadata[-len(batch_plan):]).to_csv(
                    batch_csv_path, 
                    index=False
                )
            
            # Update progress
            pbar.update(len(batch_plan))
            pbar.set_postfix({
                'batch': f"{batch_num}/{total_batches}",
                'generated': global_idx
            })
            
            # Clear CUDA cache periodically
            if batch_num % 20 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n✗ Error in batch {batch_num}: {e}")
            continue

# ==================== SAVE TRACKING INFO ====================

print("\n" + "="*80)
print("Saving metadata and tracking info...")

# Save/append to complete metadata
new_df = pd.DataFrame(all_metadata)

if RESUME_MODE and os.path.exists(METADATA_PATH):
    # In resume mode, APPEND to existing metadata
    existing_df = pd.read_csv(METADATA_PATH)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(METADATA_PATH, index=False)
    print(f"✓ Appended {len(new_df)} new samples to existing metadata")
    print(f"✓ Total samples in dataset: {len(combined_df)}")
else:
    # First run: create new metadata file
    new_df.to_csv(METADATA_PATH, index=False)
    print(f"✓ Saved complete metadata to: {METADATA_PATH}")
    print(f"✓ Total samples generated: {len(new_df)}")

# ==================== UPDATE USED CONTENT TRACKER ====================
# CRITICAL: Save list of ALL content images used (including this run)
# This file is used in future runs to avoid duplicates

all_used_content = list(previously_used_content) + selected_content_files

with open(USED_CONTENT_TRACKER, 'w') as f:
    for content_path in all_used_content:
        f.write(content_path + '\n')

print(f"✓ Updated content tracker: {len(all_used_content)} total unique content images used")
print(f"✓ Tracker saved to: {USED_CONTENT_TRACKER}")

# ==================== SUMMARY ====================

final_df = pd.read_csv(METADATA_PATH)

print("\n" + "="*80)
print("DATASET GENERATION COMPLETE")
print("="*80)
print(f"Dataset location: {DATASET_ROOT}")
print(f"\nStatistics:")
print(f"  - Unique content images: {len(set(final_df['content_path']))}")
print(f"  - Total style images: {len(final_df)}")
print(f"  - Total synthetic images: {len(final_df)}")
print(f"  - Weight distribution:")
for w in STYLE_WEIGHTS:
    count = len(final_df[final_df['weight'] == w])
    print(f"      {w}: {count} samples ({count/len(final_df)*100:.1f}%)")
print(f"\nMetadata CSV: {METADATA_PATH}")
print(f"Used content tracker: {USED_CONTENT_TRACKER}")
print("\n" + "="*80)
print("TO ADD MORE SAMPLES LATER:")
print("1. Set RESUME_MODE = True")
print("2. Adjust TARGET_SAMPLES if needed")
print("3. Run this script again - it will automatically skip used content images!")
print("="*80)
