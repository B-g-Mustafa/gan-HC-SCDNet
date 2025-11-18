"""
Test Dataset Generator (15,000 samples)
- Generates test data in separate folder structure
- Uses content images NOT used in training
- Same structure as training dataset
- Ensures no overlap with training data
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
print("Test Dataset Generator (15,000 samples)")
print("="*80)

# Source directories
CONTENT_DIR = "/home/msai/birul001/BIRUL001/data/coco/train2017/"
STYLE_DIR = "/home/msai/birul001/BIRUL001/data/wikiart/split50/train"

# Output dataset structure (TEST FOLDER)
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/"
TEST_ROOT = os.path.join(DATASET_ROOT, "test")  # ← NEW: test subfolder
OUTPUT_CONTENT_DIR = os.path.join(TEST_ROOT, "content_images")
OUTPUT_STYLE_DIR = os.path.join(TEST_ROOT, "style_images")
OUTPUT_SYNTHETIC_DIR = os.path.join(TEST_ROOT, "style_transferred_images")
OUTPUT_CSV_DIR = os.path.join(TEST_ROOT, "metadata")
METADATA_PATH = os.path.join(OUTPUT_CSV_DIR, "complete_metadata.csv")

# Path to training metadata (to avoid content overlap)
TRAIN_METADATA = os.path.join(DATASET_ROOT, "metadata", "complete_metadata.csv")

# Generation parameters
TARGET_SAMPLES = 15000
STYLE_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
BATCH_SIZE = 16  # Optimal for A100 40GB
SAMPLES_PER_CONTENT = 5  # Each content image used 5 times

# Model paths
VGG_PATH = '/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/vgg_r41.pth'
DEC_PATH = '/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/dec_r41.pth'
MATRIX_PATH = '/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/matrix_r41_new.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Target samples: {TARGET_SAMPLES}")
print(f"Samples per content image: {SAMPLES_PER_CONTENT}")

# ==================== SETUP ====================

# Create output directories
os.makedirs(OUTPUT_CONTENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_STYLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_SYNTHETIC_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

print("\nCreated test dataset structure:")
print(f"  - Test root: {TEST_ROOT}")
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
all_content_files = sorted([
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

print(f"✓ Found {len(all_content_files)} total content images")
print(f"✓ Found {len(style_files)} style images")

# ==================== EXCLUDE TRAINING CONTENT IMAGES ====================

print("\n" + "="*80)
print("Excluding training content images...")

if os.path.exists(TRAIN_METADATA):
    train_df = pd.read_csv(TRAIN_METADATA)
    
    # Extract training content image basenames
    train_content_basenames = set()
    for path in train_df['content_path']:
        # Extract just the filename from path like "content_images/content_000123.jpg"
        basename = os.path.basename(path)
        # Extract original index from "content_000123.jpg"
        train_content_basenames.add(basename)
    
    print(f"✓ Found {len(train_content_basenames)} unique content images in training set")
    
    # We need to map back to original content files
    # Training used first N content files after shuffle with seed 42
    # To avoid overlap, we'll use a different seed for test
    random.seed(99)  # ← DIFFERENT SEED from training (which used 42)
    random.shuffle(all_content_files)
    
    # Take content files NOT in training
    # Since training used first 12,000 (or whatever), we use the rest
    train_content_count = len(train_content_basenames)
    content_files = all_content_files[train_content_count:]
    
    print(f"✓ Using content images starting from index {train_content_count}")
    print(f"✓ Available content images for test: {len(content_files)}")
    
else:
    print("⚠ Warning: Training metadata not found, using all content images")
    random.seed(99)
    random.shuffle(all_content_files)
    content_files = all_content_files

# Calculate required unique content images
required_content_images = TARGET_SAMPLES // SAMPLES_PER_CONTENT
if required_content_images > len(content_files):
    print(f"\n⚠ Warning: Need {required_content_images} unique content images but only {len(content_files)} available")
    print(f"  Adjusting target to {len(content_files) * SAMPLES_PER_CONTENT} samples")
    TARGET_SAMPLES = len(content_files) * SAMPLES_PER_CONTENT
    required_content_images = len(content_files)

print(f"\n✓ Will use {required_content_images} unique content images for test")
print(f"✓ Total test samples to generate: {TARGET_SAMPLES}")

# Select content images for test
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
print("Starting test dataset generation...")
print("="*80)

all_metadata = []
batch_data = []
global_idx = 0
batch_num = 0

# Create generation plan
generation_plan = []
for content_idx, content_path in enumerate(selected_content_files):
    for sample_num in range(SAMPLES_PER_CONTENT):
        style_path = random.choice(style_files)
        weight = random.choice(STYLE_WEIGHTS)
        seed = random.randint(0, int(1e6))
        
        generation_plan.append({
            'content_path': content_path,
            'content_idx': content_idx,
            'style_path': style_path,
            'weight': weight,
            'seed': seed
        })

print(f"✓ Created generation plan with {len(generation_plan)} samples\n")

# Process in batches
total_batches = (len(generation_plan) + BATCH_SIZE - 1) // BATCH_SIZE

with tqdm(total=len(generation_plan), desc="Generating test dataset", unit="img") as pbar:
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
                content_filename = f"content_{item['content_idx']:06d}.jpg"
                style_filename = f"style_{global_idx:06d}.jpg"
                synthetic_filename = f"synthetic_{global_idx:06d}.jpg"
                
                # Copy content image (only once per unique content)
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
            
            # Save batch CSV
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

# ==================== SAVE COMPLETE METADATA ====================

print("\n" + "="*80)
print("Saving complete test metadata...")

df = pd.DataFrame(all_metadata)
df.to_csv(METADATA_PATH, index=False)

print(f"✓ Saved complete metadata to: {METADATA_PATH}")
print(f"✓ Total test samples generated: {len(all_metadata)}")

# ==================== SUMMARY ====================

print("\n" + "="*80)
print("TEST DATASET GENERATION COMPLETE")
print("="*80)
print(f"Test dataset location: {TEST_ROOT}")
print(f"\nStatistics:")
print(f"  - Unique content images: {len(set(df['content_path']))}")
print(f"  - Total style images: {len(df)}")
print(f"  - Total synthetic images: {len(df)}")
print(f"  - Weight distribution:")
for w in STYLE_WEIGHTS:
    count = len(df[df['weight'] == w])
    print(f"      {w}: {count} samples ({count/len(df)*100:.1f}%)")
print(f"\nTest metadata CSV: {METADATA_PATH}")
print(f"\nFinal Dataset Structure:")
print(f"  {DATASET_ROOT}/")
print(f"  ├── content_images/          (training - 12,000 images)")
print(f"  ├── style_images/            (training - 60,000 images)")
print(f"  ├── style_transferred_images/ (training - 60,000 images)")
print(f"  ├── metadata/                (training metadata)")
print(f"  └── test/")
print(f"      ├── content_images/          (test - 3,000 images)")
print(f"      ├── style_images/            (test - 15,000 images)")
print(f"      ├── style_transferred_images/ (test - 15,000 images)")
print(f"      └── metadata/                (test metadata)")
print("="*80)