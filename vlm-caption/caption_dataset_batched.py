"""
Optimized Batch Captioning for Style Transfer Dataset
- Reads batch CSVs from dataset generation
- Captions synthetic images in true batches (not one-by-one)
- Generates enriched CSVs with all captions
- Much faster than sequential processing

RESUME CAPABILITIES:
1. Auto-resume from crashes: Progress saved after each batch, resumes from last position
2. Skip completed batches: Won't reprocess batches that are 100% complete
3. Manual batch selection: Set START_FROM_BATCH to skip to specific batch number
4. Interrupt-safe: Press Ctrl+C to safely stop, resume later

USAGE:
- Normal run: Just execute the script
- Resume after crash: Re-run the script (auto-detects partial progress)
- Start from batch N: Set START_FROM_BATCH = N in configuration
- Check progress: Look in dataset/captioning_progress/ for active batches
"""

from transformers import MllamaProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os
import re
import pandas as pd
from tqdm import tqdm
import glob

# ==================== CONFIGURATION ====================
print("="*80)
print("Optimized Batch Captioning for Style Transfer Dataset")
print("="*80)

# Paths
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/"
INPUT_CSV_DIR = os.path.join(DATASET_ROOT, "metadata")
OUTPUT_CSV_DIR = os.path.join(DATASET_ROOT, "captioned_metadata")
INPUT_CSV_PATTERN = "metadata_batch_*.csv"  # Process batch CSVs

# Model settings
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LOCAL_FILES_ONLY = False  # Set True to skip download and use cached model only

# Performance settings
BATCH_SIZE = 8  # Process 8 images simultaneously (adjust based on GPU memory)
MAX_NEW_TOKENS = 300
USE_FLASH_ATTENTION = False  # Disabled (requires CUDA dev tools to install)

# Resume settings
RESUME_MODE = True  # Automatically resume from partially processed CSVs
SKIP_EXISTING = True  # Skip batches that are 100% complete
START_FROM_BATCH = None  # Set to batch number (e.g., 1, 2, 3) to start from specific batch, None = auto-detect

# Progress tracking
PROGRESS_DIR = os.path.join(DATASET_ROOT, "captioning_progress")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Input CSVs: {INPUT_CSV_DIR}")
print(f"Output CSVs: {OUTPUT_CSV_DIR}")

# ==================== LOAD MODEL ====================
print("\n" + "="*80)
print("Loading model...")

# Set cache directory and offline mode
CACHE_DIR = "/home/msai/birul001/hf_cache"  # Persistent cache location
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variables for Hugging Face
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

# Try loading with retries for network issues
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Loading processor (attempt {attempt+1}/{max_retries})...")
        processor = MllamaProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            resume_download=True,  # Resume interrupted downloads
            local_files_only=USE_LOCAL_FILES_ONLY,  # Skip download if cached
        )
        
        print(f"Loading model (attempt {attempt+1}/{max_retries})...")
        
        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            "device_map": "auto",
            "cache_dir": CACHE_DIR,
            "resume_download": True,
            "local_files_only": USE_LOCAL_FILES_ONLY,  # Skip download if cached
        }
        
        # Enable Flash Attention 2 if available (2-3x faster)
        if USE_FLASH_ATTENTION and DEVICE == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("✓ Flash Attention 2 enabled")
            except:
                print("⚠ Flash Attention 2 not available, using default attention")
        
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, **model_kwargs)
        model.eval()  # Set to eval mode for faster inference
        
        print("✓ Model loaded successfully")
        break
        
    except Exception as e:
        print(f"✗ Attempt {attempt+1} failed: {str(e)[:200]}")
        if attempt < max_retries - 1:
            print(f"  Retrying in 30 seconds...")
            import time
            time.sleep(30)
        else:
            print("\n" + "="*80)
            print("ERROR: Failed to load model after all retries")
            print("="*80)
            print("\nPossible solutions:")
            print("1. Check internet connection")
            print("2. Download model manually first:")
            print(f"   huggingface-cli download {MODEL_ID} --cache-dir {CACHE_DIR}")
            print("3. Use a different network or VPN")
            print("4. Set local_files_only=True if model is already cached")
            raise

# ==================== CAPTION EXTRACTION ====================

def extract_captions_from_text(text: str):
    """
    Extract captions from model output.
    Updated to handle the structured format we're requesting.
    """
    
    # Pattern to match the structured output
    pattern = re.compile(
        r"\*\*Content Caption:\*\*\s*([^\n\r]*)\n+\*\*Style Caption:\*\*\s*([^\n\r]*)\n+\*\*Style Name:\*\*\s*([^\n\r]*)\n+\*\*Final Caption:\*\*\s*((?:(?!<\|eot_id\|>).)*)",
        re.DOTALL
    )
    
    try:
        match = pattern.search(text)
        if not match:
            # Return empty strings if parsing fails
            return {
                "content_caption": "",
                "style_caption": "",
                "style_name": "",
                "final_caption": ""
            }
        
        content_caption, style_caption, style_name, final_caption = match.groups()
        
        return {
            "content_caption": content_caption.strip(),
            "style_caption": style_caption.strip(),
            "style_name": style_name.strip(),
            "final_caption": final_caption.strip()
        }
        
    except Exception as e:
        print(f"Error parsing text: {str(e)}")
        return {
            "content_caption": "",
            "style_caption": "",
            "style_name": "",
            "final_caption": ""
        }

# ==================== BATCH CAPTIONING ====================

def create_prompt_messages():
    """Create the prompt template for captioning"""
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                "You are an expert vision-language model trained to describe both the *content* and *style* of images "
                "for synthetic dataset annotation. Analyze the image carefully and produce captions that cover three aspects:\n"
                "1. The **content** — what objects, scenes, or actions are visible.\n"
                "2. The **style** — describe the artistic or visual characteristics such as brushwork, color palette, texture, "
                "or digital art traits. If recognizable, identify the **style name or movement** (e.g., 'Van Gogh', 'Impressionism', "
                "'Cyberpunk', 'Watercolor', '3D render', 'Anime', etc.).\n"
                "3. The **final caption** — combine both content and style naturally, describing how the two interact visually.\n\n"
                "Output format:\n"
                "**Content Caption:** <description>\n"
                "**Style Caption:** <style description>\n"
                "**Style Name:** <style name or Unknown>\n"
                "**Final Caption:** <combined description>\n\n"
                "Rules:\n"
                "- Be factual and concise (1–2 sentences per field).\n"
                "- Do not hallucinate or make assumptions beyond visible traits.\n"
                "- Avoid extra commentary or explanations."
            )}
        ]}
    ]

def process_image_batch(image_paths, base_path):
    """
    Process a batch of images simultaneously.
    This is the key optimization - process multiple images at once.
    """
    
    # Load all images in the batch
    images = []
    valid_indices = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            # Construct full path
            full_path = os.path.join(base_path, img_path)
            
            # Load image - NO RESIZING (model handles it internally)
            image = Image.open(full_path).convert("RGB")
            images.append(image)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"✗ Error loading {img_path}: {e}")
            # Append None for failed images
            images.append(None)
    
    # Filter out None images
    valid_images = [img for img in images if img is not None]
    
    if not valid_images:
        return [None] * len(image_paths)
    
    # Create prompts for batch
    messages = create_prompt_messages()
    
    # Process batch
    try:
        # Apply chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Batch processing - process all valid images at once
        inputs = processor(
            images=valid_images,
            text=[input_text] * len(valid_images),  # Same prompt for all
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Generate captions for entire batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # Deterministic for consistency
                temperature=None,  # Disable sampling
                top_p=None,
                use_cache=True,  # Enable KV cache for faster generation
            )
        
        # Decode all outputs
        batch_captions = []
        for output in outputs:
            output_text = processor.decode(output, skip_special_tokens=True)
            captions = extract_captions_from_text(output_text)
            batch_captions.append(captions)
        
        # Map back to original indices (accounting for failed loads)
        results = []
        valid_idx = 0
        for idx in range(len(image_paths)):
            if idx in valid_indices:
                results.append(batch_captions[valid_idx])
                valid_idx += 1
            else:
                results.append(None)
        
        return results
        
    except Exception as e:
        print(f"✗ Batch processing error: {e}")
        return [None] * len(image_paths)

# ==================== PROCESS BATCH CSVs ====================

def process_batch_csv(csv_path, base_path):
    """Process a single batch CSV and generate enriched output"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(csv_path)}")
    
    # Read input CSV
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}")
    
    # Check if output already exists
    output_filename = os.path.basename(csv_path).replace("metadata_batch_", "captioned_batch_")
    output_path = os.path.join(OUTPUT_CSV_DIR, output_filename)
    
    # Progress tracking file for this batch
    batch_name = os.path.splitext(output_filename)[0]
    progress_file = os.path.join(PROGRESS_DIR, f"{batch_name}_progress.txt")
    
    # Check resume state
    start_idx = 0
    if RESUME_MODE and os.path.exists(progress_file):
        # Read last completed index
        with open(progress_file, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"  ↻ Resuming from row {start_idx}")
        
        # Load partially completed output if exists
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"  ↻ Loaded partial results")
    elif SKIP_EXISTING and os.path.exists(output_path):
        # Check if 100% complete
        existing_df = pd.read_csv(output_path)
        if 'final_caption' in existing_df.columns:
            completed = (existing_df['final_caption'] != "").sum()
            if completed == len(existing_df):
                print(f"  ✓ Already 100% complete, skipping")
                return
            else:
                print(f"  ↻ Partially complete ({completed}/{len(existing_df)}), resuming")
                df = existing_df
                start_idx = completed
    
    # Initialize caption columns if not present
    if 'content_caption' not in df.columns:
        df['content_caption'] = ""
        df['style_caption'] = ""
        df['style_name'] = ""
        df['final_caption'] = ""
    
    # Get synthetic image paths
    synthetic_paths = df['synthetic_path'].tolist()
    
    # Process from start_idx onwards
    remaining = len(synthetic_paths) - start_idx
    if remaining <= 0:
        print(f"  ✓ Already complete")
        return
    
    print(f"  Processing {remaining} remaining images (from {start_idx}/{len(synthetic_paths)})")
    
    # Process in batches
    with tqdm(total=remaining, desc="  Captioning", unit="img", initial=0) as pbar:
        for i in range(start_idx, len(synthetic_paths), BATCH_SIZE):
            batch_paths = synthetic_paths[i:i+BATCH_SIZE]
            
            # Process batch
            batch_results = process_image_batch(batch_paths, base_path)
            
            # Update dataframe
            for j, result in enumerate(batch_results):
                row_idx = i + j
                if result is not None:
                    df.at[row_idx, 'content_caption'] = result['content_caption']
                    df.at[row_idx, 'style_caption'] = result['style_caption']
                    df.at[row_idx, 'style_name'] = result['style_name']
                    df.at[row_idx, 'final_caption'] = result['final_caption']
            
            # Save progress immediately after each batch
            df.to_csv(output_path, index=False)
            
            # Update progress tracker
            with open(progress_file, 'w') as f:
                f.write(str(min(i + BATCH_SIZE, len(synthetic_paths))))
            
            pbar.update(len(batch_paths))
            
            # Clear CUDA cache periodically
            if i % (BATCH_SIZE * 5) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Mark as complete
    print(f"  ✓ Completed: {output_filename}")
    
    # Clean up progress file once 100% complete
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    # Print statistics
    captioned_count = (df['final_caption'] != "").sum()
    print(f"  ✓ Successfully captioned: {captioned_count}/{len(df)} ({captioned_count/len(df)*100:.1f}%)")

# ==================== MAIN ====================

def main():
    print("\n" + "="*80)
    print("Starting batch processing...")
    print("="*80)
    
    # Find all batch CSVs
    csv_pattern = os.path.join(INPUT_CSV_DIR, INPUT_CSV_PATTERN)
    batch_csvs = sorted(glob.glob(csv_pattern))
    
    if not batch_csvs:
        print(f"✗ No batch CSVs found matching pattern: {INPUT_CSV_PATTERN}")
        return
    
    print(f"✓ Found {len(batch_csvs)} batch CSV files")
    
    # Apply START_FROM_BATCH filter if specified
    if START_FROM_BATCH is not None:
        print(f"↻ Starting from batch {START_FROM_BATCH}")
        batch_csvs = [csv for csv in batch_csvs if 
                     int(re.search(r'batch_(\d+)', csv).group(1)) >= START_FROM_BATCH]
        print(f"  → Processing {len(batch_csvs)} batches")
    
    # Show resume status
    if RESUME_MODE:
        progress_files = glob.glob(os.path.join(PROGRESS_DIR, "*_progress.txt"))
        if progress_files:
            print(f"↻ Resume mode: Found {len(progress_files)} partially completed batches")
    
    # Process each batch CSV
    for idx, csv_path in enumerate(batch_csvs, 1):
        try:
            print(f"\n[Batch {idx}/{len(batch_csvs)}]")
            process_batch_csv(csv_path, DATASET_ROOT)
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            print("✓ Progress saved - you can resume by running this script again")
            return
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(csv_path)}: {e}")
            continue
    
    # Combine all captioned batches into one complete CSV
    print("\n" + "="*80)
    print("Combining all batches into complete CSV...")
    
    captioned_csvs = sorted(glob.glob(os.path.join(OUTPUT_CSV_DIR, "captioned_batch_*.csv")))
    
    if captioned_csvs:
        all_dfs = [pd.read_csv(csv) for csv in captioned_csvs]
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        complete_output = os.path.join(OUTPUT_CSV_DIR, "complete_captioned_metadata.csv")
        combined_df.to_csv(complete_output, index=False)
        
        print(f"✓ Combined {len(captioned_csvs)} batch CSVs")
        print(f"✓ Total rows: {len(combined_df)}")
        print(f"✓ Saved to: {complete_output}")
        
        # Statistics
        captioned = (combined_df['final_caption'] != "").sum()
        print(f"\nFinal Statistics:")
        print(f"  Total samples: {len(combined_df)}")
        print(f"  Successfully captioned: {captioned} ({captioned/len(combined_df)*100:.1f}%)")
        print(f"  Failed: {len(combined_df) - captioned}")
    
    print("\n" + "="*80)
    print("CAPTIONING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
