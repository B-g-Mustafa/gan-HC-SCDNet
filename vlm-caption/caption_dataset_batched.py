"""
Optimized Batch Captioning for Style Transfer Dataset (Qwen2-VL)
- Reads batch CSVs from dataset generation
- Captions synthetic images using Qwen2-VL-7B (TRUE batch processing)
- Generates enriched CSVs with all captions (1 output CSV per input CSV)
- Much faster than Llama while maintaining caption quality

RESUME CAPABILITIES:
1. Auto-resume from crashes: Progress saved after each mini-batch, resumes from last position
2. Skip completed batches: Won't reprocess batches that are 100% complete
3. Manual batch selection: Set START_FROM_BATCH to skip to specific batch number
4. Interrupt-safe: Press Ctrl+C to safely stop, resume later

USAGE:
- Normal run: Just execute the script
- Resume after crash: Re-run the script (auto-detects partial progress)
- Start from batch N: Set START_FROM_BATCH = N in configuration
- Check progress: Look in dataset/captioning_progress/ for active batches
"""

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import torch
import os
import re
import json
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
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LOCAL_FILES_ONLY = True  # Set True to skip download and use cached model only
USE_QUANTIZATION = False  # Use FP16 for best quality (A100 40GB has enough memory)

# Performance settings
MINI_BATCH_SIZE = 12  # Process 12 images simultaneously with FP16 (adjust based on GPU memory)
MAX_NEW_TOKENS = 200  # Qwen produces concise captions, 200 is enough
SAVE_FREQUENCY = 50  # Save progress every N mini-batches

# Resume settings
RESUME_MODE = True  # Automatically resume from partially processed CSVs
SKIP_EXISTING = True  # Skip batches that are 100% complete
START_FROM_BATCH = None  # Set to batch number (e.g., 1, 2, 3) to start from specific batch, None = auto-detect

# Progress tracking
PROGRESS_DIR = os.path.join(DATASET_ROOT, "captioning_progress")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(PROGRESS_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"Mini-batch size: {MINI_BATCH_SIZE} images (processed simultaneously)")
print(f"Quantization: {'4-bit' if USE_QUANTIZATION else 'FP16'}")
print(f"Input CSVs: {INPUT_CSV_DIR}")
print(f"Output CSVs: {OUTPUT_CSV_DIR}")
print(f"Note: Each input CSV will produce one output CSV with same row count")

# ==================== LOAD MODEL ====================
print("\n" + "="*80)
print("Loading model...")

# Set cache directory and offline mode
CACHE_DIR = "/home/msai/birul001/.cache/huggingface"  # Use existing HF cache
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variables for Hugging Face
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

# Try loading with retries for network issues
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Loading processor (attempt {attempt+1}/{max_retries})...")
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=USE_LOCAL_FILES_ONLY,
        )
        
        print(f"Loading model (attempt {attempt+1}/{max_retries})...")
        
        # Configure quantization for speed
        if USE_QUANTIZATION:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"
            )
            print("✓ 4-bit quantization enabled")
        else:
            quantization_config = None
        
        # Load model with optimizations
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if not USE_QUANTIZATION else None,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=USE_LOCAL_FILES_ONLY,
        )
        model.eval()  # Set to eval mode for faster inference
        
        print("✓ Model loaded successfully")
        print(f"  Device: {model.device if hasattr(model, 'device') else 'auto-mapped'}")
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
    Qwen2-VL returns JSON format which we parse directly.
    """
    try:
        # Try to parse as JSON first (Qwen2-VL should output clean JSON)
        # Remove any markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Parse JSON
        data = json.loads(text)
        
        return {
            "content_caption": data.get("content_caption", "").strip(),
            "style_caption": data.get("style_caption", "").strip(),
            "style_name": data.get("style_name", "").strip(),
            "final_caption": data.get("final_caption", "").strip()
        }
        
    except json.JSONDecodeError:
        # Fallback to regex parsing if JSON fails
        pattern = re.compile(
            r'"content_caption"\s*:\s*"([^"]*)".*?"style_caption"\s*:\s*"([^"]*)".*?"style_name"\s*:\s*"([^"]*)".*?"final_caption"\s*:\s*"([^"]*)"',
            re.DOTALL
        )
        
        match = pattern.search(text)
        if match:
            return {
                "content_caption": match.group(1).strip(),
                "style_caption": match.group(2).strip(),
                "style_name": match.group(3).strip(),
                "final_caption": match.group(4).strip()
            }
        
        # Return empty if all parsing fails
        return {
            "content_caption": "",
            "style_caption": "",
            "style_name": "",
            "final_caption": ""
        }
        
    except Exception as e:
        print(f"  ✗ Error parsing text: {str(e)[:100]}")
        return {
            "content_caption": "",
            "style_caption": "",
            "style_name": "",
            "final_caption": ""
        }

# ==================== BATCH CAPTIONING ====================

def create_prompt_messages():
    """Create the prompt template for Qwen2-VL captioning"""
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                "Analyze this image and describe both its content and artistic style. "
                "Output a JSON object with these fields:\n\n"
                "{\n"
                '  "content_caption": "What objects, scenes, or actions are visible",\n'
                '  "style_caption": "Artistic style characteristics (brushwork, colors, texture, etc.)",\n'
                '  "style_name": "Style movement or artist name (e.g., Impressionism, Van Gogh, Anime, Cyberpunk) or Unknown",\n'
                '  "final_caption": "Combined natural description of content and style"\n'
                "}\n\n"
                "Rules:\n"
                "- Be factual and concise (1-2 sentences per field)\n"
                "- Output valid JSON only, no markdown, no extra text\n"
                "- Focus on visible elements and recognizable style traits"
            )}
        ]}
    ]

def process_image_batch(image_paths, base_path):
    """
    Process a batch of images simultaneously using Qwen2-VL.
    This is TRUE batch processing - multiple images at once.
    """
    
    # Load all images in the batch
    images = []
    valid_indices = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            full_path = os.path.join(base_path, img_path)
            image = Image.open(full_path).convert("RGB")
            images.append(image)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  ✗ Error loading {os.path.basename(img_path)}: {e}")
    
    if not images:
        return [None] * len(image_paths)
    
    # Create prompt messages
    messages = create_prompt_messages()
    
    try:
        # Apply chat template
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Batch processing - process all images at once
        inputs = processor(
            images=images,
            text=[input_text] * len(images),
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Generate captions for entire batch
        with torch.amp.autocast('cuda', dtype=torch.float16):
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    num_beams=1,  # Greedy decoding for speed
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None,
                    eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else None,
                )
        
        # Decode all outputs
        batch_captions = []
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i][inputs['input_ids'].shape[1]:]
            output_text = processor.decode(generated_ids, skip_special_tokens=True)
            
            # Print first few outputs for debugging/regex development
            if i < 3:  # Print first 3 outputs
                print(f"\n{'='*80}")
                print(f"Sample Output {i+1}:")
                print(f"{'='*80}")
                print(output_text)
                print(f"{'='*80}\n")
            
            captions = extract_captions_from_text(output_text)
            batch_captions.append(captions)
        
        # Map back to original indices
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
        print(f"  ✗ Batch processing error: {str(e)[:200]}")
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
    
    # Process in mini-batches
    batch_count = 0
    with tqdm(total=remaining, desc="  Captioning", unit="img", initial=0) as pbar:
        for i in range(start_idx, len(synthetic_paths), MINI_BATCH_SIZE):
            batch_paths = synthetic_paths[i:i+MINI_BATCH_SIZE]
            
            # Process mini-batch (TRUE batch processing - all images simultaneously)
            batch_results = process_image_batch(batch_paths, base_path)
            
            # Update dataframe
            for j, result in enumerate(batch_results):
                row_idx = i + j
                if row_idx < len(synthetic_paths):  # Safety check
                    if result is not None:
                        df.at[row_idx, 'content_caption'] = result['content_caption']
                        df.at[row_idx, 'style_caption'] = result['style_caption']
                        df.at[row_idx, 'style_name'] = result['style_name']
                        df.at[row_idx, 'final_caption'] = result['final_caption']
            
            batch_count += 1
            
            # Save progress periodically (not every mini-batch to reduce I/O)
            if batch_count % SAVE_FREQUENCY == 0 or i + MINI_BATCH_SIZE >= len(synthetic_paths):
                df.to_csv(output_path, index=False)
                
                # Update progress tracker
                with open(progress_file, 'w') as f:
                    f.write(str(min(i + MINI_BATCH_SIZE, len(synthetic_paths))))
            
            pbar.update(len(batch_paths))
            
            # Clear CUDA cache periodically
            if batch_count % 10 == 0 and torch.cuda.is_available():
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
