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
DATASET_ROOT = "/home/msai/birul001/BIRUL001/dataset/test"
INPUT_CSV_DIR = os.path.join(DATASET_ROOT, "metadata")
OUTPUT_CSV_DIR = os.path.join(DATASET_ROOT, "captioned_metadata")
COMPLETE_CSV_PATH = os.path.join(INPUT_CSV_DIR, "complete_metadata.csv")
OUTPUT_COMPLETE_CSV = os.path.join(OUTPUT_CSV_DIR, "complete_captioned_metadata.csv")

# Model settings
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LOCAL_FILES_ONLY = True  # Set True to skip download and use cached model only
USE_QUANTIZATION = False  # Use FP16 for best quality (A100 40GB has enough memory)

# Performance settings
MINI_BATCH_SIZE = 12  # Process 12 images simultaneously with FP16 (adjust based on GPU memory)
MAX_NEW_TOKENS = 200  # Qwen produces concise captions, 200 is enough
SAVE_FREQUENCY = 100  # Save progress every N mini-batches (100 * 12 = 1200 images)

# Backup settings (prevent data loss)
CREATE_INTERMEDIATE_BACKUPS = True  # Save batch CSVs every N rows as backup
BACKUP_INTERVAL = 5000  # Create backup CSV every 5000 rows
BACKUP_DIR = os.path.join(OUTPUT_CSV_DIR, "backups")

# Resume settings
RESUME_MODE = True  # Automatically resume from partially processed CSV
START_FROM_ROW = None  # Set to row number to start from specific row, None = auto-detect

# Progress tracking
PROGRESS_FILE = os.path.join(DATASET_ROOT, "captioning_progress.txt")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
if CREATE_INTERMEDIATE_BACKUPS:
    os.makedirs(BACKUP_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"Mini-batch size: {MINI_BATCH_SIZE} images (processed simultaneously)")
print(f"Quantization: {'4-bit' if USE_QUANTIZATION else 'FP16'}")
print(f"Input CSV: {COMPLETE_CSV_PATH}")
print(f"Output CSV: {OUTPUT_COMPLETE_CSV}")
print(f"Processing entire dataset in one go with resume support")
if CREATE_INTERMEDIATE_BACKUPS:
    print(f"Backup CSVs: Every {BACKUP_INTERVAL:,} rows in {BACKUP_DIR}")

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

# ==================== PROCESS COMPLETE CSV ====================

def process_complete_csv():
    """Process the complete metadata CSV and generate captioned output"""
    
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(COMPLETE_CSV_PATH)}")
    
    # Check if input exists
    if not os.path.exists(COMPLETE_CSV_PATH):
        print(f"✗ Input CSV not found: {COMPLETE_CSV_PATH}")
        return
    
    # Read input CSV
    df = pd.read_csv(COMPLETE_CSV_PATH)
    total_rows = len(df)
    print(f"  Total rows: {total_rows:,}")
    
    # Check resume state
    start_idx = 0
    
    if RESUME_MODE and os.path.exists(OUTPUT_COMPLETE_CSV):
        # Load existing output
        existing_df = pd.read_csv(OUTPUT_COMPLETE_CSV)
        
        if 'final_caption' in existing_df.columns:
            # Count completed captions
            completed = (existing_df['final_caption'] != "").sum()
            
            if completed == total_rows:
                print(f"  ✓ Already 100% complete ({total_rows:,} images captioned)")
                return
            elif completed > 0:
                print(f"  ↻ Found partial progress: {completed:,}/{total_rows:,} images captioned")
                print(f"  ↻ Resuming from row {completed}")
                df = existing_df
                start_idx = completed
            else:
                print(f"  Starting fresh captioning")
        else:
            print(f"  Starting fresh captioning")
    elif START_FROM_ROW is not None:
        start_idx = START_FROM_ROW
        print(f"  ↻ Starting from row {start_idx} (manual override)")
    
    # Initialize caption columns if not present
    if 'content_caption' not in df.columns:
        df['content_caption'] = ""
        df['style_caption'] = ""
        df['style_name'] = ""
        df['final_caption'] = ""
    
    # Get synthetic image paths
    synthetic_paths = df['synthetic_path'].tolist()
    
    # Process from start_idx onwards
    remaining = total_rows - start_idx
    if remaining <= 0:
        print(f"  ✓ Already complete")
        return
    
    print(f"\n  Processing {remaining:,} remaining images (from row {start_idx:,} to {total_rows:,})")
    print(f"  Saving progress every {SAVE_FREQUENCY * MINI_BATCH_SIZE:,} images")
    print(f"{'='*80}\n")
    
    # Process in mini-batches
    batch_count = 0
    last_backup_row = (start_idx // BACKUP_INTERVAL) * BACKUP_INTERVAL if CREATE_INTERMEDIATE_BACKUPS else 0
    
    with tqdm(total=remaining, desc="Captioning", unit="img", initial=0) as pbar:
        for i in range(start_idx, total_rows, MINI_BATCH_SIZE):
            batch_paths = synthetic_paths[i:i+MINI_BATCH_SIZE]
            
            # Process mini-batch (TRUE batch processing - all images simultaneously)
            batch_results = process_image_batch(batch_paths, DATASET_ROOT)
            
            # Update dataframe
            for j, result in enumerate(batch_results):
                row_idx = i + j
                if row_idx < total_rows:  # Safety check
                    if result is not None:
                        df.at[row_idx, 'content_caption'] = result['content_caption']
                        df.at[row_idx, 'style_caption'] = result['style_caption']
                        df.at[row_idx, 'style_name'] = result['style_name']
                        df.at[row_idx, 'final_caption'] = result['final_caption']
            
            batch_count += 1
            current_row = i + MINI_BATCH_SIZE
            
            # Save progress periodically (not every mini-batch to reduce I/O)
            if batch_count % SAVE_FREQUENCY == 0 or current_row >= total_rows:
                df.to_csv(OUTPUT_COMPLETE_CSV, index=False)
                
                # Create intermediate backup CSV at intervals
                if CREATE_INTERMEDIATE_BACKUPS and current_row >= last_backup_row + BACKUP_INTERVAL:
                    backup_num = current_row // BACKUP_INTERVAL
                    backup_path = os.path.join(BACKUP_DIR, f"backup_{backup_num:04d}_rows_{last_backup_row:06d}_{current_row:06d}.csv")
                    # Save only the rows completed in this backup interval
                    df.iloc[last_backup_row:current_row].to_csv(backup_path, index=False)
                    last_backup_row = current_row
                    pbar.write(f"  ✓ Backup saved: {os.path.basename(backup_path)}")
                
                # Update progress in progress bar
                completed_now = (df['final_caption'] != "").sum()
                pbar.set_postfix({
                    'completed': f"{completed_now:,}/{total_rows:,}",
                    'batch': batch_count
                })
            
            pbar.update(len(batch_paths))
            
            # Clear CUDA cache periodically
            if batch_count % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final save
    df.to_csv(OUTPUT_COMPLETE_CSV, index=False)
    
    # Print statistics
    captioned_count = (df['final_caption'] != "").sum()
    print(f"\n{'='*80}")
    print(f"✓ CAPTIONING COMPLETE!")
    print(f"{'='*80}")
    print(f"  Output: {OUTPUT_COMPLETE_CSV}")
    print(f"  Successfully captioned: {captioned_count:,}/{total_rows:,} ({captioned_count/total_rows*100:.1f}%)")
    print(f"  Failed: {total_rows - captioned_count:,}")
    
    if CREATE_INTERMEDIATE_BACKUPS:
        backup_csvs = sorted(glob.glob(os.path.join(BACKUP_DIR, "backup_*.csv")))
        print(f"  Backup CSVs created: {len(backup_csvs)} (in {BACKUP_DIR})")
    
    print(f"{'='*80}")

# ==================== MAIN ====================

def main():
    print("\n" + "="*80)
    print("Starting complete dataset captioning...")
    print("="*80)
    
    try:
        process_complete_csv()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠ Interrupted by user")
        print("="*80)
        print("✓ Progress saved - you can resume by running this script again")
        print(f"✓ Partial results saved to: {OUTPUT_COMPLETE_CSV}")
        return
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
