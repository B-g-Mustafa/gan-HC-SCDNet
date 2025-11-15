"""
Optimized Batch Captioning for Style Transfer Dataset
- Reads batch CSVs from dataset generation
- Captions synthetic images in true batches (not one-by-one)
- Generates enriched CSVs with all captions
- Much faster than sequential processing
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

# Performance settings
BATCH_SIZE = 8  # Process 8 images simultaneously (adjust based on GPU memory)
MAX_NEW_TOKENS = 300
USE_FLASH_ATTENTION = True  # Enable if available (faster inference)

# Resume settings
RESUME_MODE = False  # Set to True to skip already processed batch CSVs
SKIP_EXISTING = True  # Skip batches that already have output CSVs

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Input CSVs: {INPUT_CSV_DIR}")
print(f"Output CSVs: {OUTPUT_CSV_DIR}")

# ==================== LOAD MODEL ====================
print("\n" + "="*80)
print("Loading model...")

processor = MllamaProcessor.from_pretrained(MODEL_ID)

# Load model with optimizations
model_kwargs = {
    "torch_dtype": torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    "device_map": "auto",
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

print("✓ Model loaded")

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
    
    if SKIP_EXISTING and os.path.exists(output_path):
        print(f"  ✓ Already processed, skipping")
        return
    
    # Initialize caption columns
    df['content_caption'] = ""
    df['style_caption'] = ""
    df['style_name'] = ""
    df['final_caption'] = ""
    
    # Get synthetic image paths
    synthetic_paths = df['synthetic_path'].tolist()
    
    # Process in batches
    total_batches = (len(synthetic_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with tqdm(total=len(synthetic_paths), desc="  Captioning", unit="img") as pbar:
        for i in range(0, len(synthetic_paths), BATCH_SIZE):
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
            
            pbar.update(len(batch_paths))
            
            # Clear CUDA cache periodically
            if i % (BATCH_SIZE * 5) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save enriched CSV
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_filename}")
    
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
    
    # Process each batch CSV
    for csv_path in batch_csvs:
        try:
            process_batch_csv(csv_path, DATASET_ROOT)
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
