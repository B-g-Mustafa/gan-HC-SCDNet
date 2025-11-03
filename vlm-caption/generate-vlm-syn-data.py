import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List
import random


# --- 1. Model Setup (no VAE needed) ---

def setup_models():
    """Load the Vision-Language Model (BLIP) safely and handle torch version issues."""
    print("🔄 Loading Vision-Language Model (BLIP)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("📟 Device:", device)

    # Check torch version
    required_version = (2, 6, 0)
    current_version = tuple(map(int, torch.__version__.split("+")[0].split(".")))
    if current_version < required_version:
        print(f"⚠️ Detected torch {torch.__version__}, which is below 2.6.0.")
        print("➡️  Either upgrade PyTorch to >=2.6.0 or ensure safetensors are used.")
        print("   Run: pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121")
        raise RuntimeError(
            "PyTorch version too old for safe model loading. "
            "Upgrade to >=2.6.0 or use safetensors format."
        )

    # Load using transformers (uses safetensors if available)
    vlm_processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", trust_remote_code=False
    )
    vlm_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    vlm_model.eval()
    print("✅ BLIP model loaded successfully.")
    return vlm_processor, vlm_model, device



# --- 2. Image utilities ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def list_images(input_dir: str, recursive: bool = True) -> List[str]:
    """Collect image file paths from a directory."""
    img_paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTS:
                    img_paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            p = os.path.join(input_dir, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                img_paths.append(p)
    return sorted(img_paths)


# --- 3. Captioning for a single image ---
def caption_image(vlm_processor, vlm_model, device: str, image_path: str, min_words: int = 5, max_words: int = 50) -> str:
    """Runs BLIP captioning on a single image path and returns the caption.
    
    Args:
        vlm_processor: The BLIP processor
        vlm_model: The BLIP model  
        device: Device to run inference on
        image_path: Path to the image
        min_words: Minimum number of words in caption (default: 5)
        max_words: Maximum number of words in caption (default: 50)
        
    Returns:
        Generated caption string
    """
    with Image.open(image_path) as im:
        image = im.convert("RGB")

    inputs = vlm_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Randomly sample target length between min_words and max_words
    # Convert words to approximate token length (roughly 1.3 tokens per word)
    target_words = random.randint(min_words, max_words)
    max_length = int(target_words * 1.3) + 10  # Add buffer for special tokens

    with torch.no_grad():
        generated_ids = vlm_model.generate(
            pixel_values=pixel_values, 
            max_length=max_length,
            do_sample=True,  # Enable sampling for more diverse captions
            temperature=0.7,  # Add some randomness
            top_p=0.9        # Nucleus sampling for better quality
        )

    # Use processor's batch_decode (preferred) and take the first item
    captions = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)
    caption = captions[0] if captions else ""
    
    # Post-process to ensure word count is within desired range
    words = caption.split()
    if len(words) > max_words:
        caption = ' '.join(words[:max_words])
    elif len(words) < min_words and len(words) > 0:
        # If caption is too short, try to extend it by regenerating with higher max_length
        # For now, just keep the shorter caption to maintain diversity
        pass
    
    return caption


# --- 4. Main Dataset Captioning Loop ---
def create_dataset(input_dir: str, output_dir: str = "vlm_captions", recursive: bool = True, 
                  min_words: int = 5, max_words: int = 50):
    """Caption all images found in input_dir and write a metadata CSV.

    - input_dir: directory containing already-decoded images (from VAE or otherwise)
    - output_dir: where to store metadata (CSV); images are read in-place, not copied
    - recursive: whether to search subdirectories
    - min_words: minimum number of words in generated captions (default: 5)
    - max_words: maximum number of words in generated captions (default: 50)
    """
    image_paths = list_images(input_dir, recursive=recursive)
    if len(image_paths) == 0:
        print(f"No images found in {input_dir} (recursive={recursive}). Nothing to do.")
        return
    dataset_records = []

    print(f"Captioning {len(image_paths)} images from: {input_dir}")
    print(f"📝 Using variable caption length: {min_words}-{max_words} words")
    
    word_counts = []  # Track word counts for statistics
    
    for img_path in tqdm(image_paths):
        try:
            caption = caption_image(vlm_processor, vlm_model, device, img_path, 
                                  min_words=min_words, max_words=max_words)
            word_count = len(caption.split())
            word_counts.append(word_count)
            dataset_records.append({
                "image_path": img_path, 
                "caption": caption,
                "word_count": word_count
            })
        except Exception as e:
            # Keep going, but record the error for visibility
            dataset_records.append({
                "image_path": img_path, 
                "caption": "", 
                "word_count": 0,
                "error": str(e)
            })

    # --- 5. Save Dataset Metadata ---
    df = pd.DataFrame(dataset_records)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    
    # Print statistics
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        print(f"📊 Caption statistics:")
        print(f"   Average words: {avg_words:.1f}")
        print(f"   Range: {min(word_counts)}-{max(word_counts)} words")
        print(f"   Target range: {min_words}-{max_words} words")
    
    print(f"Captioning complete. Metadata saved to {metadata_path}")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    vlm_processor, vlm_model, device = setup_models()


# def parse_args():
#     parser = argparse.ArgumentParser(description="Generate captions for images in a directory using BLIP.")
#     parser.add_argument("--input_dir", type=str, required=True, help="Directory with decoded images to caption.")
#     parser.add_argument("--output_dir", type=str, default="vlm_captions", help="Output directory for metadata CSV.")
#     parser.add_argument("--no_recursive", action="store_true", help="Disable recursive search in subdirectories.")
#     parser.add_argument("--min_words", type=int, default=5, help="Minimum number of words in generated captions.")
#     parser.add_argument("--max_words", type=int, default=50, help="Maximum number of words in generated captions.")
#     return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()
    create_dataset(
        input_dir="/home/msai/birul001/BIRUL001/data/synthetic_dataset",
        output_dir="/home/msai/birul001/BIRUL001/data/synthetic_caption_dataset/test100",
        recursive=False,
        min_words=5,      # Minimum 5 words for meaningful descriptions
        max_words=50,     # Maximum 50 words to balance detail and quality
    )
