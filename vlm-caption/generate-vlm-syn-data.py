import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List


# --- 1. Model Setup (no VAE needed) ---
def setup_models(device: str | None = None):
    """Load the Vision-Language Model for captioning."""
    print("Loading Vision-Language Model (BLIP)...")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vlm_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    vlm_model.eval()
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
def caption_image(vlm_processor, vlm_model, device: str, image_path: str, max_length: int = 50) -> str:
    """Runs BLIP captioning on a single image path and returns the caption."""
    with Image.open(image_path) as im:
        image = im.convert("RGB")

    inputs = vlm_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(pixel_values=pixel_values, max_length=max_length)

    # Use processor's batch_decode (preferred) and take the first item
    captions = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)
    return captions[0] if captions else ""


# --- 4. Main Dataset Captioning Loop ---
def create_dataset(input_dir: str, output_dir: str = "vlm_captions", recursive: bool = True, max_length: int = 50):
    """Caption all images found in input_dir and write a metadata CSV.

    - input_dir: directory containing already-decoded images (from VAE or otherwise)
    - output_dir: where to store metadata (CSV); images are read in-place, not copied
    - recursive: whether to search subdirectories
    - max_length: max caption length for generation
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    vlm_processor, vlm_model, device = setup_models()

    image_paths = list_images(input_dir, recursive=recursive)
    if len(image_paths) == 0:
        print(f"No images found in {input_dir} (recursive={recursive}). Nothing to do.")
        return

    dataset_records = []

    print(f"Captioning {len(image_paths)} images from: {input_dir}")
    for img_path in tqdm(image_paths):
        try:
            caption = caption_image(vlm_processor, vlm_model, device, img_path, max_length=max_length)
            dataset_records.append({"image_path": img_path, "caption": caption})
        except Exception as e:
            # Keep going, but record the error for visibility
            dataset_records.append({"image_path": img_path, "caption": "", "error": str(e)})

    # --- 5. Save Dataset Metadata ---
    df = pd.DataFrame(dataset_records)
    metadata_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(metadata_path, index=False)
    print(f"Captioning complete. Metadata saved to {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions for images in a directory using BLIP.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with decoded images to caption.")
    parser.add_argument("--output_dir", type=str, default="vlm_captions", help="Output directory for metadata CSV.")
    parser.add_argument("--no_recursive", action="store_true", help="Disable recursive search in subdirectories.")
    parser.add_argument("--max_length", type=int, default=50, help="Max length for generated captions.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=not args.no_recursive,
        max_length=args.max_length,
    )
