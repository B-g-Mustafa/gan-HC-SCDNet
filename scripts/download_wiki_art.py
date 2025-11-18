"""
download_wikiart.py
Downloads the Hugging Face 'huggan/wikiart' dataset (256x256) and stores it locally
in data/wikiart/{style}/ folders.

Requirements:
    pip install datasets tqdm pillow
"""

import os
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

def download_wikiart(save_root="gan-project/gan-HC-SCDNet/b-vae/data/wikiart", split="train"):
    os.makedirs(save_root, exist_ok=True)
    print(f"Downloading 'huggan/wikiart' dataset to: {save_root}")

    # Disable HuggingFace cache to save directly without caching
    os.environ["HF_DATASETS_CACHE"] = save_root
    
    # Load the Hugging Face dataset (each entry has 'image' and 'style')
    dataset = load_dataset("huggan/wikiart", split=split, cache_dir=save_root)

    # Iterate and save images by style
    for i, sample in enumerate(tqdm(dataset, desc="Saving images", total=len(dataset))):
        style = sample.get("style", "unknown").replace(" ", "_")
        style_dir = os.path.join(save_root, style)
        os.makedirs(style_dir, exist_ok=True)

        img = sample["image"]
        img_path = os.path.join(style_dir, f"{i:06d}.jpg")
        if isinstance(img, Image.Image):
            img.save(img_path)
        else:
            # If dataset returns a dict with bytes
            Image.fromarray(img).save(img_path)

    print("✅ Download complete.")
    print(f"Total images saved: {len(dataset)}")
    print(f"Dataset directory structure:\n{save_root}/<style_name>/*.jpg")

if __name__ == "__main__":
    download_wikiart()
