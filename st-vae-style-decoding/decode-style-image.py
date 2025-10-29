import random
import torch
import pandas as pd
from tqdm import tqdm
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
from PIL import Image, ImageOps
import os
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer
from pathlib import Path

# --- CONFIGURATION ---
print("Starting style decoding script...")
CONTENT_DIR = "/home/msai/birul001/BIRUL001/data/coco/train2017/"
STYLE_DIR = "/home/msai/birul001/BIRUL001/data/wikiart/split50/"
OUTPUT_DIR = "/home/msai/birul001/BIRUL001/data/syn2/"
METADATA_PATH = "/home/msai/birul001/BIRUL001/data/synthetic_dataset/csv/metadata.csv"
NUM_SAMPLES = 60000
STYLE_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models...")
vgg = encoder4()
dec = decoder4()
matrix = MulLayer(z_dim=256)
vgg.load_state_dict(torch.load('/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/vgg_r41.pth',map_location=torch.device(device)))
dec.load_state_dict(torch.load('/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/dec_r41.pth',map_location=torch.device(device)))
matrix.load_state_dict(torch.load('/home/msai/birul001/gan-project/gan-HC-SCDNet/st-vae-style-decoding/models/matrix_r41_new.pth',map_location=torch.device(device)))

print("Models loaded.")
vgg.to(device)
dec.to(device)
matrix.to(device)


transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

print("Preparing image lists...")
print("Validating content images...")
content_files = [os.path.join(CONTENT_DIR,f) for f in os.listdir(CONTENT_DIR)]

print("Validating style images...")
style_files = []
for root, dirs, files in os.walk(STYLE_DIR):
    for f in files:
        style_files.append(os.path.join(root, f))


print(f"Found {len(content_files)} valid content images and {len(style_files)} valid style images.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata_rows = []

matrix.eval()
vgg.eval()
dec.eval()
print("device:", device)
success_count = 0
while success_count < NUM_SAMPLES:
    # Random selection
    content_path = random.choice(content_files)
    style_path = random.choice(style_files)
    weight = random.choice(STYLE_WEIGHTS)
    seed = random.randint(0, 1e6)
    random.seed(seed)
    torch.manual_seed(seed)
    
    try:
        content = Image.open(content_path).convert('RGB')
        ref = Image.open(style_path).convert('RGB')
    except Exception as e:
        print(f"Error loading images, skipping pair: {e}")
        continue

    content = transform(content).unsqueeze(0).to(device)
    ref = transform(ref).unsqueeze(0).to(device)

    with torch.no_grad():
        sF = vgg(ref)
        cF = vgg(content)
        feature, _, _,out2 = matrix(cF['r41'], sF['r41'],weight)
        #out2 is newly predicted
        prediction = dec(out2)

        prediction = prediction.data[0].cpu().permute(1, 2, 0)

    prediction = prediction * 255.0
    prediction = prediction.clamp(0, 255)    

    save_name = os.path.join(OUTPUT_DIR, f"{success_count:0>6d}.jpg")
    # Save metadata
    metadata_rows.append({
        "image_id": save_name,
        "content_path": content_path,
        "style_path": style_path,
        "weight": weight,
        "seed": seed
    })
    success_count += 1

# Save metadata CSV
df = pd.DataFrame(metadata_rows)
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
df.to_csv(METADATA_PATH, index=False)

print(f"Done! {NUM_SAMPLES} synthetic images saved to {OUTPUT_DIR}")