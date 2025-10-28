import os
import random
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
CONTENT_DIR = "content/"
STYLE_DIR = "style/"
OUTPUT_DIR = "synthetic/images/"
METADATA_PATH = "synthetic/metadata.csv"
NUM_SAMPLES = 50000
STYLE_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load your pre-trained VAE here
# Example placeholder: replace with your actual VAE
class DummyVAE:
    def encode(self, x):
        return x.float() / 255.0  # dummy latent
    def decode(self, z):
        return (z * 255).byte()   # dummy decode

vae = DummyVAE()

# Image transform
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

# --- UTILITIES ---
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)  # shape: [1,3,512,512]

def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(path)

# --- MAIN ---
# Get list of images
content_files = [os.path.join(CONTENT_DIR,f) for f in os.listdir(CONTENT_DIR)]
style_files = [os.path.join(STYLE_DIR,f) for f in os.listdir(STYLE_DIR)]

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata_rows = []

for i in tqdm(range(NUM_SAMPLES)):
    # Random selection
    content_path = random.choice(content_files)
    style_path = random.choice(style_files)
    weight = random.choice(STYLE_WEIGHTS)
    seed = random.randint(0, 1e6)
    random.seed(seed)
    torch.manual_seed(seed)

    # Load images
    content_img = load_image(content_path)
    style_img = load_image(style_path)

    # Encode
    z_c = vae.encode(content_img)
    z_s = vae.encode(style_img)

    # Latent fusion
    z_fused = (1 - weight) * z_c + weight * z_s

    # Decode
    synth_img = vae.decode(z_fused)

    # Save image
    filename = f"synthetic_{i:05d}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    save_image(synth_img, save_path)

    # Save metadata
    metadata_rows.append({
        "image_id": filename,
        "content_path": content_path,
        "style_path": style_path,
        "weight": weight,
        "seed": seed
    })

# Save metadata CSV
df = pd.DataFrame(metadata_rows)
os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
df.to_csv(METADATA_PATH, index=False)

print(f"Done! {NUM_SAMPLES} synthetic images saved to {OUTPUT_DIR}")
#Need to process this
from __future__ import print_function
import argparse

import os
import torch
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

parser = argparse.ArgumentParser(description='LT-VAE Style transfer')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--image_dataset', type=str, default='Test')
parser.add_argument("--latent", type=int, default=256, help='length of latent vector')
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/matrix_r41_new.pth', help='pre-trained model path')


# opt = parser.parse_args()
# opt=parser.parse_known_args()
# opt = list(opt)
# print(opt[0][0])
device = 'cpu'

# map_location=torch.device('cpu')
vgg = encoder4()
dec = decoder4()
matrix = MulLayer(z_dim=256)
vgg.load_state_dict(torch.load('models/vgg_r41.pth',map_location=torch.device('mps')))
dec.load_state_dict(torch.load('models/dec_r41.pth',map_location=torch.device('mps')))
matrix.load_state_dict(torch.load('models/matrix_r41_new.pth',map_location=torch.device('mps')))

vgg.to(device)
dec.to(device)
matrix.to(device)



transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

alpha=0.00001
matrix.eval()
vgg.eval()
dec.eval()
# content_path = os.path.join(opt.image_dataset, 'content')
# output_path = os.path.join(opt.image_dataset, 'result')
# ref_path = os.path.join(opt.image_dataset, 'style')

t0 = time.time()
content = Image.open('Test/content/in3.png').convert('RGB')
ref = Image.open('Test/style/picasso_self_portrait.jpg').convert('RGB')

content = transform(content).unsqueeze(0).to(device)
# print("content",content)
ref = transform(ref).unsqueeze(0).to(device)
# print("ref",ref)

with torch.no_grad():
    sF = vgg(ref)
    cF = vgg(content)
    feature, _, _,out2 = matrix(cF['r41'], sF['r41'],alpha)
    #out2 is newly predicted
    prediction = dec(out2)

    prediction = prediction.data[0].cpu().permute(1, 2, 0)

t1 = time.time()
#print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

prediction = prediction * 255.0
prediction = prediction.clamp(0, 255)

# file_name = cont_file.split('.')[0] + '_' + ref_file.split('.')[0] + '.jpg'
# save_name = os.path.join('self-tests', 'test2.jpg')
# Image.fromarray(np.uint8(prediction)).save(save_name)
Image.fromarray(np.uint8(prediction))
