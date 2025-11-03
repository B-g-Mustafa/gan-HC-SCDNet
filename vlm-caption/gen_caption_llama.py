from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import os
import re
import pandas as pd
from tqdm import tqdm

def list_images(input_dir: str, recursive: bool = True):
    """Collect image file paths from a directory."""
    img_paths = []
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

def extract_json_from_text(text: str, img_path: str):
    pattern = re.compile(
    r"\*\*Content Caption:\*\*\s*([^\n\r]*)\n+\*\*Style Caption:\*\*\s*([^\n\r]*)\n+\*\*Style Name:\*\*\s*([^\n\r]*)\n+\*\*Final Caption:\*\*\s*((?:(?!<\|eot_id\|>).)*)",
    re.DOTALL
    )
    match = pattern.search(text)
    if match:
        content_caption, style_caption, style_name, final_caption = match.groups()
        if(content_caption == '' or style_caption == '' or style_name == '' or final_caption == ''):
            print("some of the captions are missing: ")
            return None
        else:
            val_dict = {}
            val_dict["Content Caption"] = content_caption
            val_dict["Style Caption"] = style_caption
            val_dict["Style Name"] = style_name
            val_dict["Final Caption"] = final_caption
            val_dict["Image Path"] = img_path
    return val_dict

def save_batch_to_csv(batch_data, batch_num, output_dir="/home/msai/birul001/BIRUL001/data/synthetic_caption_dataset/llama"):
    """Save a batch of outputs to a CSV file"""
    if not batch_data:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"captions_batch_{batch_num:04d}.csv")
    
    df = pd.DataFrame(batch_data)
    df.to_csv(output_file, index=False)
    print(f"✓ Saved batch {batch_num} to {output_file}")

def create_dataset():
    if len(image_paths) == 0:
        print(f"No images found in {input_dir}. Nothing to do.")
        return

    print(f"Captioning {len(image_paths)} images from: {input_dir}")
    print(f"Processing in batches of {BATCH_SIZE}")
    
    batch_output_list = []
    batch_num = 1
    
    for idx, img_path in enumerate(tqdm(image_paths), 1):
        try:
            # Load and preprocess the image
            image = Image.open(img_path)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(image, input_text, return_tensors="pt").to(model.device)
            
            output = model.generate(**inputs, max_new_tokens=300)
            output_text = processor.decode(output[0])
            output_json = extract_json_from_text(output_text, img_path)
            
            if output_json is not None:
                batch_output_list.append(output_json)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            batch_output_list.append({
                "Image Path": img_path,
                "Content Caption": "",
                "Style Caption": "",
                "Style Name": "",
                "Final Caption": "",
                "Error": str(e)
            })
        
        # Save batch when we reach BATCH_SIZE or at the end
        if idx % BATCH_SIZE == 0 or idx == len(image_paths):
            save_batch_to_csv(batch_output_list, batch_num)
            batch_output_list = []  # Clear the batch list
            batch_num += 1
            
            # Optional: clear CUDA cache every few batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto"
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
BATCH_SIZE = 100
input_dir = "/home/msai/birul001/BIRUL001/data/synthetic_dataset"
image_paths = list_images(input_dir)
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "You are an expert vision-language model trained to describe both the *content* and *style* of images " "for synthetic dataset annotation. Analyze the image carefully and produce captions that cover three aspects:\n" "1. The **content** — what objects, scenes, or actions are visible.\n" "2. The **style** — describe the artistic or visual characteristics such as brushwork, color palette, texture, " "or digital art traits. If recognizable, identify the **style name or movement** (e.g., 'Van Gogh', 'Impressionism', " "'Cyberpunk', 'Watercolor', '3D render', 'Anime', etc.).\n" "3. The **final caption** — combine both content and style naturally, describing how the two interact visually.\n\n" "Output must be in valid JSON format with **four fields**:\n" "{\n" "  'content_caption': str,        # factual description of the image’s content\n" "  'style_caption': str,          # description of style with possible style name\n" "  'style_name': str,             # name of the recognized style or 'Unknown' if not clear\n" "  'final_caption': str           # natural combined caption reflecting both\n" "}\n\n" "Rules:\n" "- Be factual and concise (1–2 sentences per field).\n" "- Do not hallucinate or make assumptions beyond visible traits.\n" "- Avoid extra commentary or explanations.\n" "- Always output valid JSON only — no markdown, no prefixes, no code blocks."}
    ]}
]

create_dataset()