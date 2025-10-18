
# 🧪 Testing HC-SCDNet with Pre-trained Models
## Quick Start Guide (Before Fine-tuning)

## Why Test First?

Testing with pre-trained models allows you to:
- ✅ **Verify architecture** works correctly in minutes
- ✅ **Get baseline results** without waiting for training
- ✅ **Test controllability** immediately
- ✅ **Identify bugs** before investing in training
- ✅ **Establish comparison baseline** for fine-tuned model

## Step-by-Step Testing Guide

### 1. Install Dependencies (1 minute)

```bash
pip install torch torchvision diffusers transformers accelerate
pip install pillow numpy matplotlib gradio
```

### 2. Prepare Test Images (2 minutes)

Create a `test_images` directory:
```bash
mkdir test_images
```

Download or add:
- `test_images/content.jpg` - A photo (e.g., portrait, landscape)
- `test_images/style.jpg` - An artistic painting (e.g., Van Gogh, Monet)

Example sources:
- Content: Your own photos or from Unsplash
- Style: WikiArt paintings or Google "famous paintings"

### 3. Run Quick Test (3-5 minutes)

```bash
python test_pretrained_models.py
```

This will:
- Load Stable Diffusion components (~5GB download first time)
- Test all controllability features
- Measure speed and memory
- Output performance metrics

### 4. Test with Real Images (5 minutes)

Create this test script:

```python
# test_real_images.py
from test_pretrained_models import PretrainedHCSCDNet
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PretrainedHCSCDNet(device=device)

# Load your images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

content_img = Image.open('test_images/content.jpg').convert('RGB')
style_img = Image.open('test_images/style.jpg').convert('RGB')

content_tensor = transform(content_img).unsqueeze(0).to(device)
style_tensor = transform(style_img).unsqueeze(0).to(device)

# Generate with different control settings
results = {}

print("Generating variations...")

# Test different combinations
settings = [
    ("low_style", 0.3, 1.0),
    ("medium_style", 0.7, 1.0),
    ("high_style", 1.0, 1.0),
    ("low_content", 1.0, 0.3),
    ("balanced", 0.7, 0.7),
]

for name, style_str, content_pres in settings:
    print(f"  {name}: style={style_str}, content={content_pres}")

    with torch.no_grad():
        result = model.controllable_style_transfer(
            content_tensor,
            style_tensor,
            style_strength=style_str,
            content_preservation=content_pres,
            num_inference_steps=15,
            style_prompt="artistic painting in the style of the reference"
        )

    # Convert to PIL and save
    result_img = (result[0] + 1) / 2  # [-1,1] -> [0,1]
    result_img = result_img.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    result_pil = Image.fromarray((result_img * 255).astype('uint8'))
    result_pil.save(f'test_images/result_{name}.jpg')
    print(f"    Saved: test_images/result_{name}.jpg")

print("\nDone! Check test_images/ for results")
```

Run it:
```bash
python test_real_images.py
```

### 5. Create Controllability Grid (2 minutes)

```python
# create_control_grid.py
from test_pretrained_models import PretrainedHCSCDNet
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PretrainedHCSCDNet(device=device)

# Load images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

content_img = Image.open('test_images/content.jpg').convert('RGB')
style_img = Image.open('test_images/style.jpg').convert('RGB')

content_tensor = transform(content_img).unsqueeze(0).to(device)
style_tensor = transform(style_img).unsqueeze(0).to(device)

# Create 3x3 grid
style_values = [0.2, 0.6, 1.0]
content_values = [0.2, 0.6, 1.0]

grid_images = []

print("Generating 3x3 control grid...")

for i, style_val in enumerate(style_values):
    row_images = []
    for j, content_val in enumerate(content_values):
        print(f"  Row {i+1}, Col {j+1}: style={style_val}, content={content_val}")

        with torch.no_grad():
            result = model.controllable_style_transfer(
                content_tensor,
                style_tensor,
                style_strength=style_val,
                content_preservation=content_val,
                num_inference_steps=10
            )

        # Convert to numpy
        result_np = (result[0] + 1) / 2
        result_np = result_np.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        result_np = (result_np * 255).astype('uint8')

        row_images.append(result_np)

    # Concatenate row
    row_concat = np.concatenate(row_images, axis=1)
    grid_images.append(row_concat)

# Concatenate all rows
grid_array = np.concatenate(grid_images, axis=0)
grid_img = Image.fromarray(grid_array)
grid_img.save('test_images/controllability_grid.jpg')

print("\nSaved: test_images/controllability_grid.jpg")
print("Grid layout:")
print("  Rows (top to bottom): Style strength 0.2, 0.6, 1.0")
print("  Cols (left to right): Content preservation 0.2, 0.6, 1.0")
```

Run it:
```bash
python create_control_grid.py
```

## Expected Results

### ✅ What Should Work:
1. **Basic generation** completes without errors
2. **Speed**: 5-15 seconds per image (will improve after fine-tuning)
3. **Memory**: Should fit in 4-8GB GPU RAM
4. **Controllability**: Different settings produce visually different results

### ⚠️ What Won't Be Perfect (Yet):
1. **Style quality** may be moderate (will improve with fine-tuning)
2. **Content preservation** may not be perfect
3. **Disentanglement** won't be as clean as after β-VAE training
4. Some artifacts or blurriness possible

## Decision Points

### ✅ Proceed to Fine-tuning IF:
- Architecture runs without errors
- You can see controllability working (different settings = different outputs)
- Speed is reasonable (within 2x of target)
- Basic style transfer is happening (even if imperfect)

### ❌ Debug First IF:
- Crashes or out-of-memory errors
- No visual differences between control settings
- Complete failure to transfer style
- Takes >1 minute per image

## Advantages of This Approach

| Aspect | Pre-trained Test | Direct Fine-tuning |
|--------|-----------------|-------------------|
| Time to first results | 10 minutes | 5-7 days |
| Can verify architecture | ✅ Yes | ❌ No (find out after training) |
| Baseline for comparison | ✅ Yes | ❌ No |
| Resource usage | Minimal | Heavy (GPU days) |
| Risk | Very low | Medium-high |

## Next Steps After Testing

### If Results are Good:
1. Document baseline metrics
2. Proceed with Phase 1 fine-tuning (β-VAE)
3. Compare fine-tuned results with this baseline
4. Iterate and improve

### If Results Need Work:
1. Adjust controllability parameters
2. Try different style prompts
3. Modify blending strategy
4. Test with different image pairs

## Pro Tips

1. **Start small**: Test with 256x256 first, then 512x512
2. **Use diverse images**: Test multiple content/style combinations
3. **Document everything**: Save all test results with settings used
4. **Compare visually**: Create side-by-side comparisons
5. **Measure quantitatively**: Run evaluation metrics on test results

## Troubleshooting

### Out of Memory?
```python
# Reduce image size
transforms.Resize((256, 256))  # Instead of 512

# Reduce inference steps
num_inference_steps=5  # Instead of 10-20
```

### Too slow?
```python
# Use fewer steps
num_inference_steps=5

# Enable xformers (if installed)
model.unet.enable_xformers_memory_efficient_attention()
```

### Poor quality?
```python
# Increase steps
num_inference_steps=30

# Adjust prompts
style_prompt="high quality artistic painting, detailed, masterpiece"
```

---

**You can get working results in 15 minutes total!**

This lets you validate the entire approach before committing to the 2-week fine-tuning process.
