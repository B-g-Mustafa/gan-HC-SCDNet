# SD1.5 Fine-tuning Suite

Complete training scripts for fine-tuning Stable Diffusion 1.5 on custom style transfer dataset using multiple methods.

## Dataset Structure

Your CSV should have these columns:
- `Content Path`: Path to content image (COCO)
- `Style Path`: Path to style image (WikiArt)
- `Image Path`: Path to synthetic style-transferred image
- `Final Caption`: Text description of the style-transferred image
- `Style Name`: Name of the art style
- `Seed`: Random seed used for generation

## Training Methods

### 1. LoRA Fine-tuning (`train_lora_proper.py`)
**What it does**: Trains lightweight LoRA adapters on top of SD1.5  
**Input**: Synthetic images + final captions  
**Best for**: Efficient fine-tuning with minimal parameters  
**Training time**: ~2-3 days on A100  

**Key features**:
- Proper diffusion loss (MSE between predicted and actual noise)
- Gradient accumulation for large effective batch size
- Trains only LoRA parameters (~16M params)
- Saves best model + periodic checkpoints

### 2. ControlNet Training (`train_controlnet.py`)
**What it does**: Trains ControlNet to generate style-transferred images from content images  
**Input**: Content images (conditioning) + synthetic images (target) + final captions  
**Best for**: Controllable generation with spatial guidance  
**Training time**: ~5-7 days on A100  

**Key features**:
- Uses content images as spatial conditioning
- Generates style-transferred images matching content structure
- Can be combined with fine-tuned SD1.5
- Preserves content while applying style

### 3. DreamBooth Training (`train_dreambooth.py`)
**What it does**: Fine-tunes entire SD1.5 model on style-specific images  
**Input**: Synthetic images + style-specific prompts  
**Best for**: Learning specific artistic styles deeply  
**Training time**: ~3-5 days on A100  

**Key features**:
- Trains both UNet and text encoder
- Style-specific prompting ("a photo in {style} style")
- Optional prior preservation (simplified in current version)
- Full model fine-tuning

## Quick Start

### 1. Setup Environment
```bash
pip install torch torchvision
pip install diffusers transformers accelerate peft
pip install pandas pillow tqdm opencv-python
```

### 2. Update Configuration
Edit each training script and update these paths:
```python
CSV_PATH = "/path/to/your/combined.csv"
IMAGE_DIR = "/path/to/your/images/"
OUTPUT_DIR = "/path/to/output/"
```

### 3. Run Training

**Option A: Run individual method**
```bash
# LoRA training
python train_lora_proper.py

# ControlNet training
python train_controlnet.py

# DreamBooth training
python train_dreambooth.py
```

**Option B: Use the launcher**
```bash
# Run all methods sequentially
python run_all_training.py --method all

# Run specific method
python run_all_training.py --method lora
python run_all_training.py --method controlnet
python run_all_training.py --method dreambooth
```

**Option C: Submit to SLURM cluster**
```bash
# Generate SLURM scripts
bash create_slurm_scripts.sh

# Submit jobs
sbatch run_lora.sh
sbatch run_controlnet.sh
sbatch run_dreambooth.sh
```

## Inference and Evaluation

After training, generate comparison images:

```bash
python inference_comparison.py
```

This will:
1. Load all trained models
2. Generate images with each method
3. Create side-by-side comparison grids
4. Save results to output directory

Output includes:
- Comparison images showing: Content, Ground Truth, Base SD1.5, LoRA, ControlNet, DreamBooth
- CSV with generation metadata
- Individual model outputs

## Hyperparameter Tuning

### LoRA
```python
LORA_RANK = 16              # Higher = more capacity, more memory
LORA_ALPHA = 32             # Scaling factor
LEARNING_RATE = 1e-4        # Try 5e-5 to 2e-4
NUM_EPOCHS = 20             # Monitor validation loss
```

### ControlNet
```python
LEARNING_RATE = 1e-5        # Lower than LoRA (more parameters)
NUM_EPOCHS = 50             # Requires more epochs
CONTROLNET_CONDITIONING_SCALE = 1.0  # Adjust at inference (0.5-1.5)
```

### DreamBooth
```python
LEARNING_RATE = 2e-6        # Very small (full fine-tuning)
LEARNING_RATE_TEXT = 5e-7   # Even smaller for text encoder
NUM_EPOCHS = 30             # Monitor overfitting
```

## Expected Results

### LoRA
- ✓ Fast training, low memory
- ✓ Good style reproduction
- ✓ Easy to combine multiple LoRAs
- ✗ May not capture all style nuances

### ControlNet
- ✓ Excellent content preservation
- ✓ Controllable style transfer
- ✓ Works with any SD1.5 checkpoint
- ✗ Slower inference (two models)

### DreamBooth
- ✓ Deep style learning
- ✓ High-quality results
- ✓ Style-specific generations
- ✗ May overfit, requires care

## Monitoring Training

All scripts log to CSV files:
- `lora_training_log.csv`: Loss per epoch
- `controlnet_training_log.csv`: Loss per epoch
- `dreambooth_training_log.csv`: Loss per epoch

Watch logs in real-time:
```bash
tail -f /path/to/output/logs/*.log
```

## Troubleshooting

**CUDA Out of Memory**:
- Reduce `BATCH_SIZE`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Use smaller resolution (256x256)
- Enable gradient checkpointing

**Training diverges**:
- Lower learning rate
- Add gradient clipping (already included)
- Check data quality
- Reduce batch size

**Poor results**:
- Train longer (more epochs)
- Check data distribution
- Adjust LoRA rank/alpha
- Try different random seeds

## File Structure

```
SDFinetune/
├── train_lora_proper.py          # LoRA training
├── train_controlnet.py            # ControlNet training
├── train_dreambooth.py            # DreamBooth training
├── inference_comparison.py        # Evaluation script
├── run_all_training.py            # Master launcher
├── create_slurm_scripts.sh        # Generate SLURM scripts
├── finetune.py                    # Original (placeholder loss - don't use)
└── README.md                      # This file
```

## Next Steps

1. **Train LoRA first** - fastest to validate your dataset
2. **Run inference** - check if results are good
3. **Train ControlNet** - for controllable generation
4. **Compare results** - use inference_comparison.py
5. **Iterate** - adjust hyperparameters based on results

## Citation

Based on:
- Stable Diffusion: https://github.com/Stability-AI/stablediffusion
- LoRA: https://arxiv.org/abs/2106.09685
- ControlNet: https://arxiv.org/abs/2302.05543
- DreamBooth: https://arxiv.org/abs/2208.12242
