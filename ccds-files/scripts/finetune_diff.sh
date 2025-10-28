#!/bin/bash
#SBATCH --partition=MGPU-TC2          # GPU partition
#SBATCH --qos=normal                  # QoS level
#SBATCH --nodes=1                     # Use 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=30G                     # Allocate 30 GB of memory
#SBATCH --time=05:55:00               # Max job time = 5 hours 55 minutes
#SBATCH --job-name=vae_train          # Job name
#SBATCH --output=/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files/logs/finetune/output_%x_%j.out   # Std output log
#SBATCH --error=/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files/logs/finetune/error_%x_%j.err     # Std error log

export WANDB_API_KEY="d9a74b72096b984643e4b3246a816e62be94d572"
export HF_TOKEN="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"
# -----------------------------
# Step 2: Activate your environment
# -----------------------------
HOME_PATH="/home/msai/birul001/"   # <-- change this to your conda/env path
source "${HOME_PATH}/birul001-gan/bin/activate"

# -----------------------------
# Step 2.1: Login to Hugging Face (uses token env var)
# -----------------------------
# export HF_TOKEN="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"
# echo "Logging into Hugging Face Hub..."
# huggingface-cli login --token "$HF_TOKEN"
# -----------------------------
# Step 3: Ensure required packages are installed
# -----------------------------
echo "Checking and installing required Python libraries..."
REQUIREMENTS_FILE="/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files/requirement.txt"  # path to your requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "No requirements.txt found, installing default AI libraries..."
    pip install torch torchvision torchaudio transformers numpy
fi

# -----------------------------
# Step 3: Define paths for the AI task
# -----------------------------
PYTHON_SCRIPT="/home/msai/birul001/gan-project/gan-HC-SCDNet/SDFinetune/fintune_per.py"  # <-- your Python script path
# CONTENT_PATH="/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/data/coco_split/train/"
# CONTENT_PATH_val="/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/data/coco_split/test/"
# STYLE_PATH="/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/data/wikiart_split/train/"
# STYLE_PATH_val="/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/data/wikiart_split/test/"
# CHECKPOINT_PATH="/home/msai/birul001/gan-project/gan-HC-SCDNet/b-vae/model/checkpoints/"

# -----------------------------
# Step 3.1: Prepare a new output file for this run
# -----------------------------
CCDS_DIR="/home/msai/birul001/gan-project/gan-HC-SCDNet/ccds-files"
mkdir -p "$CCDS_DIR"

# Prepare finetune_output directory (fixes duplicated path issue)
FINETUNE_OUTPUT_DIR="${CCDS_DIR}/finetune_output"
mkdir -p "$FINETUNE_OUTPUT_DIR"

# Create a unique, timestamped output file and write a header
RUN_TS=$(date +"%Y%m%d_%H%M%S")
JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_TXT="${FINETUNE_OUTPUT_DIR}/output_${RUN_TS}_${JOB_ID}.txt"
{
  echo "==== VAE Training Run ===="
  echo "Started: $(date)"
  echo "Job Name: ${SLURM_JOB_NAME:-VAE-Train}"
  echo "Job ID: ${SLURM_JOB_ID:-N/A}"
  echo "Node: $(hostname)"
  echo "GPU(s) requested: ${SLURM_GPUS:-1}"
  echo "Script: $PYTHON_SCRIPT"
  echo "Content Images Path: $CONTENT_PATH"
  echo "Content Images Path (Validation): $CONTENT_PATH_val"
  echo "Style Images Path: $STYLE_PATH"
  echo "Style Images Path (Validation): $STYLE_PATH_val"
  echo "Checkpoint Path: $CHECKPOINT_PATH"
  echo "Output File: $OUTPUT_TXT"
  echo "==========================="
  echo
} > "$OUTPUT_TXT"

# -----------------------------
# Step 4: Run your AI task
# -----------------------------
echo "Running AI job on GPU..."

if [ -f "$PYTHON_SCRIPT" ]; then
    echo "Executing: python $PYTHON_SCRIPT" >> "$OUTPUT_TXT"
    python "$PYTHON_SCRIPT" >> "$OUTPUT_TXT" 2>&1
else
    echo "Python script not found at $PYTHON_SCRIPT" >> "$OUTPUT_TXT"
fi

# -----------------------------
# Step 5: Job completion message
# -----------------------------
echo "AI job completed at $(date)" | tee -a "$OUTPUT_TXT"
