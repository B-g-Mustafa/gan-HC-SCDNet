# Distributed Training Guide for SD3.5 LoRA Fine-tuning

## Overview

This script supports **distributed fine-tuning across 2 GPUs** using PyTorch's DistributedDataParallel (DDP). Each GPU runs a separate Python process that coordinates with the other for gradient synchronization.

## Configuration

### Optimized for NVIDIA A100 40GB

- **Batch size per GPU**: 4
- **Gradient accumulation steps**: 4
- **Effective batch size per GPU**: 16 (4 × 4)
- **Total effective batch size**: 32 (16 × 2 GPUs)
- **Mixed precision**: FP16
- **Model**: Stable Diffusion 3.5 Large (~8B parameters)

## How to Run

### Option 1: Using the Launcher Script (Recommended)

```bash
cd SDFinetune
chmod +x run_distributed.sh
./run_distributed.sh
```

This will:
1. Launch GPU 0 (main process) and GPU 1 (worker process)
2. Create log files: `gpu0.log` and `gpu1.log`
3. Monitor both processes
4. Show you how to check progress

### Option 2: Manual Launch (Two Terminals)

**Terminal 1 (GPU 0 - Main Process):**
```bash
cd SDFinetune
python train_sd35_lora.py --gpu 0 --world_size 2 --master_addr localhost --master_port 12355
```

**Terminal 2 (GPU 1 - Worker Process):**
```bash
cd SDFinetune
python train_sd35_lora.py --gpu 1 --world_size 2 --master_addr localhost --master_port 12355
```

⚠️ **Important**: Start both terminals within 30 seconds of each other!

## What You'll See

### GPU 0 (Main Process)
```
================================================================================
🚀 DISTRIBUTED TRAINING - PROCESS 0/1
================================================================================
✓ Process Rank: 0
✓ GPU Device: cuda:0
✓ World Size: 2
✓ Master: localhost:12355
✓ Backend: NCCL
================================================================================

================================================================================
🎯 STARTING DISTRIBUTED FINE-TUNING
================================================================================
Total GPUs: 2
Batch size per GPU: 4
Gradient accumulation: 4
Effective batch per GPU: 16
Total effective batch: 32
Learning rate: 5e-05
Epochs: 20
GPU Memory: A100 40GB
================================================================================

📦 LOADING MODEL - STABLE DIFFUSION 3.5 LARGE
...
🏋️  EPOCH 1/20 - TRAINING IN PROGRESS
...
```

### GPU 1 (Worker Process)
```
================================================================================
🚀 DISTRIBUTED TRAINING - PROCESS 1/1
================================================================================
✓ Process Rank: 1
✓ GPU Device: cuda:1
✓ World Size: 2
✓ Master: localhost:12355
✓ Backend: NCCL
================================================================================

[GPU 1] ⚡ Worker process ready and waiting for main process...

[GPU 1] 📦 Loading model on cuda:1...
[GPU 1] ✓ Model loaded on cuda:1
[GPU 1] 🔧 Applying LoRA...
[GPU 1] ✓ DDP wrapper applied
[GPU 1] 📊 Loading dataset...
[GPU 1] ✓ Dataset ready: 30,000 samples
[GPU 1] 🚀 Ready to start training loop

[GPU 1] 🏋️  Epoch 1/20 - Fine-tuning in progress...
[GPU 1] Step 100/7500 - Loss: 0.0234
...
```

## Monitoring

### Check Training Progress
```bash
# Main process (shows progress bars)
tail -f gpu0.log

# Worker process
tail -f gpu1.log

# Both at once
tail -f gpu0.log gpu1.log
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

You should see both GPUs with high utilization (~90-100%)

### Check WandB
Only GPU 0 logs to WandB. Check your dashboard at:
```
https://wandb.ai/<your-username>/style-transfer-sd35-lora
```

## Key Features

### 1. Process Identification
- **GPU 0 (Main)**: Handles WandB logging, checkpointing, and progress bars
- **GPU 1 (Worker)**: Processes data, computes gradients, syncs with main

### 2. Automatic Gradient Synchronization
- DDP automatically averages gradients across GPUs
- No manual synchronization needed

### 3. Coordinated Checkpointing
- Only GPU 0 saves checkpoints to avoid conflicts
- Models saved every 5 epochs and at completion

### 4. Distributed Data Loading
- Dataset automatically split between GPUs
- Each GPU processes ~30k samples (60k total ÷ 2)
- DistributedSampler ensures no overlap

## Troubleshooting

### Problem: "Connection timeout" or "Address already in use"

**Solution**: Kill existing processes and change port
```bash
# Kill processes on port 12355
lsof -ti:12355 | xargs kill -9

# Or use a different port
python train_sd35_lora.py --gpu 0 --world_size 2 --master_port 12356
```

### Problem: "One GPU is idle"

**Solution**: Ensure both processes started within 30 seconds
- Restart both processes
- Or use the launcher script which handles timing

### Problem: "CUDA out of memory"

**Solution**: Reduce batch size
```python
# In train_sd35_lora.py, change:
BATCH_SIZE = 3  # Instead of 4
```

### Problem: "Different loss values between GPUs"

**Solution**: This is normal!
- Each GPU processes different data batches
- Gradients are synchronized, not losses
- Only compare final epoch losses

## Stopping Training

### Gracefully
```bash
# Find process IDs
ps aux | grep train_sd35_lora

# Kill both processes
kill <GPU0_PID> <GPU1_PID>
```

### Force Stop
```bash
pkill -9 -f train_sd35_lora
```

## Performance Expectations

### Training Speed
- **Single GPU A100**: ~8-10 hours per epoch (60k dataset)
- **2× GPU A100 (distributed)**: ~4-5 hours per epoch
- **20 epochs**: ~80-100 hours → **40-50 hours with 2 GPUs**

### Memory Usage
- **Per GPU**: ~35-38 GB / 40 GB
- **Model**: ~16 GB (FP16)
- **Activations + Gradients**: ~20 GB
- **Safe margin**: ~2-5 GB

## Advanced Options

### Different Number of GPUs

For **4 GPUs**:
```bash
# Terminal 1
python train_sd35_lora.py --gpu 0 --world_size 4

# Terminal 2
python train_sd35_lora.py --gpu 1 --world_size 4

# Terminal 3
python train_sd35_lora.py --gpu 2 --world_size 4

# Terminal 4
python train_sd35_lora.py --gpu 3 --world_size 4
```

### Multiple Nodes (Cluster)

**Node 1** (IP: 192.168.1.10):
```bash
python train_sd35_lora.py --gpu 0 --world_size 4 --master_addr 192.168.1.10 --master_port 12355
python train_sd35_lora.py --gpu 1 --world_size 4 --master_addr 192.168.1.10 --master_port 12355
```

**Node 2** (IP: 192.168.1.11):
```bash
python train_sd35_lora.py --gpu 2 --world_size 4 --master_addr 192.168.1.10 --master_port 12355
python train_sd35_lora.py --gpu 3 --world_size 4 --master_addr 192.168.1.10 --master_port 12355
```

## Output Files

### Checkpoints
```
/home/msai/birul001/BIRUL001/models/sd35_lora/
├── checkpoint-epoch-5/          # After epoch 5
├── checkpoint-epoch-10/         # After epoch 10
├── checkpoint-epoch-15/         # After epoch 15
├── checkpoint-epoch-20/         # After epoch 20
├── final/                       # Final LoRA weights
└── pipeline-final/              # Complete pipeline
```

### Logs
```
SDFinetune/
├── gpu0.log                     # Main process output
├── gpu1.log                     # Worker process output
└── train_sd35_lora.py           # Training script
```

## Verification

### Both GPUs are Training
```bash
# Should show 2 Python processes
ps aux | grep train_sd35_lora

# Should show both GPUs active
nvidia-smi
```

### Gradients are Syncing
Check the logs - loss values should be reasonably similar (not identical) between GPUs.

### WandB Logging
Only GPU 0 logs. You should see:
- `train/loss` every 50 steps
- `epoch/loss` every epoch
- `checkpoint/epoch` every 5 epochs

## Questions?

The distributed training uses:
- **PyTorch DistributedDataParallel (DDP)**
- **NCCL backend** for GPU communication
- **DistributedSampler** for data splitting
- **Process group** for coordination

Each process is **completely independent** but synchronizes gradients automatically via NCCL collective operations.
