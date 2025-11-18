# Quick Reference: Distributed Training Commands

## Start Training (Easy Way)
```bash
cd SDFinetune
./run_distributed.sh
```

## Start Training (Manual - 2 Terminals)

**Terminal 1 (GPU 0):**
```bash
python train_sd35_lora.py --gpu 0 --world_size 2 --master_addr localhost --master_port 12355
```

**Terminal 2 (GPU 1):**
```bash
python train_sd35_lora.py --gpu 1 --world_size 2 --master_addr localhost --master_port 12355
```

## Monitor Training

```bash
# Watch main process
tail -f gpu0.log

# Watch worker process  
tail -f gpu1.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check WandB
# Visit: https://wandb.ai/<your-username>/style-transfer-sd35-lora
```

## Stop Training

```bash
# Find processes
ps aux | grep train_sd35_lora

# Kill gracefully
kill <PID_GPU0> <PID_GPU1>

# Force kill
pkill -9 -f train_sd35_lora

# Clean up port
lsof -ti:12355 | xargs kill -9
```

## Configuration Summary

| Parameter | Value |
|-----------|-------|
| GPUs | 2× NVIDIA A100 40GB |
| Batch size per GPU | 4 |
| Gradient accumulation | 4 |
| Effective batch per GPU | 16 |
| Total effective batch | 32 |
| Learning rate | 5e-5 |
| Epochs | 20 |
| Dataset | 60k samples |
| Model | SD3.5 Large (~8B) |
| LoRA rank | 32 |
| Precision | FP16 |

## Expected Training Time

- **Per epoch**: ~4-5 hours (2 GPUs)
- **20 epochs**: ~80-100 hours
- **Speedup vs 1 GPU**: ~2x

## Verification Checklist

- [ ] Both Python processes running (`ps aux | grep train`)
- [ ] Both GPUs active in nvidia-smi (~90-100% utilization)
- [ ] Log files created: `gpu0.log` and `gpu1.log`
- [ ] WandB showing updates (GPU 0 only)
- [ ] No CUDA OOM errors in logs
- [ ] Loss decreasing over time

## Common Issues

**Connection timeout:**
```bash
lsof -ti:12355 | xargs kill -9
# Then restart both processes
```

**One GPU idle:**
- Restart both processes within 30 seconds
- Use launcher script for automatic timing

**Out of memory:**
- Reduce `BATCH_SIZE` from 4 to 3 or 2
- Or reduce `GRADIENT_ACCUMULATION_STEPS`

## Process Indicators

**GPU 0 (Main) should show:**
- 🚀 DISTRIBUTED TRAINING - PROCESS 0/1
- Progress bars (tqdm)
- WandB logging
- Checkpoint saving messages
- "✓ Checkpoint saved" every 5 epochs

**GPU 1 (Worker) should show:**
- 🚀 DISTRIBUTED TRAINING - PROCESS 1/1
- "[GPU 1] ⚡ Worker process ready"
- "[GPU 1] Step X/Y - Loss: Z"
- No WandB messages
- No checkpoint messages
