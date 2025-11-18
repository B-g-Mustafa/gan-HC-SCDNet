# PBS Job Scheduler Guide for Distributed Training

## Overview

Your SD3.5 LoRA distributed training now supports **PBS (Portable Batch System)** job scheduler. The script automatically detects PBS environment and coordinates between 2 GPU processes.

## Quick Start

### Option 1: Automated Submission (Recommended)

```bash
cd /home/users/ntu/birul001/gan-project/git/gan-HC-SCDNet/SDFinetune
chmod +x submit_distributed_pbs.sh
./submit_distributed_pbs.sh
```

This will:
1. Submit GPU 0 job (main process)
2. Wait 10 seconds
3. Submit GPU 1 job (worker process)
4. Show you job IDs and monitoring commands

### Option 2: Manual Submission

```bash
cd /home/users/ntu/birul001/gan-project/git/gan-HC-SCDNet/SDFinetune

# Submit GPU 0 (Main Process)
qsub run_distributed_pbs_gpu0.sh

# Wait ~10 seconds, then submit GPU 1
qsub run_distributed_pbs_gpu1.sh
```

## PBS Job Scripts

### GPU 0 Script (`run_distributed_pbs_gpu0.sh`)
- **Main process** that coordinates training
- Auto-detects master IP and available port
- Writes connection info to `/tmp/sd35_master_addr.txt` and `/tmp/sd35_master_port.txt`
- Logs to WandB
- Saves checkpoints every 5 epochs

### GPU 1 Script (`run_distributed_pbs_gpu1.sh`)
- **Worker process** that assists training
- Waits for GPU 0 to write master connection info
- Reads from shared `/tmp` files
- Syncs gradients with GPU 0
- No WandB logging or checkpointing

## Resource Configuration

Both jobs request:
```bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=40GB
#PBS -l walltime=100:00:00
```

- **GPUs**: 1 per job (2 total)
- **CPUs**: 8 cores per job
- **Memory**: 40GB per job (matches A100 GPU memory)
- **Walltime**: 100 hours (~4-5 hours per epoch × 20 epochs)

## Monitoring Your Jobs

### Check Job Status
```bash
# List your jobs
qstat -u $USER

# Detailed info for specific job
qstat -f <JOB_ID>

# Watch job status
watch -n 5 'qstat -u $USER'
```

### View PBS Logs
```bash
# GPU 0 PBS log
tail -f ~/gan-project/outputs/logs/gpu0_job_output.log

# GPU 1 PBS log
tail -f ~/gan-project/outputs/logs/gpu1_job_output.log
```

### View Training Output
```bash
# GPU 0 training output (shows progress bars)
tail -f ~/gan-project/outputs/py_output/gpu0_output.txt

# GPU 1 training output (shows periodic updates)
tail -f ~/gan-project/outputs/py_output/gpu1_output.txt

# Watch both simultaneously
tail -f ~/gan-project/outputs/py_output/*.txt
```

## Expected Output

### GPU 0 (Main Process)
```
==========================================
GPU 0 (Main Process) - PBS Job
==========================================
Job ID: 12345.server
Job Name: sd35-lora-gpu0
Queue: normal
Node: gpu-node-01
Start Time: Mon Nov 18 10:00:00 SGT 2025
==========================================

📋 PBS JOB INFORMATION
Job ID: 12345.server
Job Name: sd35-lora-gpu0
Queue: normal
Node: gpu-node-01

🔍 Auto-detected master IP: 192.168.1.10
🔍 Auto-detected available port: 12355

🚀 DISTRIBUTED TRAINING - PROCESS 0/1
✓ Process Rank: 0
✓ GPU Device: cuda:0
✓ World Size: 2
✓ Master: 192.168.1.10:12355
✓ Backend: NCCL
✓ Hostname: gpu-node-01

🎯 STARTING DISTRIBUTED FINE-TUNING
Total GPUs: 2
Batch size per GPU: 4
...
```

### GPU 1 (Worker Process)
```
==========================================
GPU 1 (Worker Process) - PBS Job
==========================================
Job ID: 12346.server
Job Name: sd35-lora-gpu1
Queue: normal
Node: gpu-node-02
Start Time: Mon Nov 18 10:00:15 SGT 2025
==========================================

Waiting for GPU 0 to initialize...
Looking for master address file...
⏳ Waiting for GPU 0 to write master info...
✓ Found master info:
  Address: 192.168.1.10
  Port: 12355

[GPU 1] ⏳ Waiting for main process to write master IP...
[GPU 1] 📖 Read master IP from file: 192.168.1.10
[GPU 1] 📖 Read master port from file: 12355

🚀 DISTRIBUTED TRAINING - PROCESS 1/1
✓ Process Rank: 1
✓ GPU Device: cuda:1
✓ World Size: 2
✓ Master: 192.168.1.10:12355
✓ Backend: NCCL
✓ Hostname: gpu-node-02

[GPU 1] ⚡ Worker process ready and waiting for main process...
...
```

## How It Works

### 1. Job Submission
```
You run: ./submit_distributed_pbs.sh
    ↓
PBS schedules: GPU 0 job on available node
    ↓
PBS schedules: GPU 1 job on available node
    ↓
Both jobs wait in queue
```

### 2. Job Execution
```
GPU 0 starts first:
  - Detects its IP address
  - Finds available port
  - Writes to /tmp/sd35_master_addr.txt
  - Writes to /tmp/sd35_master_port.txt
  - Waits for GPU 1 to connect

GPU 1 starts:
  - Waits for master info files
  - Reads IP and port
  - Connects to GPU 0

Both connected:
  - Initialize NCCL backend
  - Start distributed training
  - Sync gradients automatically
```

### 3. Training Process
```
Each epoch:
  GPU 0: Processes ~30k samples → computes gradients
  GPU 1: Processes ~30k samples → computes gradients
         ↓
  NCCL: Automatically averages gradients
         ↓
  Both GPUs: Update model with averaged gradients
```

## Troubleshooting

### Problem: "Timeout waiting for master info files"

**Cause**: GPU 1 started before GPU 0 wrote connection info

**Solution**:
```bash
# Check if GPU 0 is running
qstat -u $USER

# If GPU 0 is pending/not running, wait for it to start
# Or cancel and resubmit both jobs
qdel <GPU1_JOB_ID>
./submit_distributed_pbs.sh
```

### Problem: Jobs scheduled on same node

**Cause**: Cluster may assign both jobs to same physical machine

**Result**: This is actually FINE! Both GPUs can still train if they're on the same node.

**Verify**:
```bash
# Check which nodes jobs are on
qstat -f <JOB_ID> | grep exec_host
```

### Problem: "Connection refused" or "Address already in use"

**Cause**: Port conflict

**Solution**: Script auto-detects available ports, but if it fails:
```bash
# Clean up old files
rm -f /tmp/sd35_master_addr.txt /tmp/sd35_master_port.txt

# Resubmit jobs
./submit_distributed_pbs.sh
```

### Problem: One job completes, other still running

**Cause**: One process crashed or training finished

**Solution**:
```bash
# Cancel the running job
qdel <RUNNING_JOB_ID>

# Check logs for errors
cat ~/gan-project/outputs/py_output/gpu0_output.txt
cat ~/gan-project/outputs/py_output/gpu1_output.txt
```

## Customization

### Change Walltime
Edit both PBS scripts:
```bash
#PBS -l walltime=200:00:00  # 200 hours instead of 100
```

### Change Memory
```bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=50GB  # 50GB instead of 40GB
```

### Change Queue
```bash
#PBS -q gpu-long  # Use different queue
```

### Change Number of Epochs
Edit `train_sd35_lora.py`:
```python
NUM_EPOCHS = 10  # Instead of 20
```

## File Locations

### Scripts
```
/home/users/ntu/birul001/gan-project/git/gan-HC-SCDNet/SDFinetune/
├── run_distributed_pbs_gpu0.sh      # GPU 0 PBS script
├── run_distributed_pbs_gpu1.sh      # GPU 1 PBS script
├── submit_distributed_pbs.sh        # Launcher script
└── train_sd35_lora.py               # Training script
```

### Outputs
```
/home/users/ntu/birul001/gan-project/outputs/
├── logs/
│   ├── gpu0_job_output.log          # GPU 0 PBS stdout/stderr
│   └── gpu1_job_output.log          # GPU 1 PBS stdout/stderr
└── py_output/
    ├── gpu0_output.txt               # GPU 0 training output
    └── gpu1_output.txt               # GPU 1 training output
```

### Models
```
/home/msai/birul001/BIRUL001/models/sd35_lora/
├── checkpoint-epoch-5/
├── checkpoint-epoch-10/
├── checkpoint-epoch-15/
├── checkpoint-epoch-20/
└── final/
```

## Quick Commands Reference

```bash
# Submit both jobs
./submit_distributed_pbs.sh

# Check job status
qstat -u $USER

# Monitor training (GPU 0 - main)
tail -f ~/gan-project/outputs/py_output/gpu0_output.txt

# Monitor training (GPU 1 - worker)
tail -f ~/gan-project/outputs/py_output/gpu1_output.txt

# Cancel jobs
qdel <JOB_ID_GPU0> <JOB_ID_GPU1>

# Check PBS logs
ls -lh ~/gan-project/outputs/logs/

# Check training outputs
ls -lh ~/gan-project/outputs/py_output/

# Check saved models
ls -lh /home/msai/birul001/BIRUL001/models/sd35_lora/
```

## Training Timeline

| Time | Event |
|------|-------|
| T+0 min | Submit both PBS jobs |
| T+1-5 min | Jobs move from queue to running state |
| T+5-10 min | GPU 0 loads SD3.5 model (~16GB) |
| T+10-15 min | GPU 1 loads SD3.5 model |
| T+15-20 min | Both GPUs start training epoch 1 |
| T+4-5 hours | Epoch 1 complete |
| T+24-25 hours | Epoch 5 complete, checkpoint saved |
| T+80-100 hours | All 20 epochs complete |

## Expected Results

- **Training speed**: ~4-5 hours per epoch (2 GPUs)
- **Total time**: ~80-100 hours for 20 epochs
- **Checkpoints**: Saved every 5 epochs
- **Final model**: `/home/msai/birul001/BIRUL001/models/sd35_lora/final/`
- **WandB logs**: Only from GPU 0 (avoids duplicates)
- **Memory usage**: ~35-38 GB per GPU

## Support

If you encounter issues:
1. Check PBS logs: `~/gan-project/outputs/logs/`
2. Check training output: `~/gan-project/outputs/py_output/`
3. Verify both jobs are running: `qstat -u $USER`
4. Check GPU usage on compute nodes (if you have access)
5. Look for error messages in output files
