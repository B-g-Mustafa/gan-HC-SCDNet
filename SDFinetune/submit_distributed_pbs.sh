#!/bin/bash

# ==========================================
# Launcher Script for PBS Distributed Training
# Submits both GPU jobs to PBS scheduler
# ==========================================

HOME_PATH="/home/users/ntu/birul001/"
PROJECT_PATH="${HOME_PATH}gan-project/git/gan-HC-SCDNet/SDFinetune"
OUTPUT_PATH="${HOME_PATH}gan-project/outputs"

# Create necessary directories
mkdir -p "${OUTPUT_PATH}/logs"
mkdir -p "${OUTPUT_PATH}/py_output"

echo "=========================================="
echo "PBS Distributed Training Launcher"
echo "=========================================="
echo ""
echo "Submitting 2 PBS jobs for distributed training:"
echo "  - GPU 0 (Main Process)"
echo "  - GPU 1 (Worker Process)"
echo ""
echo "Configuration:"
echo "  - Batch size per GPU: 4"
echo "  - Gradient accumulation: 4"
echo "  - Effective batch per GPU: 16"
echo "  - Total effective batch: 32"
echo "  - Epochs: 20"
echo "  - Dataset: 60k samples"
echo ""

# Submit GPU 0 job (Main Process)
echo "Submitting GPU 0 (Main Process)..."
cd "${PROJECT_PATH}"
JOB0=$(qsub run_distributed_pbs_gpu0.sh)

if [ $? -eq 0 ]; then
    echo "✓ GPU 0 job submitted: $JOB0"
else
    echo "✗ Failed to submit GPU 0 job"
    exit 1
fi

# Wait a bit for GPU 0 to start
echo ""
echo "Waiting 10 seconds before submitting GPU 1..."
sleep 10

# Submit GPU 1 job (Worker Process)
echo ""
echo "Submitting GPU 1 (Worker Process)..."
JOB1=$(qsub run_distributed_pbs_gpu1.sh)

if [ $? -eq 0 ]; then
    echo "✓ GPU 1 job submitted: $JOB1"
else
    echo "✗ Failed to submit GPU 1 job"
    echo "⚠ GPU 0 job ($JOB0) is still running"
    exit 1
fi

echo ""
echo "=========================================="
echo "Both jobs submitted successfully!"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  GPU 0 (Main): $JOB0"
echo "  GPU 1 (Worker): $JOB1"
echo ""
echo "Monitor jobs:"
echo "  qstat -u \$USER"
echo "  qstat -f $JOB0"
echo "  qstat -f $JOB1"
echo ""
echo "Check logs:"
echo "  tail -f ${OUTPUT_PATH}/logs/gpu0_job_output.log"
echo "  tail -f ${OUTPUT_PATH}/logs/gpu1_job_output.log"
echo ""
echo "Check training output:"
echo "  tail -f ${OUTPUT_PATH}/py_output/gpu0_output.txt"
echo "  tail -f ${OUTPUT_PATH}/py_output/gpu1_output.txt"
echo ""
echo "Cancel jobs if needed:"
echo "  qdel $JOB0"
echo "  qdel $JOB1"
echo ""
echo "=========================================="
echo "Training will start automatically when both"
echo "jobs are allocated by PBS scheduler"
echo "=========================================="
