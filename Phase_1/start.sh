#!/bin/bash 
#SBATCH --job-name=satt_stage1 
#SBATCH --partition=general 
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=32 
#SBATCH --mem=120G 
#SBATCH --time=48:00:00 
LOGFILE="run.log"

### Redirect EVERYTHING into run.log
exec > >(tee -a $LOGFILE) 2>&1
echo "====================================" 
echo "JOB START" 
date 
echo "===================================="

echo "Loading Python module" 
module load python/3.11.14

echo "Creating virtual environment" 
uv venv env

echo "Activating environment" 
source env/bin/activate

echo "Installing packages" 
uv pip install torch>=2.0.0 torchvision transformers>=4.35.0 nibabel>=5.0.0 scipy>=1.10.0 numpy>=1.23.0 huggingface_hub>=0.19.0 pandas>=2.0.0 matplotlib>=3.7.0 datasets>=3.2.0 accelerate deepspeed peft>=0.7.0 bitsandbytes>=0.41.0 tqdm

echo "Starting training" 
python -u main.py

echo "====================================" 
echo "JOB FINISHED" 
date 
echo "===================================="