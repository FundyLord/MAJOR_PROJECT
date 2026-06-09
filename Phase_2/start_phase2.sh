#!/bin/bash
#SBATCH --job-name=satt_phase2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/phase2_%j.out

mkdir -p logs

module load python/3.11.14
source $HOME/merlin_env/bin/activate

echo "Phase 2 training started at $(date)"
echo "Node: $(hostname)"
nvidia-smi

python main.py \
  --phase        2 \
  --mode         train \
  --data_dir     /data/$USER/merlin/merlin_data \
  --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
  --checkpoint_dir $HOME/checkpoints \
  --resume_from  latest \
  --num_slices   64 \
  --batch_size   1 \
  --grad_accum_steps 8 \
  --num_workers  16 \
  --num_epochs   5 \
  --lr_phase2    2e-5 \
  --log_every    100 \
  --save_every   500

echo "Phase 2 finished at $(date)"
