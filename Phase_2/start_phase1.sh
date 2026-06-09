#!/bin/bash
#SBATCH --job-name=satt_phase1
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/phase1_%j.out

mkdir -p logs

module load python/3.11.14
source $HOME/merlin_env/bin/activate

echo "Phase 1 training started at $(date)"
echo "Node: $(hostname)"
nvidia-smi

python main.py \
  --phase        1 \
  --mode         train \
  --data_dir     /data/$USER/merlin/merlin_data \
  --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
  --checkpoint_dir $HOME/checkpoints \
  --resume_from  latest \
  --num_slices   64 \
  --batch_size   2 \
  --grad_accum_steps 4 \
  --num_workers  16 \
  --num_epochs   10 \
  --lr           1e-4 \
  --log_every    100 \
  --save_every   500

echo "Phase 1 finished at $(date)"
