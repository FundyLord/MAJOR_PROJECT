#!/bin/bash
#SBATCH --job-name=satt_phase2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/yashjadhav23/phase2_%j.out

cd /home/yashjadhav23/MAJOR_PROJECT/Phase_2

module load python/3.11.14
source $HOME/MAJOR_PROJECT/major_env/bin/activate

echo "Phase 2 started at $(date)"
nvidia-smi

python main.py \
  --phase        2 \
  --mode         train \
  --data_dir     /data/yashjadhav23/merlin/merlin_data \
  --reports_xlsx /data/yashjadhav23/merlin/reports_final.xlsx \
  --checkpoint_dir /home/yashjadhav23/checkpoints \
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
