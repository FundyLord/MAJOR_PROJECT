#!/bin/bash
#SBATCH --job-name=merlin_preprocess
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/preprocess_%j.out

mkdir -p logs

module load python/3.11.14
source $HOME/merlin_env/bin/activate

echo "Starting preprocessing at $(date)"

python preprocess_dataset.py \
  --src_dir      /data/$USER/merlin/merlin_data \
  --dst_dir      /data/$USER/merlin/merlin_data_npy \
  --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
  --num_workers  32

echo "Preprocessing finished at $(date)"
