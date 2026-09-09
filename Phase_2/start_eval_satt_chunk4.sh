#!/bin/bash
#SBATCH --job-name=eval_satt4
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yashjadhav23/eval_satt_chunk4_%j.out

cd /home/yashjadhav23/MAJOR_PROJECT/Phase_2
module load python/3.11.14
source /home/yashjadhav23/MAJOR_PROJECT/major_env/bin/activate

echo "=== Eval SATT chunk=4 started at $(date) ==="
nvidia-smi

# Install new metric packages if not already present
uv pip install sacrebleu rouge-score bert-score nltk radgraph --quiet

python main.py \
  --phase 2 \
  --mode eval \
  --model_type satt \
  --chunk_size 4 \
  --data_dir /data/yashjadhav23/merlin/merlin_data \
  --reports_xlsx /data/yashjadhav23/merlin/reports_final.xlsx \
  --checkpoint_dir /home/yashjadhav23/checkpoints \
  --num_slices 64 \
  --num_workers 16 \
  --eval_max_samples 1000

echo "=== Eval SATT chunk=4 finished at $(date) ==="
