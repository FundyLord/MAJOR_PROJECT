#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yashjadhav23/%x_%j.out

# Corrected evaluation ("eval v2") of ONE variant on the same 1,000 test scans
# (seed 42), with the decoding chosen on validation scans in the pilot:
# repetition_penalty 1.1, no_repeat_ngram_size 0.
# Usage: sbatch --job-name=v2_c4 start_eval_v2.sh satt 4

MODEL_TYPE="$1"
CHUNK="$2"
if [ -z "$MODEL_TYPE" ] || [ -z "$CHUNK" ]; then
  echo "usage: sbatch --job-name=<name> start_eval_v2.sh <baseline|satt> <chunk_size>"
  exit 1
fi

cd /home/yashjadhav23/MAJOR_PROJECT/Phase_2
module load python/3.11.14
source /home/yashjadhav23/MAJOR_PROJECT/major_env/bin/activate

echo "=== eval v2: model_type=$MODEL_TYPE chunk_size=$CHUNK started at $(date) ==="
nvidia-smi

python main.py \
  --phase 2 \
  --mode eval \
  --model_type "$MODEL_TYPE" \
  --chunk_size "$CHUNK" \
  --data_dir /data/yashjadhav23/merlin/merlin_data \
  --reports_xlsx /data/yashjadhav23/merlin/reports_final.xlsx \
  --checkpoint_dir /home/yashjadhav23/checkpoints \
  --num_slices 64 \
  --num_workers 16 \
  --eval_split test \
  --eval_max_samples 1000 \
  --repetition_penalty 1.1 \
  --no_repeat_ngram_size 0 \
  --eval_out_dir /home/yashjadhav23/checkpoints/eval_v2
rc=$?

echo "=== eval v2: model_type=$MODEL_TYPE chunk_size=$CHUNK finished at $(date), python exit code $rc ==="
exit $rc
