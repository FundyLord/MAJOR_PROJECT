#!/bin/bash
#SBATCH --job-name=decode_pilot
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/yashjadhav23/decode_pilot_%j.out

# Decoding pilot: the SATT c=4 model (the one variant whose evaluation was
# valid) on 100 VALIDATION scans, under four decoding settings. Decoding is
# chosen on validation data so the test set is only used once, afterwards.
# Expected: about 45-50 minutes per setting, 3-3.5 hours in total.

cd /home/yashjadhav23/MAJOR_PROJECT/Phase_2
module load python/3.11.14
source /home/yashjadhav23/MAJOR_PROJECT/major_env/bin/activate
nvidia-smi

run () {   # run <name> <repetition_penalty> <no_repeat_ngram_size>
  echo "=== Pilot $1 (repetition_penalty=$2, no_repeat_ngram_size=$3) started at $(date) ==="
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
    --eval_split val \
    --eval_max_samples 100 \
    --repetition_penalty "$2" \
    --no_repeat_ngram_size "$3" \
    --eval_out_dir "/home/yashjadhav23/checkpoints/pilot_$1"
  echo "=== Pilot $1 finished at $(date) (exit code $?) ==="
}

run A_current   1.3 4     # the setting used for every result so far
run B_off       1.0 0     # plain greedy decoding, no repetition control
run C_mild      1.1 0     # mild penalty only
run D_loopguard 1.0 12    # no penalty; only blocks repeats of 12+ tokens

echo "=== Decoding pilot finished at $(date) ==="
