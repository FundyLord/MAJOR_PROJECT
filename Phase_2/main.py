"""
main.py  —  Entry point for SATT Medical VLM (v2)

Supports:
  --model_type satt      : SATT adapter with temporal chunking (default)
  --model_type baseline  : Simple mean-pool baseline for comparison
  --chunk_size 2/4/8     : SATT temporal chunk size (ablation study)

Usage examples
--------------
# SATT Phase 1 (default chunk=4):
python main.py --phase 1 --mode train --model_type satt --chunk_size 4 \
    --data_dir /data/$USER/merlin/merlin_data \
    --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
    --checkpoint_dir ./checkpoints/satt_chunk4

# Mean-pool baseline Phase 1:
python main.py --phase 1 --mode train --model_type baseline \
    --data_dir /data/$USER/merlin/merlin_data \
    --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
    --checkpoint_dir ./checkpoints/baseline

# SATT chunk_size=2 ablation:
python main.py --phase 1 --mode train --model_type satt --chunk_size 2 \
    --checkpoint_dir ./checkpoints/satt_chunk2 ...

# Evaluation:
python main.py --phase 2 --mode eval --model_type satt --chunk_size 4 \
    --checkpoint_dir ./checkpoints/satt_chunk4 \
    --eval_max_samples 1000
"""

import argparse
import logging
import os


def setup_logging(log_path: str = "run.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )


def parse_args():
    p = argparse.ArgumentParser(description="SATT Medical VLM")

    # Required
    p.add_argument("--phase", type=int, required=True, choices=[1, 2])
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--reports_xlsx", type=str, required=True)

    # Model type — controls which adapter is used
    p.add_argument("--model_type", type=str, default="satt",
                   choices=["satt", "baseline"],
                   help="'satt' = SATT temporal adapter | 'baseline' = mean-pool")
    p.add_argument("--chunk_size", type=int, default=4,
                   choices=[2, 4, 8],
                   help="SATT temporal chunk size (ignored for baseline)")

    # Mode
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "eval"])

    # Paths
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume_from", type=str, default="latest")

    # Data
    p.add_argument("--num_slices",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)

    # Training
    p.add_argument("--num_epochs",       type=int,   default=3)
    p.add_argument("--batch_size",       type=int,   default=2)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lr_phase2",        type=float, default=2e-5)
    p.add_argument("--grad_accum_steps", type=int,   default=4)
    p.add_argument("--max_text_len",     type=int,   default=512)

    # Logging and saving
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--save_every", type=int, default=500)

    # Evaluation
    p.add_argument("--eval_max_samples", type=int, default=1000,
                   help="Max test samples for evaluation. 0 = all 5123.")

    return p.parse_args()


def build_adapter(args):
    """Build the correct adapter based on --model_type and --chunk_size."""
    from model import SATTAdapter, MeanPoolAdapter
    if args.model_type == "baseline":
        logging.info("[Model] Using MeanPoolAdapter (baseline)")
        return MeanPoolAdapter()
    else:
        logging.info(f"[Model] Using SATTAdapter (chunk_size={args.chunk_size})")
        return SATTAdapter(chunk_size=args.chunk_size)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logging(os.path.join(args.checkpoint_dir, "run.log"))

    logging.info(f"Mode={args.mode}  Phase={args.phase}  "
                 f"ModelType={args.model_type}  ChunkSize={args.chunk_size}")
    logging.info(f"data_dir={args.data_dir}")
    logging.info(f"checkpoint_dir={args.checkpoint_dir}")
    logging.info(f"batch_size={args.batch_size}  accum={args.grad_accum_steps}  "
                 f"effective_batch={args.batch_size * args.grad_accum_steps}")

    if args.mode == "train":
        from train import train_phase1, train_phase2
        if args.phase == 1:
            train_phase1(args)
        else:
            train_phase2(args)
    elif args.mode == "eval":
        from evaluate import run_evaluation
        run_evaluation(args)


if __name__ == "__main__":
    main()