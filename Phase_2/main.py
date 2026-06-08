"""
main.py  —  Entry point for SATT Medical VLM

Usage examples
--------------
# Phase 1 (SATT alignment):
python main.py --phase 1 --mode train \
    --data_dir /data/$USER/merlin/merlin_data \
    --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
    --checkpoint_dir ./checkpoints \
    --batch_size 2 --num_workers 8

# Phase 2 (QLoRA fine-tuning):
python main.py --phase 2 --mode train \
    --data_dir /data/$USER/merlin/merlin_data \
    --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
    --checkpoint_dir ./checkpoints \
    --batch_size 1 --num_workers 8 --lr_phase2 2e-5

# Evaluation:
python main.py --phase 2 --mode eval \
    --data_dir /data/$USER/merlin/merlin_data \
    --reports_xlsx /data/$USER/merlin/reports_final.xlsx \
    --checkpoint_dir ./checkpoints
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
    p.add_argument("--phase", type=int, required=True, choices=[1, 2],
                   help="1=SATT alignment, 2=QLoRA fine-tuning")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to flat merlin_data/ folder")
    p.add_argument("--reports_xlsx", type=str, required=True,
                   help="Path to reports_final.xlsx")

    # Mode
    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "eval"])

    # Paths
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume_from", type=str, default="latest",
                   help="'latest' to auto-resume, 'none' to start fresh")

    # Data
    p.add_argument("--num_slices",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)

    # Training
    p.add_argument("--num_epochs",       type=int,   default=10)
    p.add_argument("--batch_size",       type=int,   default=2)
    p.add_argument("--lr",               type=float, default=1e-4,
                   help="Phase 1 learning rate")
    p.add_argument("--lr_phase2",        type=float, default=2e-5,
                   help="Phase 2 learning rate")
    p.add_argument("--grad_accum_steps", type=int,   default=4,
                   help="Gradient accumulation steps (effective_batch = batch_size * accum)")
    p.add_argument("--max_text_len",     type=int,   default=512,
                   help="Max tokenised findings length")

    # Logging and saving
    p.add_argument("--log_every",  type=int, default=100,
                   help="Log loss every N gradient steps")
    p.add_argument("--save_every", type=int, default=500,
                   help="Save checkpoint every N gradient steps")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    setup_logging(os.path.join(args.checkpoint_dir, "run.log"))

    logging.info(f"Mode={args.mode}  Phase={args.phase}")
    logging.info(f"data_dir={args.data_dir}")
    logging.info(f"reports_xlsx={args.reports_xlsx}")
    logging.info(f"checkpoint_dir={args.checkpoint_dir}")
    logging.info(
        f"batch_size={args.batch_size}  accum={args.grad_accum_steps}  "
        f"effective_batch={args.batch_size * args.grad_accum_steps}"
    )

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
