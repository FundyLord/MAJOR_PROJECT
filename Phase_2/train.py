"""
train.py  —  Phase 1 and Phase 2 training loops (v2)

Supports --model_type satt | baseline and --chunk_size 2|4|8
via args passed from main.py build_adapter().
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MerlinCTDataset, merlin_collate_fn
from model import (
    build_vision_encoder,
    build_llm_phase1,
    build_llm_phase2,
    build_tokenizer,
    encode_volume_slices,
)


# ── Prompt helpers ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Analyze this abdominal CT scan and generate a clinical radiology report.\n"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


def build_prompt(findings: str) -> tuple:
    return SYSTEM_PROMPT, findings + "<|eot_id|>"


# ── Tokenisation ─────────────────────────────────────────────────────────────

def tokenize_batch(tokenizer, findings_list, device, max_length=512):
    all_input_ids, all_labels = [], []

    for findings in findings_list:
        prompt, target = build_prompt(findings)
        full_text = prompt + target

        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        full_ids = tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        ).input_ids[0]

        labels = full_ids.clone()
        labels[: len(prompt_ids)] = -100

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    max_len = max(t.shape[0] for t in all_input_ids)

    def pad(seq, pad_val):
        out = torch.full((max_len,), pad_val, dtype=torch.long)
        out[: seq.shape[0]] = seq
        return out

    input_ids = torch.stack(
        [pad(t, tokenizer.pad_token_id) for t in all_input_ids]
    ).to(device)
    labels = torch.stack(
        [pad(t, -100) for t in all_labels]
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return input_ids, attention_mask, labels


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _ckpt_prefix(args):
    """Generate checkpoint filename prefix based on model type."""
    if args.model_type == "baseline":
        return "baseline"
    return f"satt_chunk{args.chunk_size}"


def save_checkpoint(adapter, optimizer, step, epoch, loss, ckpt_dir, phase, args):
    os.makedirs(ckpt_dir, exist_ok=True)
    prefix    = _ckpt_prefix(args)
    ckpt_path = os.path.join(
        ckpt_dir, f"phase{phase}_{prefix}_step{step:07d}.pt"
    )
    torch.save(
        {
            "step":            step,
            "epoch":           epoch,
            "loss":            loss,
            "model_type":      args.model_type,
            "chunk_size":      getattr(args, "chunk_size", None),
            "satt_state":      adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    pointer = os.path.join(ckpt_dir, f"phase{phase}_{prefix}_latest.txt")
    with open(pointer, "w") as f:
        f.write(ckpt_path)
    logging.info(f"[Checkpoint] saved step={step}  →  {ckpt_path}")


def load_checkpoint(ckpt_dir, phase, adapter, args, optimizer=None):
    prefix  = _ckpt_prefix(args)
    pointer = os.path.join(ckpt_dir, f"phase{phase}_{prefix}_latest.txt")
    if not os.path.exists(pointer):
        # Fallback: try old-style pointer (for SATT chunk=4 trained previously)
        pointer = os.path.join(ckpt_dir, f"phase{phase}_latest.txt")
    if not os.path.exists(pointer):
        logging.info("[Checkpoint] No existing checkpoint — starting from scratch.")
        return 0, 0

    with open(pointer) as f:
        ckpt_path = f.read().strip()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(ckpt["satt_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logging.info(f"[Checkpoint] Resumed step={ckpt['step']} epoch={ckpt['epoch']}")
    return ckpt["step"], ckpt["epoch"]


# ── Shared forward pass ───────────────────────────────────────────────────────

def forward_pass(slices, findings, vision_encoder, adapter, llm,
                 tokenizer, llm_device, args):
    visual_tokens = encode_volume_slices(
        vision_encoder, adapter, slices, micro_batch=8
    )
    visual_tokens = visual_tokens.to(llm_device)

    input_ids, attn_mask, labels = tokenize_batch(
        tokenizer, findings, llm_device, args.max_text_len
    )
    text_embeds   = llm.get_input_embeddings()(input_ids)
    vis_tokens    = visual_tokens.to(text_embeds.dtype)
    inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)

    vis_mask    = torch.ones(
        vis_tokens.shape[0], vis_tokens.shape[1],
        device=llm_device, dtype=attn_mask.dtype
    )
    full_mask   = torch.cat([vis_mask, attn_mask], dim=1)
    vis_labels  = torch.full(
        (vis_tokens.shape[0], vis_tokens.shape[1]),
        -100, device=llm_device, dtype=labels.dtype
    )
    full_labels = torch.cat([vis_labels, labels], dim=1)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
    return out.loss


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def train_phase1(args):
    from main import build_adapter

    logging.info("=" * 60)
    logging.info(f"PHASE 1 — Alignment Training "
                 f"[{args.model_type} chunk={getattr(args,'chunk_size','N/A')}]")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx, split="train",
        num_slices=args.num_slices,
    )
    val_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx, split="val",
        num_slices=args.num_slices,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=merlin_collate_fn,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=merlin_collate_fn,
        pin_memory=True,
    )

    vision_encoder = build_vision_encoder().to(device)
    adapter        = build_adapter(args).to(device)
    llm            = build_llm_phase1()
    tokenizer      = build_tokenizer()
    llm_device     = next(llm.parameters()).device

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=0.01
    )

    start_step, start_epoch = 0, 0
    if args.resume_from == "latest":
        start_step, start_epoch = load_checkpoint(
            args.checkpoint_dir, phase=1, adapter=adapter,
            args=args, optimizer=optimizer
        )

    global_step  = start_step
    accum        = args.grad_accum_steps
    best_val     = float("inf")

    for epoch in range(start_epoch, args.num_epochs):
        adapter.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            loss = forward_pass(
                batch["slices"], batch["findings"],
                vision_encoder, adapter, llm, tokenizer, llm_device, args,
            )
            (loss / accum).backward()
            running_loss += loss.item()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running_loss / args.log_every
                    logging.info(
                        f"Epoch {epoch}  step {global_step}  train_loss={avg:.4f}"
                    )
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        adapter, optimizer, global_step, epoch,
                        loss.item(), args.checkpoint_dir, phase=1, args=args,
                    )

        # Validation
        adapter.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vb in val_loader:
                vl = forward_pass(
                    vb["slices"], vb["findings"],
                    vision_encoder, adapter, llm, tokenizer, llm_device, args,
                )
                val_loss += vl.item()
        val_loss /= max(len(val_loader), 1)
        logging.info(f"Epoch {epoch} complete  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            prefix   = _ckpt_prefix(args)
            best_path = os.path.join(
                args.checkpoint_dir, f"phase1_best_{prefix}.pt"
            )
            torch.save({
                "step":       global_step,
                "epoch":      epoch,
                "loss":       val_loss,
                "model_type": args.model_type,
                "chunk_size": getattr(args, "chunk_size", None),
                "satt_state": adapter.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, best_path)
            logging.info(f"[Best] New best val_loss={val_loss:.4f} → {best_path}")
        else:
            logging.info(
                f"[Best] val_loss={val_loss:.4f} did not improve from {best_val:.4f}"
            )

    save_checkpoint(
        adapter, optimizer, global_step, epoch,
        val_loss, args.checkpoint_dir, phase=1, args=args,
    )
    logging.info("Phase 1 complete.")


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def train_phase2(args):
    from main import build_adapter

    logging.info("=" * 60)
    logging.info(f"PHASE 2 — QLoRA Fine-Tuning "
                 f"[{args.model_type} chunk={getattr(args,'chunk_size','N/A')}]")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx, split="train",
        num_slices=args.num_slices,
    )
    val_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx, split="val",
        num_slices=args.num_slices,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=merlin_collate_fn,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=merlin_collate_fn,
        pin_memory=True,
    )

    vision_encoder = build_vision_encoder().to(device)
    adapter        = build_adapter(args).to(device)
    llm            = build_llm_phase2()
    tokenizer      = build_tokenizer()
    llm_device     = next(llm.parameters()).device

    # Load Phase 1 best weights — try variant-specific first, then generic
    prefix = _ckpt_prefix(args)
    p1_best = os.path.join(args.checkpoint_dir, f"phase1_best_{prefix}.pt")
    if not os.path.exists(p1_best):
        p1_best = os.path.join(args.checkpoint_dir, "phase1_best.pt")
    if os.path.exists(p1_best):
        ckpt = torch.load(p1_best, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ckpt["satt_state"])
        logging.info(f"[Phase 2] Loaded Phase 1 best from {p1_best} "
                     f"(val_loss={ckpt['loss']:.4f})")
    else:
        logging.warning("[Phase 2] No Phase 1 checkpoint found — random init.")

    trainable = list(adapter.parameters()) + \
                [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr_phase2, weight_decay=0.01)

    start_step, start_epoch = 0, 0
    if args.resume_from == "latest":
        start_step, start_epoch = load_checkpoint(
            args.checkpoint_dir, phase=2, adapter=adapter,
            args=args, optimizer=optimizer
        )

    global_step  = start_step
    accum        = args.grad_accum_steps
    best_val     = float("inf")

    for epoch in range(start_epoch, args.num_epochs):
        adapter.train()
        llm.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            loss = forward_pass(
                batch["slices"], batch["findings"],
                vision_encoder, adapter, llm, tokenizer, llm_device, args,
            )
            (loss / accum).backward()
            running_loss += loss.item()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running_loss / args.log_every
                    logging.info(
                        f"Epoch {epoch}  step {global_step}  train_loss={avg:.4f}"
                    )
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        adapter, optimizer, global_step, epoch,
                        loss.item(), args.checkpoint_dir, phase=2, args=args,
                    )

        # Validation
        adapter.eval()
        llm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vb in val_loader:
                vl = forward_pass(
                    vb["slices"], vb["findings"],
                    vision_encoder, adapter, llm, tokenizer, llm_device, args,
                )
                val_loss += vl.item()
        val_loss /= max(len(val_loader), 1)
        logging.info(f"Epoch {epoch} complete  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            # Save adapter weights
            best_satt_path = os.path.join(
                args.checkpoint_dir, f"phase2_best_satt_{prefix}.pt"
            )
            torch.save({
                "step":       global_step,
                "epoch":      epoch,
                "loss":       val_loss,
                "model_type": args.model_type,
                "chunk_size": getattr(args, "chunk_size", None),
                "satt_state": adapter.state_dict(),
            }, best_satt_path)
            # Save LoRA adapter
            best_lora_dir = os.path.join(
                args.checkpoint_dir, f"phase2_best_lora_{prefix}"
            )
            os.makedirs(best_lora_dir, exist_ok=True)
            llm.save_pretrained(best_lora_dir)
            logging.info(
                f"[Best] New best val_loss={val_loss:.4f}  "
                f"SATT→{best_satt_path}  LoRA→{best_lora_dir}"
            )
        else:
            logging.info(
                f"[Best] val_loss={val_loss:.4f} did not improve from {best_val:.4f}"
            )

    save_checkpoint(
        adapter, optimizer, global_step, epoch,
        val_loss, args.checkpoint_dir, phase=2, args=args,
    )
    logging.info("Phase 2 complete.")