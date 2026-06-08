"""
train.py  —  Phase 1 and Phase 2 training loops

Phase 1 : Only SATTAdapter is trainable. SigLIP and Llama are frozen.
          Loss = cross-entropy on report tokens (visual tokens masked with -100).

Phase 2 : SATTAdapter + LoRA adapters inside Llama are trainable.
          Loads SATT weights from the best Phase 1 checkpoint automatically.
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MerlinCTDataset, merlin_collate_fn
from model import (
    SATTAdapter,
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
    """Return (prompt_text, target_text) for causal-LM training."""
    return SYSTEM_PROMPT, findings + "<|eot_id|>"


# ── Tokenisation ─────────────────────────────────────────────────────────────

def tokenize_batch(
    tokenizer,
    findings_list: list,
    device,
    max_length: int = 512,
):
    """
    Tokenise a list of findings strings.

    Returns:
        input_ids      : (B, L)  — full sequence (prompt + findings)
        attention_mask : (B, L)
        labels         : (B, L)  — prompt tokens masked with -100
    """
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
        labels[: len(prompt_ids)] = -100   # mask prompt — only predict findings

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    # Pad to same length
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

def save_checkpoint(satt, optimizer, step, epoch, loss, ckpt_dir, phase):
    """Save SATT weights + optimizer state. Write a 'latest' pointer file."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"phase{phase}_step{step:07d}.pt")
    torch.save(
        {
            "step":           step,
            "epoch":          epoch,
            "loss":           loss,
            "satt_state":     satt.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    # Overwrite latest pointer
    pointer = os.path.join(ckpt_dir, f"phase{phase}_latest.txt")
    with open(pointer, "w") as f:
        f.write(ckpt_path)
    logging.info(f"[Checkpoint] saved  step={step}  →  {ckpt_path}")


def load_checkpoint(ckpt_dir, phase, satt, optimizer=None):
    """Load from latest checkpoint if it exists. Returns (start_step, start_epoch)."""
    pointer = os.path.join(ckpt_dir, f"phase{phase}_latest.txt")
    if not os.path.exists(pointer):
        logging.info("[Checkpoint] No existing checkpoint — starting from scratch.")
        return 0, 0

    with open(pointer) as f:
        ckpt_path = f.read().strip()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    satt.load_state_dict(ckpt["satt_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    logging.info(f"[Checkpoint] Resumed  step={ckpt['step']}  epoch={ckpt['epoch']}  from {ckpt_path}")
    return ckpt["step"], ckpt["epoch"]


# ── Shared forward pass ───────────────────────────────────────────────────────

def forward_pass(slices, findings, vision_encoder, satt, llm, tokenizer, llm_device, args):
    """
    Shared forward pass for both phases.
    Returns the scalar loss.
    """
    # 1. Vision encoding
    visual_tokens = encode_volume_slices(
        vision_encoder, satt, slices, micro_batch=8
    )                                                   # (B, T*N, llm_dim)
    visual_tokens = visual_tokens.to(llm_device)

    # 2. Tokenise findings
    input_ids, attn_mask, labels = tokenize_batch(
        tokenizer, findings, llm_device, args.max_text_len
    )

    # 3. Text embeddings
    text_embeds = llm.get_input_embeddings()(input_ids)  # (B, L, llm_dim)
    vis_tokens  = visual_tokens.to(text_embeds.dtype)

    # 4. Concatenate visual + text tokens
    inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)

    vis_mask   = torch.ones(
        vis_tokens.shape[0], vis_tokens.shape[1],
        device=llm_device, dtype=attn_mask.dtype
    )
    full_mask  = torch.cat([vis_mask, attn_mask], dim=1)

    vis_labels = torch.full(
        (vis_tokens.shape[0], vis_tokens.shape[1]),
        -100, device=llm_device, dtype=labels.dtype
    )
    full_labels = torch.cat([vis_labels, labels], dim=1)

    # 5. LLM forward
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
    return out.loss


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def train_phase1(args):
    logging.info("=" * 60)
    logging.info("PHASE 1 — SATT Alignment Training")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx,
        split="train", num_slices=args.num_slices,
    )
    val_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx,
        split="val", num_slices=args.num_slices,
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

    # Models
    vision_encoder = build_vision_encoder().to(device)
    satt           = SATTAdapter().to(device)
    llm            = build_llm_phase1()
    tokenizer      = build_tokenizer()
    llm_device     = next(llm.parameters()).device

    # Only SATT is trainable in Phase 1
    optimizer = torch.optim.AdamW(satt.parameters(), lr=args.lr, weight_decay=0.01)

    # Resume
    start_step, start_epoch = 0, 0
    if args.resume_from == "latest":
        start_step, start_epoch = load_checkpoint(
            args.checkpoint_dir, phase=1, satt=satt, optimizer=optimizer
        )

    global_step = start_step
    accum       = args.grad_accum_steps

    for epoch in range(start_epoch, args.num_epochs):
        satt.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            loss = forward_pass(
                batch["slices"], batch["findings"],
                vision_encoder, satt, llm, tokenizer, llm_device, args,
            )
            (loss / accum).backward()
            running_loss += loss.item()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(satt.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = running_loss / args.log_every
                    logging.info(
                        f"Epoch {epoch}  step {global_step}  "
                        f"train_loss={avg:.4f}"
                    )
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        satt, optimizer, global_step, epoch,
                        loss.item(), args.checkpoint_dir, phase=1,
                    )

        # Validation
        satt.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                vl = forward_pass(
                    vbatch["slices"], vbatch["findings"],
                    vision_encoder, satt, llm, tokenizer, llm_device, args,
                )
                val_loss += vl.item()
        val_loss /= max(len(val_loader), 1)
        logging.info(f"Epoch {epoch} complete  val_loss={val_loss:.4f}")

    # Final save
    save_checkpoint(
        satt, optimizer, global_step, epoch,
        val_loss, args.checkpoint_dir, phase=1,
    )
    logging.info("Phase 1 complete.")


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def train_phase2(args):
    logging.info("=" * 60)
    logging.info("PHASE 2 — QLoRA Fine-Tuning")
    logging.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx,
        split="train", num_slices=args.num_slices,
    )
    val_ds = MerlinCTDataset(
        args.data_dir, args.reports_xlsx,
        split="val", num_slices=args.num_slices,
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

    # Models
    vision_encoder = build_vision_encoder().to(device)
    satt           = SATTAdapter().to(device)
    llm            = build_llm_phase2()
    tokenizer      = build_tokenizer()
    llm_device     = next(llm.parameters()).device

    # Load Phase 1 SATT weights
    p1_pointer = os.path.join(args.checkpoint_dir, "phase1_latest.txt")
    if os.path.exists(p1_pointer):
        with open(p1_pointer) as f:
            p1_path = f.read().strip()
        ckpt = torch.load(p1_path, map_location="cpu")
        satt.load_state_dict(ckpt["satt_state"])
        logging.info(f"[Phase 2] Loaded Phase 1 SATT from {p1_path}")
    else:
        logging.warning("[Phase 2] No Phase 1 checkpoint found — SATT starts from random init.")

    # Trainable: SATT + LoRA params
    trainable = list(satt.parameters()) + [p for p in llm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr_phase2, weight_decay=0.01)

    start_step, start_epoch = 0, 0
    if args.resume_from == "latest":
        start_step, start_epoch = load_checkpoint(
            args.checkpoint_dir, phase=2, satt=satt, optimizer=optimizer
        )

    global_step = start_step
    accum       = args.grad_accum_steps

    for epoch in range(start_epoch, args.num_epochs):
        satt.train()
        llm.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            loss = forward_pass(
                batch["slices"], batch["findings"],
                vision_encoder, satt, llm, tokenizer, llm_device, args,
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
                        f"Epoch {epoch}  step {global_step}  "
                        f"train_loss={avg:.4f}"
                    )
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    save_checkpoint(
                        satt, optimizer, global_step, epoch,
                        loss.item(), args.checkpoint_dir, phase=2,
                    )

        # Validation
        satt.eval()
        llm.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vbatch in val_loader:
                vl = forward_pass(
                    vbatch["slices"], vbatch["findings"],
                    vision_encoder, satt, llm, tokenizer, llm_device, args,
                )
                val_loss += vl.item()
        val_loss /= max(len(val_loader), 1)
        logging.info(f"Epoch {epoch} complete  val_loss={val_loss:.4f}")

    save_checkpoint(
        satt, optimizer, global_step, epoch,
        val_loss, args.checkpoint_dir, phase=2,
    )
    logging.info("Phase 2 complete.")
