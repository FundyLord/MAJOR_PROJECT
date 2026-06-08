"""
model.py  —  SATT Model Components

Defines:
  SATTAdapter          — the novel temporal compression + projection module
  build_vision_encoder — frozen SigLIP-base-patch16-224
  build_llm_phase1     — Llama-3.2-3B in BF16, fully frozen
  build_llm_phase2     — Llama-3.2-3B in 4-bit NF4 + LoRA adapters
  build_tokenizer      — Llama tokenizer
  encode_volume_slices — batched SigLIP + SATT forward pass
"""

import torch
import torch.nn as nn
from transformers import (
    SiglipVisionModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ── Model IDs ────────────────────────────────────────────────────────────────
SIGLIP_ID   = "google/siglip-base-patch16-224"
LLAMA_ID    = "meta-llama/Llama-3.2-3B-Instruct"

# ── Dims (do not change — tied to model architecture) ────────────────────────
VISION_DIM  = 768    # SigLIP-base hidden size
LLM_DIM     = 3072   # Llama-3.2-3B hidden size
NUM_PATCHES = 196    # (224 / 16)^2 patches per slice


# ── SATT Adapter ─────────────────────────────────────────────────────────────

class SATTAdapter(nn.Module):
    """
    Slice-Aware Temporal Transformer Adapter.

    Compresses Z SigLIP embeddings into a compact token sequence
    suitable for Llama's embedding space.

    Input  shape : (Z, N, vision_dim)   e.g. (64, 196, 768)
    Output shape : (1, T*N, llm_dim)    e.g. (1, 3136, 3072)

    where T = Z // chunk_size  (default 64 // 4 = 16)
    """

    def __init__(
        self,
        vision_dim: int = VISION_DIM,
        llm_dim:    int = LLM_DIM,
        chunk_size: int = 4,
        num_heads:  int = 8,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (Z, N, D)  — num_slices × num_patches × vision_dim
        Returns:
            (1, T*N, llm_dim)
        """
        Z, N, D = x.shape
        T = Z // self.chunk_size

        # 1. Temporal grouping + mean pool  →  (T, N, D)
        compressed = x.view(T, self.chunk_size, N, D).mean(dim=1)

        # 2. Flatten to sequence  →  (1, T*N, D)
        sequence = compressed.view(1, T * N, D)

        # 3. Temporal Transformer
        contextualized = self.temporal_transformer(sequence)

        # 4. MLP projection  →  (1, T*N, llm_dim)
        return self.projector(contextualized)


# ── Builder helpers ──────────────────────────────────────────────────────────

def build_vision_encoder() -> SiglipVisionModel:
    """Load SigLIP with all parameters frozen (never trained)."""
    model = SiglipVisionModel.from_pretrained(SIGLIP_ID)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def build_llm_phase1() -> AutoModelForCausalLM:
    """
    Phase 1: Llama in BF16, fully frozen.
    On A6000 (48 GB) we can afford BF16 — no quantisation needed.
    """
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_llm_phase2() -> AutoModelForCausalLM:
    """
    Phase 2: Llama in 4-bit NF4 + LoRA on all 7 projection layers.
    Matches the validated notebook configuration.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(LLAMA_ID)
    tok.pad_token = tok.eos_token
    return tok


# ── Volume encoding ──────────────────────────────────────────────────────────

def encode_volume_slices(
    vision_encoder: nn.Module,
    satt_adapter:   SATTAdapter,
    slices:         torch.Tensor,
    micro_batch:    int = 8,
) -> torch.Tensor:
    """
    Encode a batch of CT volumes through SigLIP + SATTAdapter.

    Args:
        vision_encoder : frozen SigLIP (or a mock in tests)
        satt_adapter   : SATTAdapter module
        slices         : (B, Z, 3, H, W)
        micro_batch    : number of slices per SigLIP forward (VRAM control)

    Returns:
        visual_tokens  : (B, T*N, llm_dim)
    """
    B, Z, C, H, W = slices.shape
    device = next(satt_adapter.parameters()).device
    tokens_list = []

    for b in range(B):
        sample = slices[b].to(device)   # (Z, 3, H, W)

        # SigLIP in micro-batches to avoid OOM
        embeddings = []
        for i in range(0, Z, micro_batch):
            chunk = sample[i : i + micro_batch]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = vision_encoder(pixel_values=chunk)
            embeddings.append(out.last_hidden_state.float())   # (mb, 196, 768)

        E      = torch.cat(embeddings, dim=0)   # (Z, 196, 768)
        tokens = satt_adapter(E)                # (1, T*N, llm_dim)
        tokens_list.append(tokens)

    return torch.cat(tokens_list, dim=0)        # (B, T*N, llm_dim)
