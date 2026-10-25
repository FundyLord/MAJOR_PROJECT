"""
param_count.py — exact parameter counts for every part of the SATT system.

Answers Prof. Ghane's review point 2 ("clarify trainable parameters; state
what 0.75% represents"). Nothing is trained or loaded onto the GPU: the
vision encoder and Llama are built from their configs on the "meta" device
(shapes only, no weights), so this runs on a login node in a few seconds.
The configs come from the Hugging Face cache already on the cluster.

Run from Phase_2/ in major_env:
    python param_count.py
"""

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, SiglipVisionConfig, SiglipVisionModel

from model import LLAMA_ID, SIGLIP_ID, ChunkPoolAdapter, MeanPoolAdapter, SATTAdapter


def n(module):
    return sum(p.numel() for p in module.parameters())


def m(x):
    return f"{x:>15,d}  ({x / 1e6:8.2f} M)"


# ---- vision encoder (frozen) -------------------------------------------------
with init_empty_weights():
    vision = SiglipVisionModel(SiglipVisionConfig.from_pretrained(SIGLIP_ID))
vision_total = n(vision)
# The pooling head's location differs between transformers versions
# (vision.vision_model.head in some, vision.head in others), so find it by name.
heads = [mod for name, mod in vision.named_modules() if name.split(".")[-1] == "head"]
vision_head = n(heads[0]) if heads else 0

# ---- language model (frozen base) + LoRA (trainable) ---------------------------
with init_empty_weights():
    llm_cfg = AutoConfig.from_pretrained(LLAMA_ID)
    llm = AutoModelForCausalLM.from_config(llm_cfg)
    # Llama 3.2-3B ties the output layer to the input embedding (one shared
    # 128,256 x 3,072 matrix). On the meta device the tie is not always applied,
    # which would count that matrix twice (+394M), so tie and check explicitly.
    llm.tie_weights()
    llm_base = n(llm)
    if (getattr(llm_cfg, "tie_word_embeddings", False)
            and llm.lm_head.weight is not llm.get_input_embeddings().weight):
        llm_base -= llm.lm_head.weight.numel()
    print(f"[check] tie_word_embeddings = {getattr(llm_cfg, 'tie_word_embeddings', None)}")
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(llm, lora_cfg)
lora = sum(p.numel() for name, p in peft_model.named_parameters() if "lora_" in name)

# ---- adapters (trainable) ----------------------------------------------------
satt = SATTAdapter(chunk_size=4)
satt_tf = n(satt) - n(satt.projector)
adapters = {
    "SATT (any c)": n(satt),
    "ChunkPool (c=4)": n(ChunkPoolAdapter(chunk_size=4)),
    "Mean-Pool": n(MeanPoolAdapter()),
}

print("=" * 72)
print("FROZEN COMPONENTS")
print(f"  SigLIP vision encoder, total       {m(vision_total)}")
print(f"    of which pooling head (unused)   {m(vision_head)}")
print(f"    used for patch features          {m(vision_total - vision_head)}")
print(f"  Llama 3.2-3B base weights          {m(llm_base)}")
print()
print("TRAINABLE COMPONENTS")
print(f"  SATT adapter                       {m(adapters['SATT (any c)'])}")
print(f"    Transformer encoder              {m(satt_tf)}")
print(f"    MLP projector                    {m(n(satt.projector))}")
print(f"  ChunkPool adapter                  {m(adapters['ChunkPool (c=4)'])}")
print(f"  Mean-Pool adapter                  {m(adapters['Mean-Pool'])}")
print(f"  LoRA (r=16, 7 projections)         {m(lora)}")
print()
print("RATIOS")
print(f"  LoRA / Llama base                  {100 * lora / llm_base:.3f} %")
print(f"  LoRA / (Llama base + LoRA)         {100 * lora / (llm_base + lora):.3f} %"
      "   <- what PEFT's print_trainable_parameters() reports")
print()
print("PHASE 2 TRAINABLE (adapter + LoRA) AND SHARE OF THE WHOLE SYSTEM")
for name, a in adapters.items():
    total = vision_total + llm_base + a + lora
    print(f"  {name:<16} trainable {m(a + lora)}   whole system {total / 1e9:.3f} B"
          f"   trainable share {100 * (a + lora) / total:.3f} %")
print("=" * 72)