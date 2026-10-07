"""
eval_paths.py — resolves which Phase-2 checkpoint files an evaluation loads.

Kept free of heavy imports so it can be tested without a GPU.

Background: until October 2026, evaluate.py looked for the adapter under
"phase2_best_satt_chunk<c>.pt" while train.py saved it as
"phase2_best_satt_satt_chunk<c>.pt", and silently fell back to the original
c=4 file when the name was not found. It also always loaded the LoRA weights
from "phase2_best_lora". As a result the c=2 and c=8 evaluations ran the c=4
weights, and the Mean-Pool evaluation paired its adapter with the c=4 LoRA.

Rules enforced here:
  * adapter and LoRA must belong to the SAME training run;
  * names follow train.py exactly: phase2_best_satt_<tag>.pt and
    phase2_best_lora_<tag>, with <tag> = "baseline" or "satt_chunk<c>";
  * the un-suffixed files from the original run are accepted ONLY for
    SATT c=4, the configuration that produced them;
  * anything else raises an error. There is no fallback to another variant.
"""

import os


def variant_tag(model_type: str, chunk_size: int) -> str:
    """Same naming as train._ckpt_prefix."""
    return "baseline" if model_type == "baseline" else f"satt_chunk{chunk_size}"


def resolve_checkpoints(checkpoint_dir: str, model_type: str, chunk_size: int):
    """Return (adapter_path, lora_dir, source_description)."""
    tag = variant_tag(model_type, chunk_size)
    adapter = os.path.join(checkpoint_dir, f"phase2_best_satt_{tag}.pt")
    lora = os.path.join(checkpoint_dir, f"phase2_best_lora_{tag}")
    a_ok = os.path.isfile(adapter)
    l_ok = os.path.isfile(os.path.join(lora, "adapter_config.json"))

    if a_ok and l_ok:
        return adapter, lora, f"variant-specific files for {tag}"
    if a_ok != l_ok:
        raise FileNotFoundError(
            f"Incomplete Phase-2 checkpoint for {tag}: "
            f"adapter {'found' if a_ok else 'MISSING'} ({adapter}), "
            f"LoRA {'found' if l_ok else 'MISSING'} ({lora}). "
            f"Refusing to mix weights from different training runs.")

    if model_type == "satt" and chunk_size == 4:
        legacy_adapter = os.path.join(checkpoint_dir, "phase2_best_satt.pt")
        legacy_lora = os.path.join(checkpoint_dir, "phase2_best_lora")
        if (os.path.isfile(legacy_adapter)
                and os.path.isfile(os.path.join(legacy_lora, "adapter_config.json"))):
            return legacy_adapter, legacy_lora, "original (un-suffixed) c=4 run"

    raise FileNotFoundError(
        f"No Phase-2 checkpoint found for {tag}. Expected {adapter} and {lora}. "
        f"Refusing to fall back to another variant's weights.")
