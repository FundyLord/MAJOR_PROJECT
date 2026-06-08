import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from transformers import AutoTokenizer, AutoModelForCausalLM, SiglipVisionModel

# =========================
# 1. NIfTI Preprocessing
# =========================
def preprocess_ct(path, target_shape=(64, 224, 224)):
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)

    # Reorder axes → (Z, Y, X)
    data = np.transpose(data, (2, 1, 0))

    # Resize volume
    zoom_factors = [t/s for t, s in zip(target_shape, data.shape)]
    data = zoom(data, zoom_factors, order=3)

    # HU normalization
    data = np.clip(data, -200, 300)
    data = (data + 200) / 500

    # Convert to 3-channel
    tensor = torch.from_numpy(np.stack([data]*3, axis=1)).float()
    return tensor.unsqueeze(0)  # (1, Z, 3, H, W)


# =========================
# 2. SATT Adapter
# =========================
class SATTAdapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                batch_first=True
            ),
            num_layers=4
        )

        self.mlp_proj = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Linear(2048, 3072)
        )

    def forward(self, x):
        # x: [batch, slices, patches, dim]
        b, s, p, d = x.shape

        # Temporal grouping (compress depth)
        t = s // 4
        x = x.reshape(b, t, 4, p, d).mean(dim=2)

        # Apply temporal attention
        x = x.permute(0, 2, 1, 3).reshape(b * p, t, d)
        x = self.temporal_transformer(x)

        # Restore structure
        x = x.reshape(b, p, t, d).permute(0, 2, 1, 3)

        # Project to LLaMA embedding space
        x = self.mlp_proj(x)

        return x.reshape(b, -1, 3072)


# =========================
# 3. MAIN
# =========================
def main():
    device = "cuda"

    print("Device:", device)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # ---- LLaMA (FROZEN) ----
    llm = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16
    ).to(device)

    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # ---- Vision Encoder (FROZEN) ----
    vision_encoder = SiglipVisionModel.from_pretrained(
        "google/siglip-base-patch16-224"
    ).to(device)

    vision_encoder.eval()
    for p in vision_encoder.parameters():
        p.requires_grad = False

    # ---- Adapter (TRAINABLE) ----
    adapter = SATTAdapter().to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)

    # ---- Sample Data ----
    ct_path = "./abdomen_sample/ct.nii.gz"

    report = "Multiple hypoattenuating liver lesions suggest metastatic disease."
    prompt = "Analyze the CT scan and generate a radiology report."

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(10):

        optimizer.zero_grad()

        # ===== Vision =====
        ct = preprocess_ct(ct_path).to(device)

        with torch.no_grad():
            b, s, c, h, w = ct.shape

            # Process slices independently
            slices = ct.view(b * s, c, h, w)

            out = vision_encoder(slices)

            # Restore volume structure
            spatial = out.last_hidden_state.reshape(b, s, -1, 768)

        # Adapter converts vision → LLM tokens
        visual_tokens = adapter(spatial)
        visual_len = visual_tokens.shape[1]

        # ===== Text =====
        conv_prompt = [
            {"role": "user", "content": prompt}
        ]

        conv_full = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": report}
        ]

        # Tokenize prompt (for masking boundary)
        prompt_text = tokenizer.apply_chat_template(
            conv_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        # Tokenize full conversation
        full_text = tokenizer.apply_chat_template(
            conv_full,
            tokenize=False
        )

        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

        # Convert to embeddings
        text_embeds = llm.get_input_embeddings()(full_ids)

        # Combine vision + text
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # ===== Labels (mask everything except assistant output) =====
        labels = torch.full(
            (1, inputs_embeds.shape[1]),
            -100,
            dtype=torch.long
        ).to(device)

        # Only supervise assistant response
        labels[0, visual_len + prompt_len:] = full_ids[0, prompt_len:]

        # ===== Forward =====
        outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


# Entry
if __name__ == "__main__":
    main()