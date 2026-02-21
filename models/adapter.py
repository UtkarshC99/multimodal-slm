"""
models/adapter.py
-----------------
Projection adapter architectures that bridge vision embeddings → LLM text space.
Three variants are implemented so you can ablate them directly:
  - LinearAdapter      : single nn.Linear  (fastest, fewest params)
  - MLPAdapter         : Linear-GELU-Linear (more expressive)
  - PerceiverAdapter   : learnable query tokens + cross-attention (most powerful)

The full VisionLanguageModel wraps a frozen vision encoder, a chosen adapter,
and a frozen LLM into one nn.Module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPVisionModel,
    CLIPVisionConfig,
    SiglipVisionModel,
)
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Adapter architectures
# ---------------------------------------------------------------------------

class LinearAdapter(nn.Module):
    """Simple single-layer linear projection. Fastest training, minimal params."""

    def __init__(self, vision_dim: int = 768, text_dim: int = 2048):
        super().__init__()
        self.proj = nn.Linear(vision_dim, text_dim)

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        # vision_embeds: (B, N, vision_dim)  or  (B, vision_dim) for [CLS]-only
        return self.proj(vision_embeds)


class MLPAdapter(nn.Module):
    """Two-layer MLP with GELU activation. Adds non-linearity at minimal cost."""

    def __init__(self, vision_dim: int = 768, text_dim: int = 2048, hidden_scale: float = 2.0):
        super().__init__()
        hidden = int(vision_dim * hidden_scale)
        self.net = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, text_dim),
        )

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        return self.net(vision_embeds)


class PerceiverAdapter(nn.Module):
    """
    Perceiver-style cross-attention resampler.
    Learns a fixed set of `num_queries` tokens that attend to all patch tokens,
    compressing variable-length vision sequences into a fixed-size representation.
    Useful when you want to control exactly how many tokens the LLM sees.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 2048,
        num_queries: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, vision_dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=vision_dim,
                nhead=num_heads,
                dim_feedforward=vision_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # Final projection to LLM dimension
        self.out_proj = nn.Linear(vision_dim, text_dim)

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        # vision_embeds: (B, N_patches, vision_dim)
        B = vision_embeds.size(0)
        q = self.queries.expand(B, -1, -1)  # (B, num_queries, vision_dim)
        for layer in self.layers:
            q = layer(q, vision_embeds)
        return self.out_proj(q)  # (B, num_queries, text_dim)


ADAPTER_REGISTRY = {
    "linear": LinearAdapter,
    "mlp": MLPAdapter,
    "perceiver": PerceiverAdapter,
}


# ---------------------------------------------------------------------------
# Full multimodal model
# ---------------------------------------------------------------------------

class VisionLanguageModel(nn.Module):
    """
    Frozen vision encoder  +  trainable adapter  +  frozen LLM.
    Only the adapter weights are updated during training.

    Input format fed to the LLM:
        [PROJECTED_IMAGE_TOKENS]  [BOS]  <text tokens>

    Parameters
    ----------
    vision_model_name : str
        HuggingFace model id for the vision encoder.
        Tested with: "openai/clip-vit-base-patch16", "google/siglip-base-patch16-224"
    lm_model_name : str
        HuggingFace model id for the language model.
        Tested with: "google/gemma-2b", "microsoft/phi-2"
    adapter_type : "linear" | "mlp" | "perceiver"
    use_cls_only : bool
        If True, only the [CLS] token embedding is used from the vision encoder.
        If False, all patch token embeddings are used (more info, more compute).
    vision_dim : int
        Output embedding dim of the vision encoder (768 for ViT-B/16).
    text_dim : int
        Input embedding dim of the LLM (2048 for Gemma-2B, 2560 for Phi-2).
    """

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch16",
        lm_model_name: str = "google/gemma-2b",
        adapter_type: Literal["linear", "mlp", "perceiver"] = "linear",
        use_cls_only: bool = True,
        vision_dim: int = 768,
        text_dim: int = 2048,
        use_4bit: bool = True,
        **adapter_kwargs,
    ):
        super().__init__()

        # ---- Vision encoder (frozen) ----------------------------------------
        print(f"Loading vision encoder: {vision_model_name}")
        if "siglip" in vision_model_name.lower():
            self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_name)
        else:
            self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self._freeze(self.vision_encoder)

        # ---- Adapter (trainable) -------------------------------------------
        AdapterClass = ADAPTER_REGISTRY[adapter_type]
        self.adapter = AdapterClass(vision_dim=vision_dim, text_dim=text_dim, **adapter_kwargs)
        print(f"Adapter: {adapter_type}  |  params: {sum(p.numel() for p in self.adapter.parameters()):,}")

        # ---- Language model (frozen) ----------------------------------------
        print(f"Loading language model: {lm_model_name}  (4-bit={use_4bit})")
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # nested quant saves ~0.4 GB extra
                bnb_4bit_quant_type="nf4",
            )
            self.lm = AutoModelForCausalLM.from_pretrained(
                lm_model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                lm_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        self._freeze(self.lm)

        self.use_cls_only = use_cls_only

    # ------------------------------------------------------------------
    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen vision encoder and project into LLM embedding space.

        Returns
        -------
        torch.Tensor  shape (B, num_visual_tokens, text_dim)
        """
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pixel_values)
            if self.use_cls_only:
                # pooler_output is the [CLS] embedding
                vision_embeds = outputs.pooler_output.unsqueeze(1)  # (B, 1, D)
            else:
                # last_hidden_state contains all patch tokens
                vision_embeds = outputs.last_hidden_state  # (B, N, D)

        projected = self.adapter(vision_embeds.float())  # keep adapter in fp32
        return projected.to(self.lm.dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Standard forward pass for training.
        `labels` should be -100 for vision tokens (no loss computed on them).
        """
        # Visual tokens in LLM embedding space
        visual_embeds = self.encode_images(pixel_values)  # (B, Nv, D)

        # Text tokens
        text_embeds = self.lm.get_input_embeddings()(input_ids)  # (B, Nt, D)

        # Concatenate: [vision | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, Nv+Nt, D)

        # Extend attention mask if provided
        if attention_mask is not None:
            B, Nv = visual_embeds.shape[:2]
            vision_mask = torch.ones(B, Nv, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Extend labels if provided (mask visual positions with -100)
        if labels is not None:
            B, Nv = visual_embeds.shape[:2]
            vision_labels = torch.full((B, Nv), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([vision_labels, labels], dim=1)

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        **generate_kwargs,
    ):
        """Generate text given image + prompt prefix."""
        visual_embeds = self.encode_images(pixel_values)
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            B, Nv = visual_embeds.shape[:2]
            vision_mask = torch.ones(B, Nv, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        return self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())