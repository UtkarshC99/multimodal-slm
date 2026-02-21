"""
utils/config.py
---------------
Centralised experiment configuration using Python dataclasses.
All hyperparameters live here; this dict is also logged to W&B for
full reproducibility.

Usage:
    from utils.config import ExperimentConfig, get_config
    cfg = get_config("baseline")   # or "ablation_mlp", "ablation_perceiver", etc.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


@dataclass
class ExperimentConfig:
    # ---- Identity --------------------------------------------------------
    experiment_name: str = "baseline"

    # ---- Vision encoder --------------------------------------------------
    vision_model_name: str = "openai/clip-vit-base-patch16"
    vision_dim: int = 768            # ViT-B/16 hidden dim
    use_cls_only: bool = True        # True = 1 visual token; False = 196 patch tokens

    # ---- Language model --------------------------------------------------
    lm_model_name: str = "microsoft/phi-2"
    # Note: Gemma-2B requires HuggingFace access approval.
    # Phi-2 is freely available. Switch to "google/gemma-2b" if you have access.
    text_dim: int = 2560             # Phi-2 hidden dim (2048 for Gemma-2B)

    # ---- Adapter ---------------------------------------------------------
    adapter_type: Literal["linear", "mlp", "perceiver"] = "linear"
    # MLP-specific
    mlp_hidden_scale: float = 2.0
    # Perceiver-specific
    perceiver_num_queries: int = 16
    perceiver_num_heads: int = 8
    perceiver_num_layers: int = 2

    # ---- Dataset ---------------------------------------------------------
    dataset: Literal["coco", "cc3m"] = "coco"
    coco_root: str = "/content/coco"         # adjust to your Colab path
    cc3m_tsv: str = "/content/cc3m/train.tsv"
    cc3m_image_dir: str = "/content/cc3m/images"
    max_caption_length: int = 128
    max_train_samples: Optional[int] = None   # None = use all; set small for smoke-tests
    max_val_samples: Optional[int] = 1000

    # ---- Training --------------------------------------------------------
    num_epochs: int = 3
    batch_size: int = 16
    grad_accum_steps: int = 4          # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # ---- Loss ------------------------------------------------------------
    use_contrastive: bool = False
    contrastive_weight: float = 0.1
    use_l2: bool = False
    l2_weight: float = 0.01

    # ---- Infrastructure --------------------------------------------------
    mixed_precision: bool = True
    num_workers: int = 2
    output_dir: str = "checkpoints"
    log_every: int = 10
    save_every: int = 500
    use_wandb: bool = False
    wandb_project: str = "multimodal-slm"
    seed: int = 42

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Preset configurations for ablation studies
# ---------------------------------------------------------------------------

PRESETS = {
    # Phase 2 baseline: linear adapter, CLS-only, LM loss only
    "baseline": ExperimentConfig(
        experiment_name="baseline",
        adapter_type="linear",
        use_cls_only=True,
        use_contrastive=False,
        use_l2=False,
    ),

    # Ablation: MLP adapter instead of linear
    "ablation_mlp": ExperimentConfig(
        experiment_name="ablation_mlp",
        adapter_type="mlp",
        use_cls_only=True,
        use_contrastive=False,
    ),

    # Ablation: Perceiver adapter (16 query tokens, full patch embeddings)
    "ablation_perceiver": ExperimentConfig(
        experiment_name="ablation_perceiver",
        adapter_type="perceiver",
        use_cls_only=False,   # needs patch tokens for cross-attention to make sense
        perceiver_num_queries=16,
    ),

    # Ablation: add contrastive loss to baseline
    "ablation_contrastive": ExperimentConfig(
        experiment_name="ablation_contrastive",
        adapter_type="linear",
        use_contrastive=True,
        contrastive_weight=0.1,
    ),

    # Ablation: all patches (196 tokens) instead of CLS-only
    "ablation_patches": ExperimentConfig(
        experiment_name="ablation_patches",
        adapter_type="mlp",
        use_cls_only=False,
    ),

    # Quick smoke-test: 500 train samples, 1 epoch
    "smoke_test": ExperimentConfig(
        experiment_name="smoke_test",
        adapter_type="linear",
        use_cls_only=True,
        num_epochs=1,
        max_train_samples=500,
        max_val_samples=100,
        log_every=5,
        save_every=100,
    ),
}


def get_config(name: str = "baseline") -> ExperimentConfig:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS.keys())}")
    cfg = PRESETS[name]
    print(f"Loaded config: '{name}'")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    return cfg
