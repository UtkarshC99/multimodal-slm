"""
utils/config.py
---------------
Centralised experiment configuration. All hyperparameters live here and are
version-controlled in git. The notebook reads from this file — never hardcode
values in the notebook itself.

Drive layout assumed:
    My Drive/Projects/Multimodal SLM/
        data/coco/          ← COCO downloaded once, reused every session
        checkpoints/        ← per-experiment subdirs created automatically
"""

from dataclasses import dataclass, asdict
from typing import Optional, Literal

# ---------------------------------------------------------------------------
# Paths — single source of truth, referenced by notebook and trainer
# ---------------------------------------------------------------------------
DRIVE_ROOT      = "/content/drive/MyDrive/Projects/Multimodal SLM"
COCO_ROOT       = f"{DRIVE_ROOT}/data/coco"
CHECKPOINTS_ROOT = f"{DRIVE_ROOT}/checkpoints"


@dataclass
class ExperimentConfig:
    # ---- Identity --------------------------------------------------------
    experiment_name: str = "baseline"

    # ---- Vision encoder --------------------------------------------------
    vision_model_name: str = "openai/clip-vit-base-patch16"
    vision_dim: int = 768
    use_cls_only: bool = True          # False = 196 patch tokens; pair with perceiver

    # ---- Language model --------------------------------------------------
    lm_model_name: str = "microsoft/phi-2"
    text_dim: int = 2560               # Phi-2. Gemma-2B = 2048
    use_4bit: bool = True              # Always True on T4 — saves ~4 GB VRAM

    # ---- Adapter ---------------------------------------------------------
    adapter_type: Literal["linear", "mlp", "perceiver"] = "mlp"
    mlp_hidden_scale: float = 2.0
    perceiver_num_queries: int = 16
    perceiver_num_heads: int = 8
    perceiver_num_layers: int = 2

    # ---- Dataset ---------------------------------------------------------
    dataset: Literal["coco", "cc3m"] = "coco"
    coco_root: str = COCO_ROOT
    train_split: str = "val"           # "val" for smoke/first_run; "train" for baseline
    max_caption_length: int = 128
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 1000

    # ---- Training --------------------------------------------------------
    num_epochs: int = 3
    batch_size: int = 2                # T4-safe; effective = batch_size * grad_accum
    grad_accum_steps: int = 8          # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    num_workers: int = 0               # 0 = safest on Colab

    # ---- Loss ------------------------------------------------------------
    use_contrastive: bool = True       # Always on — prevents mode collapse
    contrastive_weight: float = 0.1
    use_l2: bool = False
    l2_weight: float = 0.01

    # ---- Infrastructure --------------------------------------------------
    output_dir: str = CHECKPOINTS_ROOT  # subdir per experiment added by notebook
    log_every: int = 10
    save_every: int = 500
    use_wandb: bool = False
    wandb_project: str = "multimodal-slm"
    seed: int = 42

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {

    # ── smoke_test ────────────────────────────────────────────────────────
    # Purpose : verify the full pipeline runs end-to-end without errors.
    # Data    : 5 000 samples from COCO val2017 (already on Drive after first
    #           download — no train2017 needed).
    # Runtime : ~10 min on T4.
    # Pass    : loss decreasing, CKA improves vs baseline, no OOM.
    "smoke_test": ExperimentConfig(
        experiment_name   = "smoke_test",
        adapter_type      = "mlp",         # MLP from the start — linear collapses
        use_cls_only      = True,
        train_split       = "val",         # uses val2017, no train download needed
        num_epochs        = 5,             # enough steps to see real movement
        max_train_samples = 5000,
        max_val_samples   = 1000,
        learning_rate     = 5e-4,          # higher LR compensates for fewer steps
        warmup_ratio      = 0.01,          # short warmup — only ~300 total steps
        use_contrastive   = True,
        contrastive_weight= 0.2,           # higher weight; LM loss alone collapses
        batch_size        = 2,
        grad_accum_steps  = 8,
        num_workers       = 0,
        log_every         = 10,
        save_every        = 999999,        # no mid-run saves; saves at epoch end only
    ),

    # ── baseline ──────────────────────────────────────────────────────────
    # Purpose : the primary trained model and reference point for ablations.
    # Data    : full COCO train2017 (118 237 images). Requires train2017
    #           downloaded to Drive (cell 1.6b in notebook).
    # Runtime : ~10–12 h on T4.
    # Pass    : CKA > 0.5, visible caption diversity in qualitative cell.
    "baseline": ExperimentConfig(
        experiment_name   = "baseline",
        adapter_type      = "mlp",
        use_cls_only      = True,
        train_split       = "train",       # requires train2017 on Drive
        num_epochs        = 3,
        max_train_samples = None,          # all 118 237 train samples
        max_val_samples   = 1000,
        learning_rate     = 2e-4,
        warmup_ratio      = 0.05,
        use_contrastive   = True,
        contrastive_weight= 0.1,
        batch_size        = 2,
        grad_accum_steps  = 8,
        num_workers       = 2,
        log_every         = 50,
        save_every        = 500,
    ),

    # ── ablation_linear ───────────────────────────────────────────────────
    # Isolates: does nonlinearity in the adapter matter?
    # Same data/epochs as smoke_test so results are directly comparable.
    "ablation_linear": ExperimentConfig(
        experiment_name   = "ablation_linear",
        adapter_type      = "linear",
        use_cls_only      = True,
        train_split       = "val",
        num_epochs        = 5,
        max_train_samples = 5000,
        max_val_samples   = 1000,
        learning_rate     = 5e-4,
        warmup_ratio      = 0.01,
        use_contrastive   = True,
        contrastive_weight= 0.2,
        batch_size        = 2,
        grad_accum_steps  = 8,
        num_workers       = 0,
        log_every         = 10,
        save_every        = 999999,
    ),

    # ── ablation_no_contrastive ───────────────────────────────────────────
    # Isolates: how much does contrastive loss contribute vs LM loss alone?
    # Identical to smoke_test except contrastive is off.
    "ablation_no_contrastive": ExperimentConfig(
        experiment_name   = "ablation_no_contrastive",
        adapter_type      = "mlp",
        use_cls_only      = True,
        train_split       = "val",
        num_epochs        = 5,
        max_train_samples = 5000,
        max_val_samples   = 1000,
        learning_rate     = 5e-4,
        warmup_ratio      = 0.01,
        use_contrastive   = False,         # ← the only difference from smoke_test
        batch_size        = 2,
        grad_accum_steps  = 8,
        num_workers       = 0,
        log_every         = 10,
        save_every        = 999999,
    ),

    # ── ablation_perceiver ────────────────────────────────────────────────
    # Isolates: does spatial patch information (vs CLS-only) improve quality?
    # Uses Perceiver to compress 196 patch tokens → 16 queries before LLM,
    # keeping sequence length manageable on T4.
    "ablation_perceiver": ExperimentConfig(
        experiment_name   = "ablation_perceiver",
        adapter_type      = "perceiver",
        use_cls_only      = False,         # all 196 patch tokens fed to perceiver
        train_split       = "val",
        num_epochs        = 5,
        max_train_samples = 5000,
        max_val_samples   = 1000,
        learning_rate     = 5e-5,          # lower LR — cross-attention is sensitive
        warmup_ratio      = 0.05,          # longer warmup for cross-attn stability
        use_contrastive   = True,
        contrastive_weight= 0.2,
        batch_size        = 2,
        grad_accum_steps  = 8,
        num_workers       = 0,
        log_every         = 10,
        save_every        = 999999,
    ),

    # ── ablation_contrastive_weight ───────────────────────────────────────
    # Isolates: what is the optimal contrastive weight?
    # Run alongside smoke_test (weight=0.2) and ablation_no_contrastive
    # (weight=0) to find the sweet spot.
    "ablation_contrastive_weight": ExperimentConfig(
        experiment_name   = "ablation_contrastive_weight",
        adapter_type      = "mlp",
        use_cls_only      = True,
        train_split       = "val",
        num_epochs        = 5,
        max_train_samples = 5000,
        max_val_samples   = 1000,
        learning_rate     = 5e-4,
        warmup_ratio      = 0.01,
        use_contrastive   = True,
        contrastive_weight= 0.05,          # ← sweep: 0, 0.05, 0.1, 0.2
        batch_size        = 2,
        grad_accum_steps  = 8,
        num_workers       = 0,
        log_every         = 10,
        save_every        = 999999,
    ),
}


def get_config(name: str = "smoke_test") -> ExperimentConfig:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    cfg = PRESETS[name]
    print(f"Config loaded: '{name}'")
    print(f"  adapter      : {cfg.adapter_type}  (use_cls_only={cfg.use_cls_only})")
    print(f"  train_split  : {cfg.train_split}  max_train={cfg.max_train_samples}")
    print(f"  epochs       : {cfg.num_epochs}  lr={cfg.learning_rate}  warmup={cfg.warmup_ratio}")
    print(f"  contrastive  : {cfg.use_contrastive}  weight={cfg.contrastive_weight}")
    print(f"  batch_size   : {cfg.batch_size}  grad_accum={cfg.grad_accum_steps}  "
          f"(effective={cfg.batch_size * cfg.grad_accum_steps})")
    return cfg
