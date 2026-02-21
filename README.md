# Multimodal SLM via Embedding Distillation

Teach a frozen small LLM to "see" by training only a tiny projection adapter
that maps CLIP/SigLIP vision embeddings into the LLM's text embedding space.

```
[Frozen CLIP ViT-B/16] ──► [Trainable Adapter] ──► [Frozen Phi-2 / Gemma-2B]
       768-dim                  ~1–50M params              2048/2560-dim
```

---

## Project Structure

```
multimodal_slm/
│
├── multimodal_slm.ipynb        ← Main notebook (start here)
│
├── models/
│   └── adapter.py              ← LinearAdapter, MLPAdapter, PerceiverAdapter,
│                                 VisionLanguageModel
├── data/
│   └── dataset.py              ← COCOCaptionsDataset, ConceptualCaptionsDataset,
│                                 VQAv2Dataset, build_dataloader
├── training/
│   └── trainer.py              ← Trainer: mixed precision, grad accum,
│                                 cosine schedule, optional contrastive/L2 loss
├── evaluation/
│   └── evaluator.py            ← VQAEvaluator, CaptionEvaluator, EmbeddingAnalyzer
│                                 (CKA, t-SNE, cosine similarity stats)
├── utils/
│   └── config.py               ← ExperimentConfig dataclass + preset ablation configs
│
└── requirements.txt
```

---

## Quickstart (Google Colab T4)

1. Upload all files to `/content/multimodal_slm/`
2. Open `multimodal_slm.ipynb`
3. Run **Phase 1** cells to install deps and download COCO val images (~1 GB)
4. Run **Phase 2** cells – smoke test with `get_config("smoke_test")` first (~5 min)
5. Switch to `get_config("baseline")` for full training (~8–12 hours)

---

## Adapter Architectures

| Name          | Class            | Params (768→2048) | Description                              |
|---------------|------------------|-------------------|------------------------------------------|
| `linear`      | LinearAdapter    | ~1.6M             | Single linear layer. Fastest.            |
| `mlp`         | MLPAdapter       | ~3.1M             | Linear-GELU-Linear. More expressive.     |
| `perceiver`   | PerceiverAdapter | ~15–50M           | Learnable queries + cross-attention.     |

---

## Training Objectives

| Loss              | Flag                  | Description                                    |
|-------------------|-----------------------|------------------------------------------------|
| LM (primary)      | always on             | Autoregressive captioning loss                 |
| Contrastive       | `use_contrastive=True`| InfoNCE: pull matched (image,text) pairs close |
| L2 alignment      | `use_l2=True`         | Auxiliary MSE between mean-pooled embeddings   |

---

## Ablation Presets

Run `get_config(name)` with any of:

| Config name              | What it tests                          |
|--------------------------|----------------------------------------|
| `baseline`               | Linear adapter, CLS token, LM loss only|
| `ablation_mlp`           | MLP adapter                            |
| `ablation_perceiver`     | Perceiver adapter, all patch tokens    |
| `ablation_contrastive`   | Adds contrastive loss term             |
| `ablation_patches`       | All 196 patch tokens vs CLS-only       |
| `smoke_test`             | 500 samples, 1 epoch – verify pipeline |

---

## Key Metrics

- **CKA similarity** (target > 0.7): measures how aligned visual and text embedding
  distributions are after the adapter projection
- **Cosine similarity gap**: matched pairs should score much higher than random pairs
- **VQA accuracy**: on VQAv2 val subset
- **BLEU-1 / CIDEr**: zero-shot captioning quality on COCO val

---

## Compute Budget

| Config            | Dataset           | Expected time (T4 16 GB) |
|-------------------|-------------------|--------------------------|
| smoke_test        | 500 COCO samples  | ~5 min                   |
| baseline          | COCO 118K         | ~8–12 hours              |
| baseline (CC3M)   | CC3M 3.3M         | ~18–20 hours             |

All within the free Colab T4 budget.

---

## References

- LLaVA (Liu et al., 2023): https://arxiv.org/abs/2304.08485
- CLIP (Radford et al., 2021): https://openai.com/research/clip
- CKA (Kornblith et al., 2019): https://arxiv.org/abs/1905.00414
