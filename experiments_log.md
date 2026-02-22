# Experiment Tracking Log

This document tracks all training runs for the Multimodal SLM project.

---

## Experiment Template

```markdown
### [experiment_name] - YYYY-MM-DD

**Config:**
- Adapter: [type]
- Samples: [train_samples] train, [val_samples] val
- Epochs: [num_epochs]
- Learning rate: [lr]
- Contrastive weight: [weight]
- Batch size: [batch] × [grad_accum] = [effective]

**Results:**
- CKA before: [value]
- CKA after: [value]
- CKA delta: [value]
- BLEU-1: [value]
- Val loss: [value]

**Compute:**
- GPU: [type]
- Runtime: [hours]h
- Credits used: [value]

**Observations:**
- [Key findings]
- [Issues encountered]
- [Next steps]

**Artifacts:**
- Checkpoint: `[path]`
- Plots: `[path]`
```

---

## Completed Experiments

### smoke_test - 2024-02-22

**Config:**
- Adapter: MLP (2-layer, ~5.1M params)
- Samples: 5,000 train (val2017), 1,000 val
- Epochs: 5
- Learning rate: 5e-4
- Contrastive weight: 0.2
- Batch size: 2 × 8 = 16 effective

**Results:**
- CKA before: 0.7118
- CKA after: 0.4955
- CKA delta: -0.2163 ⚠️
- BLEU-1: [pending]
- Val loss: [pending]

**Compute:**
- GPU: A100 80GB
- Runtime: ~3h
- Credits used: 23.27

**Observations:**
- CKA decreased during training (unexpected)
- Model correctly identifies entities (man, cat, dog)
- Generates partially correct captions with garbage text artifacts
- Garbage text likely from Phi-2 pretraining ("exercise 2", etc.)
- Entity recognition working suggests semantic learning despite CKA drop

**Artifacts:**
- Checkpoint: `/content/drive/MyDrive/Projects/Multimodal SLM/checkpoints/smoke_test/`
- Plots: `/content/drive/MyDrive/Projects/Multimodal SLM/plots/smoke_test/`

---

### ablation_no_contrastive - 2024-02-22

**Config:**
- Adapter: MLP (2-layer, ~5.1M params)
- Samples: 5,000 train (val2017), 1,000 val
- Epochs: 5
- Learning rate: 5e-4
- Contrastive weight: 0 (disabled)
- Batch size: 2 × 8 = 16 effective

**Results:**
- CKA before: 0.7965
- CKA after: 0.6760
- CKA delta: -0.1205 ⚠️
- BLEU-1: [pending]
- Val loss: [pending]

**Compute:**
- GPU: A100 80GB
- Runtime: ~3h
- Credits used: [pending]

**Observations:**
- CKA also decreased, but less than with contrastive loss
- Suggests contrastive loss may be too aggressive at weight=0.2
- Without contrastive, CKA retention is better (-0.12 vs -0.22)
- Qualitative results pending

**Artifacts:**
- Checkpoint: `/content/drive/MyDrive/Projects/Multimodal SLM/checkpoints/ablation_no_contrastive/`
- Plots: `/content/drive/MyDrive/Projects/Multimodal SLM/plots/ablation_no_contrastive/`

---

## Planned Experiments

### budget_baseline - Pending

**Config:**
- Adapter: MLP (2-layer, ~5.1M params)
- Samples: 50,000 train (train2017 subset), 1,000 val
- Epochs: 2
- Learning rate: 2e-4
- Contrastive weight: 0.05 (reduced from smoke_test)
- Batch size: 8 × 4 = 32 effective

**Expected:**
- Runtime: ~12h on A100 40GB
- Credits: ~93
- Better generalization than smoke_test (10× more data)
- Reduced garbage text with more diverse training
- Lower contrastive weight may preserve CKA better

---

## Analysis Notes

### CKA Interpretation
- CKA measures distributional similarity, not semantic quality
- Random projections can have high CKA by chance
- Training may lower CKA while improving semantic precision
- Entity recognition working despite CKA drop suggests useful learning

### Contrastive Loss Findings
- Weight=0.2 may be too aggressive (larger CKA drop)
- Weight=0 retains more CKA but may lack alignment pressure
- Optimal weight likely between 0.05-0.1

### Next Steps
1. Run budget_baseline with contrastive_weight=0.05
2. Compare qualitative results across all configs
3. Measure cosine similarity stats (matched vs unmatched pairs)
4. Consider BLEU-1 as primary metric over CKA
