"""
evaluation/evaluator.py
-----------------------
Evaluation utilities for:
  1. VQA accuracy  (exact match against VQAv2 ground truth)
  2. Zero-shot captioning  (BLEU-4, CIDEr via pycocoevalcap)
  3. Embedding space alignment  (CKA similarity)

Usage:
    from evaluation.evaluator import VQAEvaluator, CaptionEvaluator, EmbeddingAnalyzer
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# VQA Evaluator
# ---------------------------------------------------------------------------

class VQAEvaluator:
    """
    Measures VQA accuracy using exact-match (with basic normalisation).

    VQAv2 official metric: an answer is correct if it matches ≥3 of the 10
    annotated answers. For a fast approximate measure, we compare the model's
    first token / short generation against the most common annotated answer.
    """

    def __init__(self, model, tokenizer, processor, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

    @staticmethod
    def _normalize(answer: str) -> str:
        """Lower-case, remove punctuation, strip whitespace."""
        answer = answer.lower().strip()
        answer = re.sub(r"[^\w\s]", "", answer)
        return answer.strip()

    @torch.inference_mode()
    def evaluate(
        self,
        dataloader,
        max_new_tokens: int = 10,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run evaluation over `dataloader` (VQAv2Dataset).
        Returns dict with keys: accuracy, num_correct, num_total.
        """
        self.model.eval()
        correct, total = 0, 0
        results = []

        for batch in tqdm(dataloader, desc="VQA Eval", disable=not verbose):
            pixel_values   = batch["pixel_values"].to(self.device)
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            gt_answers     = batch["answer"]  # list of strings

            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            # Decode only the newly generated tokens
            prompt_len = input_ids.shape[1]
            pred_ids = generated_ids[:, prompt_len + pixel_values.shape[0]:]  # strip prompt
            pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for pred, gt in zip(pred_texts, gt_answers):
                pred_norm = self._normalize(pred.split("\n")[0])
                gt_norm   = self._normalize(gt)
                match = int(pred_norm == gt_norm or pred_norm.startswith(gt_norm))
                correct += match
                total += 1
                results.append({"pred": pred_norm, "gt": gt_norm, "correct": match})

        acc = correct / max(1, total)
        print(f"\nVQA Accuracy: {acc:.4f}  ({correct}/{total})")
        return {"accuracy": acc, "num_correct": correct, "num_total": total, "results": results}


# ---------------------------------------------------------------------------
# Captioning Evaluator
# ---------------------------------------------------------------------------

class CaptionEvaluator:
    """
    Evaluates zero-shot captioning on COCO val using BLEU and CIDEr.

    Requires: pip install pycocoevalcap pycocotools

    If not installed, falls back to a simple token-overlap BLEU-1 approximation.
    """

    def __init__(self, model, tokenizer, processor, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

        try:
            from pycocoevalcap.eval import COCOEvalCap
            from pycocotools.coco import COCO
            self._has_coco_eval = True
        except ImportError:
            print("⚠ pycocoevalcap not found. Using simple BLEU-1 fallback.")
            self._has_coco_eval = False

    @torch.inference_mode()
    def generate_captions(
        self,
        dataloader,
        prompt: str = "Describe this image:",
        max_new_tokens: int = 50,
    ) -> List[Dict]:
        """Generate captions for all images in dataloader."""
        self.model.eval()
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        records = []

        for batch in tqdm(dataloader, desc="Generating captions"):
            pixel_values = batch["pixel_values"].to(self.device)
            B = pixel_values.size(0)

            prompt_ids_expanded = prompt_ids.expand(B, -1)
            attention_mask = torch.ones_like(prompt_ids_expanded)

            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=prompt_ids_expanded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            prompt_len = prompt_ids.shape[-1]
            pred_ids = generated_ids[:, prompt_len:]
            pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for pred, gt in zip(pred_texts, batch["caption"]):
                records.append({"prediction": pred.strip(), "ground_truth": gt})

        return records

    def bleu1_approx(self, records: List[Dict]) -> float:
        """Simple token-overlap BLEU-1 (no smoothing) for quick sanity check."""
        scores = []
        for r in records:
            pred_tokens = set(r["prediction"].lower().split())
            gt_tokens   = set(r["ground_truth"].lower().split())
            if not pred_tokens:
                scores.append(0.0)
                continue
            overlap = len(pred_tokens & gt_tokens)
            scores.append(overlap / len(pred_tokens))
        return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Embedding Space Analyzer
# ---------------------------------------------------------------------------

class EmbeddingAnalyzer:
    """
    Tools for measuring and visualising embedding space alignment.

    Methods:
        cka_similarity  – Centered Kernel Alignment (linear kernel)
        tsne_plot       – t-SNE visualisation of vision vs text embeddings
        cosine_sim_hist – histogram of pairwise cosine similarities
    """

    @staticmethod
    def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute linear CKA between matrices X (n×p) and Y (n×q).
        CKA is invariant to orthogonal transformations and isotropic scaling.
        Returns a value in [0, 1] where 1 = perfectly aligned.

        Reference: Kornblith et al., 2019  https://arxiv.org/abs/1905.00414
        """
        def _hsic(A: np.ndarray, B: np.ndarray) -> float:
            n = A.shape[0]
            H = np.eye(n) - 1 / n  # centering matrix
            KA = A @ A.T
            KB = B @ B.T
            return np.trace(KA @ H @ KB @ H) / (n - 1) ** 2

        hsic_xy = _hsic(X, Y)
        hsic_xx = _hsic(X, X)
        hsic_yy = _hsic(Y, Y)
        denom = np.sqrt(hsic_xx * hsic_yy)
        return float(hsic_xy / denom) if denom > 0 else 0.0

    @torch.inference_mode()
    def compute_cka(
        self,
        model,
        dataloader,
        device: str = "cuda",
        max_batches: int = 50,
    ) -> float:
        """
        Compute CKA between projected visual embeddings and text embeddings
        over `max_batches` of `dataloader`.

        Returns CKA ∈ [0, 1]
        """
        model.eval()
        vis_list, txt_list = [], []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Visual embeddings
            vis_emb = model.encode_images(pixel_values)            # (B, Nv, D)
            vis_emb = vis_emb.mean(dim=1)                          # (B, D)

            # Text embeddings (mean over non-pad tokens)
            txt_emb = model.lm.get_input_embeddings()(input_ids)  # (B, Nt, D)
            mask = attention_mask.unsqueeze(-1).float()
            txt_emb = (txt_emb * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, D)

            vis_list.append(vis_emb.float().cpu().numpy())
            txt_list.append(txt_emb.float().cpu().numpy())

        X = np.concatenate(vis_list, axis=0)
        Y = np.concatenate(txt_list, axis=0)

        cka = self._linear_cka(X, Y)
        print(f"Linear CKA (visual ↔ text): {cka:.4f}")
        return cka

    @torch.inference_mode()
    def tsne_plot(
        self,
        model,
        dataloader,
        device: str = "cuda",
        max_batches: int = 20,
        perplexity: int = 30,
        save_path: Optional[str] = None,
    ):
        """
        2D t-SNE plot of visual (blue) and text (red) embeddings.
        Requires matplotlib and scikit-learn.
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
        except ImportError:
            print("⚠ Install matplotlib and scikit-learn for t-SNE plots.")
            return

        model.eval()
        vis_list, txt_list = [], []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            vis_emb = model.encode_images(pixel_values).mean(dim=1).float().cpu().numpy()
            txt_emb = model.lm.get_input_embeddings()(input_ids)
            mask = attention_mask.unsqueeze(-1).float()
            txt_emb = ((txt_emb * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()

            vis_list.append(vis_emb)
            txt_list.append(txt_emb)

        X_vis = np.concatenate(vis_list, axis=0)
        X_txt = np.concatenate(txt_list, axis=0)
        X_all = np.concatenate([X_vis, X_txt], axis=0)
        labels = np.array([0] * len(X_vis) + [1] * len(X_txt))

        print("Running t-SNE …")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_2d = tsne.fit_transform(X_all)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], alpha=0.4, s=8, label="Visual", c="steelblue")
        ax.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], alpha=0.4, s=8, label="Text",   c="tomato")
        ax.legend()
        ax.set_title("t-SNE: Vision vs Text Embedding Space")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"t-SNE plot saved → {save_path}")
        else:
            plt.show()

        return fig

    @torch.inference_mode()
    def cosine_similarity_stats(
        self,
        model,
        dataloader,
        device: str = "cuda",
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """
        Compute mean pairwise cosine similarity between visual and text embeddings.
        High diagonal similarity (matched pairs) vs low off-diagonal (unmatched)
        indicates good alignment.
        """
        model.eval()
        diag_sims, off_diag_sims = [], []

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            vis_emb = F.normalize(model.encode_images(pixel_values).mean(dim=1).float(), dim=-1)
            txt_emb = model.lm.get_input_embeddings()(input_ids).float()
            mask = attention_mask.unsqueeze(-1).float()
            txt_emb = F.normalize((txt_emb * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

            sim_matrix = vis_emb @ txt_emb.T  # (B, B)
            B = sim_matrix.size(0)

            for j in range(B):
                diag_sims.append(sim_matrix[j, j].item())
                for k in range(B):
                    if k != j:
                        off_diag_sims.append(sim_matrix[j, k].item())

        stats = {
            "matched_cosine_sim":   float(np.mean(diag_sims)),
            "unmatched_cosine_sim": float(np.mean(off_diag_sims)),
            "alignment_gap":        float(np.mean(diag_sims) - np.mean(off_diag_sims)),
        }
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        return stats
