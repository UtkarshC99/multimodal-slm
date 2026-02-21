"""
training/trainer.py
-------------------
Training loop for the vision-language adapter.

Features:
  - Mixed precision (fp16) via torch.cuda.amp
  - Gradient accumulation
  - Warmup + cosine LR schedule
  - Per-step loss logging
  - Per-epoch: val loss + BLEU-1 on a sample of val_loader
    (gives a visible learning signal without waiting for Phase 4)
  - Checkpoint saving to Drive / resuming
  - Optional contrastive + L2 auxiliary losses
"""

import os
import math
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def contrastive_loss(
    visual_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) loss.
    Pulls matched (image, caption) pairs together, pushes unmatched pairs apart.
    Directly prevents mode collapse: if all images map to the same point every
    off-diagonal similarity equals the diagonal and the loss is maximised.

    visual_embeds : (B, D) mean-pooled projected vision tokens
    text_embeds   : (B, D) mean-pooled LLM input embeddings
    """
    visual_embeds = F.normalize(visual_embeds, dim=-1)
    text_embeds   = F.normalize(text_embeds,   dim=-1)
    logits = (visual_embeds @ text_embeds.T) / temperature   # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def l2_alignment_loss(
    visual_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> torch.Tensor:
    v = visual_embeds.mean(dim=1) if visual_embeds.dim() == 3 else visual_embeds
    t = text_embeds.mean(dim=1)   if text_embeds.dim()   == 3 else text_embeds
    return F.mse_loss(F.normalize(v, dim=-1), F.normalize(t, dim=-1))


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Quick BLEU-1 helper (no external deps)
# ---------------------------------------------------------------------------

def _quick_bleu1(predictions: List[str], references: List[str]) -> float:
    """
    Token-overlap BLEU-1 approximation over a list of string pairs.
    Used for per-epoch progress logging — not a substitute for the full
    pycocoevalcap evaluation in Phase 4.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens  = set(ref.lower().split())
        if not pred_tokens:
            scores.append(0.0)
            continue
        scores.append(len(pred_tokens & ref_tokens) / len(pred_tokens))
    return float(sum(scores) / max(1, len(scores)))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains the VisionLanguageModel adapter.
    Only adapter parameters receive gradients.

    Parameters
    ----------
    model            : VisionLanguageModel
    train_loader     : DataLoader
    val_loader       : DataLoader (optional but recommended)
    tokenizer        : LLM tokenizer — used for per-epoch BLEU-1 logging
    output_dir       : checkpoint directory (Drive path)
    num_epochs       : total epochs
    learning_rate    : AdamW peak LR
    weight_decay     : L2 regularisation
    warmup_ratio     : fraction of total steps for LR warmup
    grad_accum_steps : accumulate over N mini-batches before stepping
    max_grad_norm    : gradient clipping
    use_contrastive  : add InfoNCE contrastive loss term
    contrastive_weight
    use_l2           : add L2 alignment auxiliary loss
    l2_weight
    log_every        : log every N optimiser steps
    save_every       : checkpoint every N optimiser steps
    bleu_batches     : number of val batches used for per-epoch BLEU-1 (0 = skip)
    use_wandb
    wandb_project
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        tokenizer=None,
        output_dir: str = "/content/drive/MyDrive/Projects/Multimodal SLM/checkpoints",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        grad_accum_steps: int = 8,
        max_grad_norm: float = 1.0,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
        use_l2: bool = False,
        l2_weight: float = 0.01,
        log_every: int = 10,
        save_every: int = 500,
        bleu_batches: int = 50,
        use_wandb: bool = False,
        wandb_project: str = "multimodal-slm",
    ):
        self.model             = model
        self.train_loader      = train_loader
        self.val_loader        = val_loader
        self.tokenizer         = tokenizer
        self.output_dir        = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs        = num_epochs
        self.grad_accum_steps  = grad_accum_steps
        self.max_grad_norm     = max_grad_norm
        self.use_contrastive   = use_contrastive
        self.contrastive_weight= contrastive_weight
        self.use_l2            = use_l2
        self.l2_weight         = l2_weight
        self.log_every         = log_every
        self.save_every        = save_every
        self.bleu_batches      = bleu_batches

        trainable = model.trainable_parameters()
        n_trainable = sum(p.numel() for p in trainable)
        print(f"Trainable parameters : {n_trainable:,}")
        print(f"Output dir           : {self.output_dir}")

        self.optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=weight_decay
        )
        total_steps   = (len(train_loader) // grad_accum_steps) * num_epochs
        warmup_steps  = max(1, int(warmup_ratio * total_steps))
        print(f"Total opt steps : {total_steps}  |  warmup steps : {warmup_steps}")
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.scaler     = GradScaler()
        self.global_step = 0

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=wandb_project)

    # ------------------------------------------------------------------
    def _device(self):
        return next(self.model.adapter.parameters()).device

    # ------------------------------------------------------------------
    def _forward_batch(self, batch: Dict[str, Any]):
        device = self._device()
        pixel_values    = batch["pixel_values"].to(device)
        input_ids       = batch["input_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        labels          = batch["labels"].to(device)

        with autocast():
            outputs  = self.model(pixel_values=pixel_values, input_ids=input_ids,
                                  attention_mask=attention_mask, labels=labels)
            lm_loss  = outputs.loss
            losses   = {"lm_loss": lm_loss.item()}
            total    = lm_loss

            if self.use_contrastive:
                vis_emb  = self.model.encode_images(pixel_values).mean(dim=1)
                txt_emb  = self.model.lm.get_input_embeddings()(input_ids)
                mask     = attention_mask.unsqueeze(-1).float()
                txt_emb  = (txt_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
                c_loss   = contrastive_loss(vis_emb, txt_emb.float())
                total    = total + self.contrastive_weight * c_loss
                losses["contrastive_loss"] = c_loss.item()

            if self.use_l2:
                vis_emb  = self.model.encode_images(pixel_values).mean(dim=1)
                txt_emb  = self.model.lm.get_input_embeddings()(input_ids)
                mask     = attention_mask.unsqueeze(-1).float()
                txt_emb  = (txt_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
                a_loss   = l2_alignment_loss(vis_emb.unsqueeze(1), txt_emb.unsqueeze(1))
                total    = total + self.l2_weight * a_loss
                losses["l2_loss"] = a_loss.item()

            losses["total_loss"] = total.item()

        return total, losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self) -> float:
        """Mean val LM loss over the full val_loader."""
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        device = self._device()
        total, count = 0.0, 0
        for batch in self.val_loader:
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            am  = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            with autocast():
                out = self.model(pixel_values=pv, input_ids=ids,
                                 attention_mask=am, labels=lbl)
            total += out.loss.item()
            count += 1
        self.model.train()
        return total / max(1, count)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _epoch_bleu1(self) -> float:
        """
        Quick per-epoch BLEU-1 over `self.bleu_batches` val batches.
        Generates up to 10 new tokens per image (enough for short answers)
        and compares against ground-truth captions in the batch.
        Returns NaN if tokenizer is not set or val_loader is None.
        """
        if self.val_loader is None or self.tokenizer is None or self.bleu_batches == 0:
            return float("nan")

        self.model.eval()
        device  = self._device()
        prompt  = "Describe this image:"
        p_ids   = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        preds, refs = [], []

        for i, batch in enumerate(self.val_loader):
            if i >= self.bleu_batches:
                break
            pv  = batch["pixel_values"].to(device)
            B   = pv.size(0)
            p_ids_b  = p_ids.expand(B, -1)
            attn     = torch.ones_like(p_ids_b)

            gen = self.model.generate(
                pixel_values   = pv,
                input_ids      = p_ids_b,
                attention_mask = attn,
                max_new_tokens = 20,
                do_sample      = False,
            )
            new_ids = gen[:, p_ids.shape[-1]:]
            decoded = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            preds.extend([d.strip() for d in decoded])
            refs.extend(batch["caption"])

        self.model.train()
        return _quick_bleu1(preds, refs)

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str = "latest"):
        path = self.output_dir / f"adapter_{tag}.pt"
        torch.save({
            "adapter_state_dict" : self.model.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step"        : self.global_step,
        }, path)
        print(f"  ✓ Checkpoint → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.adapter.load_state_dict(ckpt["adapter_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        print(f"  ✓ Resumed from step {self.global_step}")

    # ------------------------------------------------------------------
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start  = time.time()
            running      = {}
            accum_count  = 0

            for step, batch in enumerate(self.train_loader, 1):
                loss, loss_dict = self._forward_batch(batch)
                self.scaler.scale(loss / self.grad_accum_steps).backward()

                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v / self.grad_accum_steps

                accum_count += 1
                if accum_count < self.grad_accum_steps:
                    continue

                # ── optimiser step ──────────────────────────────────────
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                accum_count = 0

                # ── step logging ────────────────────────────────────────
                if self.global_step % self.log_every == 0:
                    lr  = self.scheduler.get_last_lr()[0]
                    msg = (f"[Ep {epoch}/{self.num_epochs}] "
                           f"step={self.global_step}  "
                           f"lm={running.get('lm_loss',0):.4f}  "
                           f"total={running.get('total_loss',0):.4f}  "
                           f"lr={lr:.2e}")
                    if "contrastive_loss" in running:
                        msg += f"  ctr={running['contrastive_loss']:.4f}"
                    print(msg)
                    if self.use_wandb:
                        wandb.log({"train/"+k: v for k, v in running.items()},
                                  step=self.global_step)
                        wandb.log({"train/lr": lr}, step=self.global_step)
                    running = {}

                # ── mid-epoch checkpoint ────────────────────────────────
                if self.global_step % self.save_every == 0:
                    val_loss = self._validate()
                    print(f"  → mid-epoch val_loss: {val_loss:.4f}")
                    self.save_checkpoint(tag=f"step{self.global_step}")
                    if self.use_wandb:
                        wandb.log({"val/lm_loss": val_loss}, step=self.global_step)

            # ── end of epoch ────────────────────────────────────────────
            elapsed  = time.time() - epoch_start
            val_loss = self._validate()
            bleu1    = self._epoch_bleu1()

            print(f"\n{'='*65}")
            print(f"Epoch {epoch}/{self.num_epochs}  |  {elapsed/60:.1f} min")
            print(f"  val_loss : {val_loss:.4f}")
            print(f"  BLEU-1   : {bleu1:.4f}" if not math.isnan(bleu1)
                  else "  BLEU-1   : (tokenizer not set)")
            print(f"{'='*65}\n")

            self.save_checkpoint(tag=f"epoch{epoch}")
            if self.use_wandb:
                wandb.log({"val/lm_loss": val_loss, "val/bleu1": bleu1},
                          step=self.global_step)

        self.save_checkpoint(tag="final")
        if self.use_wandb:
            wandb.finish()
        print("Training complete ✓")
