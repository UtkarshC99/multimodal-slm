"""
training/trainer.py
-------------------
Training loop for the vision-language adapter.

Features:
  - Mixed precision (fp16) via torch.cuda.amp
  - Gradient accumulation (to simulate larger batches on limited VRAM)
  - Warmup + cosine LR schedule
  - Periodic validation loss tracking
  - Checkpoint saving / resuming
  - Optional contrastive loss term alongside the primary LM loss
  - WandB logging (gracefully skipped if not installed)
"""

import os
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
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
    Symmetric InfoNCE loss pulling projected visual embeddings close to their
    paired text embeddings and away from unpaired ones.

    Parameters
    ----------
    visual_embeds : (B, D) – mean-pooled projected vision tokens
    text_embeds   : (B, D) – mean-pooled LLM text token embeddings
    """
    visual_embeds = F.normalize(visual_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = (visual_embeds @ text_embeds.T) / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.T, labels)
    return (loss_v2t + loss_t2v) / 2


def l2_alignment_loss(
    visual_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Simple L2 distance between mean-pooled visual and text embeddings.
    Useful as an auxiliary regulariser.
    """
    v = visual_embeds.mean(dim=1) if visual_embeds.dim() == 3 else visual_embeds
    t = text_embeds.mean(dim=1) if text_embeds.dim() == 3 else text_embeds
    return F.mse_loss(F.normalize(v, dim=-1), F.normalize(t, dim=-1))


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains the VisionLanguageModel adapter.

    Only the adapter parameters receive gradients. The vision encoder and LLM
    remain frozen throughout.

    Parameters
    ----------
    model            : VisionLanguageModel instance
    train_loader     : DataLoader for training split
    val_loader       : DataLoader for validation split (optional)
    output_dir       : where to save checkpoints
    num_epochs       : total training epochs
    learning_rate    : peak LR for AdamW
    weight_decay     : L2 regularisation
    warmup_ratio     : fraction of total steps used for LR warmup
    grad_accum_steps : accumulate gradients over N mini-batches before stepping
    max_grad_norm    : gradient clipping threshold
    use_contrastive  : add contrastive loss term
    contrastive_weight : weight on the contrastive term (default 0.1)
    use_l2           : add L2 alignment auxiliary loss
    l2_weight        : weight on the L2 term (default 0.01)
    log_every        : log metrics every N optimiser steps
    save_every       : save checkpoint every N optimiser steps
    use_wandb        : enable Weights & Biases logging
    wandb_project    : W&B project name
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = "checkpoints",
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_contrastive: bool = False,
        contrastive_weight: float = 0.1,
        use_l2: bool = False,
        l2_weight: float = 0.01,
        log_every: int = 10,
        save_every: int = 500,
        use_wandb: bool = False,
        wandb_project: str = "multimodal-slm",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.use_l2 = use_l2
        self.l2_weight = l2_weight
        self.log_every = log_every
        self.save_every = save_every

        # Optimiser: only adapter parameters
        trainable = model.trainable_parameters()
        print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
        self.optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

        # Scheduler
        total_steps = (len(train_loader) // grad_accum_steps) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        # Mixed precision
        self.scaler = GradScaler()

        # W&B
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=wandb_project, config={
                "lr": learning_rate, "epochs": num_epochs,
                "grad_accum": grad_accum_steps,
                "contrastive": use_contrastive,
            })

        self.global_step = 0

    # ------------------------------------------------------------------
    def _forward_batch(self, batch: Dict[str, Any]):
        """Run one forward pass and return scalar loss + component dict."""
        device = next(self.model.adapter.parameters()).device

        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with autocast():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss

            losses = {"lm_loss": lm_loss.item()}
            total_loss = lm_loss

            # ---- Optional contrastive loss --------------------------------
            if self.use_contrastive:
                # Projected visual embeddings (mean-pool over visual tokens)
                vis_emb = self.model.encode_images(pixel_values).mean(dim=1)  # (B, D)
                # Mean-pool over non-padding text embeddings
                text_emb = self.model.lm.get_input_embeddings()(input_ids)
                mask = attention_mask.unsqueeze(-1).float()
                text_emb = (text_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
                c_loss = contrastive_loss(vis_emb, text_emb.float())
                total_loss = total_loss + self.contrastive_weight * c_loss
                losses["contrastive_loss"] = c_loss.item()

            # ---- Optional L2 alignment loss --------------------------------
            if self.use_l2:
                vis_emb = self.model.encode_images(pixel_values).mean(dim=1)
                text_emb = self.model.lm.get_input_embeddings()(input_ids)
                mask = attention_mask.unsqueeze(-1).float()
                text_emb = (text_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
                a_loss = l2_alignment_loss(vis_emb.unsqueeze(1), text_emb.unsqueeze(1))
                total_loss = total_loss + self.l2_weight * a_loss
                losses["l2_loss"] = a_loss.item()

            losses["total_loss"] = total_loss.item()

        return total_loss, losses

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _validate(self) -> float:
        """Compute mean validation loss (LM only)."""
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        device = next(self.model.adapter.parameters()).device
        total, count = 0.0, 0

        for batch in self.val_loader:
            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast():
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total += outputs.loss.item()
            count += 1

        self.model.train()
        return total / max(1, count)

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str = "latest"):
        """Save only the adapter weights (the rest are frozen/pretrained)."""
        path = self.output_dir / f"adapter_{tag}.pt"
        torch.save({
            "adapter_state_dict": self.model.adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, path)
        print(f"  ✓ Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        """Resume training from a saved checkpoint."""
        ckpt = torch.load(path, map_location="cpu")
        self.model.adapter.load_state_dict(ckpt["adapter_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        print(f"  ✓ Resumed from step {self.global_step}")

    # ------------------------------------------------------------------
    def train(self):
        """Main training loop."""
        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            running = {"total_loss": 0.0, "lm_loss": 0.0}
            accum_count = 0

            for step, batch in enumerate(self.train_loader, 1):
                loss, loss_dict = self._forward_batch(batch)

                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()

                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v / self.grad_accum_steps

                accum_count += 1
                if accum_count < self.grad_accum_steps:
                    continue

                # --- Optimiser step ---
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                accum_count = 0

                # --- Logging ---
                if self.global_step % self.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    msg = (
                        f"[Epoch {epoch}/{self.num_epochs}] "
                        f"step={self.global_step}  "
                        f"lm_loss={running.get('lm_loss', 0):.4f}  "
                        f"total_loss={running.get('total_loss', 0):.4f}  "
                        f"lr={lr:.2e}"
                    )
                    print(msg)
                    if self.use_wandb:
                        wandb.log({"train/" + k: v for k, v in running.items()}, step=self.global_step)
                        wandb.log({"train/lr": lr}, step=self.global_step)
                    running = {}

                # --- Periodic checkpoint ---
                if self.global_step % self.save_every == 0:
                    val_loss = self._validate()
                    print(f"  → val_loss: {val_loss:.4f}")
                    self.save_checkpoint(tag=f"step{self.global_step}")
                    if self.use_wandb:
                        wandb.log({"val/lm_loss": val_loss}, step=self.global_step)

            elapsed = time.time() - epoch_start
            val_loss = self._validate()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} complete  |  {elapsed/60:.1f} min  |  val_loss={val_loss:.4f}")
            print(f"{'='*60}\n")
            self.save_checkpoint(tag=f"epoch{epoch}")

        self.save_checkpoint(tag="final")
        if self.use_wandb:
            wandb.finish()
        print("Training complete ✓")
