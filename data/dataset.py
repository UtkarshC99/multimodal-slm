"""
data/dataset.py
---------------
Dataset loaders for COCO Captions and Conceptual Captions.

Two concrete Dataset classes are provided:
  - COCOCaptionsDataset  : local COCO annotations + images
  - ConceptualCaptionsDataset : streaming/local CC3M TSV

Plus a `build_dataloader` factory that handles collation.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer, SiglipProcessor


# ---------------------------------------------------------------------------
# Shared image preprocessing helper
# ---------------------------------------------------------------------------

def get_processor(vision_model_name: str = "openai/clip-vit-base-patch16"):
    """Return the image processor associated with the chosen vision encoder."""
    if "siglip" in vision_model_name.lower():
        return SiglipProcessor.from_pretrained(vision_model_name)
    return CLIPProcessor.from_pretrained(vision_model_name)


# ---------------------------------------------------------------------------
# COCO Captions Dataset
# ---------------------------------------------------------------------------

class COCOCaptionsDataset(Dataset):
    """
    COCO Captions dataset for alignment pre-training.

    Expected directory layout:
        coco_root/
            annotations/
                captions_train2017.json
                captions_val2017.json
            train2017/   (images)
            val2017/     (images)

    Download:
        wget http://images.cocodataset.org/zips/train2017.zip
        wget http://images.cocodataset.org/zips/val2017.zip
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    Parameters
    ----------
    coco_root : str | Path
    split     : "train" | "val"
    tokenizer : HuggingFace tokenizer for the LLM
    processor : vision processor (output of `get_processor`)
    max_length : int  maximum token length for captions
    max_samples : int  cap dataset size (useful for quick smoke-tests)
    """

    def __init__(
        self,
        coco_root: str,
        split: str = "train",
        tokenizer=None,
        processor=None,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(coco_root)
        self.split = split
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        year = "2017"
        ann_file = self.root / "annotations" / f"captions_{split}{year}.json"
        self.image_dir = self.root / f"{split}{year}"

        print(f"Loading COCO {split} annotations from {ann_file} …")
        with open(ann_file) as f:
            coco = json.load(f)

        # Build id → filename map
        id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        # Flatten: one (image_path, caption) pair per annotation
        self.samples: List[Tuple[Path, str]] = []
        for ann in coco["annotations"]:
            img_path = self.image_dir / id2file[ann["image_id"]]
            self.samples.append((img_path, ann["caption"]))

        if max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"  → {len(self.samples):,} image-caption pairs loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, caption = self.samples[idx]

        # ---- Image ----
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # ---- Caption ----
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: same as input_ids but pad tokens → -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption,  # keep raw string for qualitative inspection
        }


# ---------------------------------------------------------------------------
# Conceptual Captions Dataset  (TSV format)
# ---------------------------------------------------------------------------

class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions 3M (CC3M) dataset.

    Expected format: a TSV file with columns [caption, url]
    Images must be pre-downloaded locally (see scripts/download_cc3m.py scaffold).

    Parameters
    ----------
    tsv_path   : path to Train_GCC-training.tsv or similar
    image_dir  : directory where images are stored as <sha256>.jpg (or index.jpg)
    tokenizer  : LLM tokenizer
    processor  : vision processor
    max_length : int
    max_samples: Optional[int]
    """

    def __init__(
        self,
        tsv_path: str,
        image_dir: str,
        tokenizer=None,
        processor=None,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        print(f"Scanning CC3M TSV: {tsv_path} …")
        self.samples: List[Tuple[Path, str]] = []
        with open(tsv_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                caption, url = parts[0], parts[1]
                # Assume image saved as {i:08d}.jpg
                img_path = self.image_dir / f"{i:08d}.jpg"
                if img_path.exists():
                    self.samples.append((img_path, caption))
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f"  → {len(self.samples):,} locally available image-caption pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption,
        }


# ---------------------------------------------------------------------------
# VQA Dataset  (for evaluation only)
# ---------------------------------------------------------------------------

class VQAv2Dataset(Dataset):
    """
    VQAv2 evaluation dataset.
    Used only for inference / accuracy measurement, not training.

    Expected files (download from https://visualqa.org/download.html):
        vqa_root/
            v2_OpenEnded_mscoco_val2014_questions.json
            v2_mscoco_val2014_annotations.json
            val2014/  (COCO val2014 images)
    """

    def __init__(
        self,
        vqa_root: str,
        processor=None,
        tokenizer=None,
        max_q_length: int = 64,
        max_samples: Optional[int] = None,
    ):
        self.vqa_root = Path(vqa_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_q_length = max_q_length

        q_file = self.vqa_root / "v2_OpenEnded_mscoco_val2014_questions.json"
        a_file = self.vqa_root / "v2_mscoco_val2014_annotations.json"

        with open(q_file) as f:
            questions = {q["question_id"]: q for q in json.load(f)["questions"]}
        with open(a_file) as f:
            annotations = json.load(f)["annotations"]

        self.samples = []
        for ann in annotations:
            qid = ann["question_id"]
            q = questions[qid]
            img_id = str(q["image_id"]).zfill(12)
            img_path = self.vqa_root / "val2014" / f"COCO_val2014_{img_id}.jpg"
            answer = ann["multiple_choice_answer"]
            self.samples.append((img_path, q["question"], answer))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"VQAv2 val: {len(self.samples):,} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, question, answer = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        prompt = f"Question: {question} Answer:"
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_q_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "question": question,
            "answer": answer,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader with sensible defaults for Colab T4."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
