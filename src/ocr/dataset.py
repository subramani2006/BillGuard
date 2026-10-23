"""PyTorch Dataset and Collator for OCR training with CTC Loss."""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .decoder import CTCTokenizer


class BillCropDataset(Dataset):
    """
    Dataset of cropped field images and ground truth text labels.
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        img_dir: Optional[str] = None,
        tokenizer: Optional[CTCTokenizer] = None,
        img_height: int = 32,
        img_width: int = 256,
    ):
        self.samples = samples
        self.img_dir = Path(img_dir) if img_dir else None
        self.tokenizer = tokenizer or CTCTokenizer()
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[int]]:
        item = self.samples[idx]
        label = item["label"]

        if self.img_dir and "filename" in item:
            img_path = self.img_dir / item["filename"]
            img = Image.open(img_path).convert("L")
        elif "image" in item:
            img = item["image"].convert("L")
        else:
            # Fallback blank
            img = Image.new("L", (self.img_width, self.img_height), color=255)

        # Resize to fixed dimension
        img = img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        arr = (arr - 0.5) / 0.5  # [-1, 1]

        tensor_img = torch.from_numpy(arr).unsqueeze(0)  # (1, 32, 256)
        token_ids = self.tokenizer.encode(label)

        return tensor_img, label, token_ids


def ctc_collate_fn(batch: List[Tuple[torch.Tensor, str, List[int]]]):
    """
    Collate function to prepare variable-length target sequences for nn.CTCLoss.
    Returns:
        images: Tensor (Batch, 1, 32, 256)
        targets: 1D Tensor of concatenated target token IDs
        input_lengths: 1D Tensor of length 64 per sample
        target_lengths: 1D Tensor of lengths of ground-truth sequences
        raw_labels: List of original text strings
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    raw_labels = [item[1] for item in batch]
    token_lists = [item[2] for item in batch]

    target_lengths = torch.tensor([len(t) for t in token_lists], dtype=torch.long)
    flat_targets = []
    for t in token_lists:
        flat_targets.extend(t)

    targets = torch.tensor(flat_targets, dtype=torch.long)
    # Architecture produces Time=64 features for Width=256
    input_lengths = torch.full(size=(len(batch),), fill_value=64, dtype=torch.long)

    return images, targets, input_lengths, target_lengths, raw_labels
