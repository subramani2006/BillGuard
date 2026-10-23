"""OCR Training Pipeline trained strictly from random initialization on synthetic bill data."""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_gen.ocr_image_generator import BillImageGenerator
from .model import CRNNOcrModel
from .decoder import CTCTokenizer
from .dataset import BillCropDataset, ctc_collate_fn
from .metrics import compute_cer, compute_wer, compute_exact_accuracy


def train_ocr_model(
    num_samples: int = 600,
    epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    output_model_path: str = "./models/ocr_crnn_scratch.pth",
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """
    Train CRNN model from random initialization on generated synthetic bill crops.
    Strictly isolated splits: 70% Train, 15% Val, 15% Test.
    """
    device = torch.device(device_str)
    out_dir = Path("./data/ocr_synth")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OCR Training] Generating {num_samples} synthetic bill text crops...")
    img_gen = BillImageGenerator(seed=42)
    crops_metadata = img_gen.generate_ocr_training_dataset(
        num_samples=num_samples,
        output_dir=str(out_dir)
    )

    tokenizer = CTCTokenizer(vocab=img_gen.vocab)

    train_samples = [c for c in crops_metadata if c["split"] == "train"]
    val_samples = [c for c in crops_metadata if c["split"] == "val"]
    test_samples = [c for c in crops_metadata if c["split"] == "test"]

    train_ds = BillCropDataset(train_samples, img_dir=str(out_dir / "images"), tokenizer=tokenizer)
    val_ds = BillCropDataset(val_samples, img_dir=str(out_dir / "images"), tokenizer=tokenizer)
    test_ds = BillCropDataset(test_samples, img_dir=str(out_dir / "images"), tokenizer=tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)

    # Initialize CRNN model with random weights
    model = CRNNOcrModel(num_classes=tokenizer.num_classes)
    model.to(device)

    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"[OCR Training] Starting training from random initialization for {epochs} epochs...")

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for images, targets, input_lengths, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)  # (Time=64, Batch, num_classes)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation step
        model.eval()
        total_val_loss = 0.0
        val_refs = []
        val_hyps = []

        with torch.no_grad():
            for images, targets, input_lengths, target_lengths, raw_labels in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                log_probs = model(images)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_val_loss += loss.item()

                decoded_batch = tokenizer.decode_greedy(log_probs)
                val_hyps.extend([d[0] for d in decoded_batch])
                val_refs.extend(raw_labels)

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        val_cer = compute_cer(val_refs, val_hyps)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val CER: {val_cer:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_cer": val_cer})

    # Final evaluation on isolated test set
    model.eval()
    test_refs = []
    test_hyps = []
    test_confs = []

    with torch.no_grad():
        for images, targets, input_lengths, target_lengths, raw_labels in test_loader:
            images = images.to(device)
            log_probs = model(images)
            decoded_batch = tokenizer.decode_greedy(log_probs)
            test_hyps.extend([d[0] for d in decoded_batch])
            test_confs.extend([d[1] for d in decoded_batch])
            test_refs.extend(raw_labels)

    test_cer = compute_cer(test_refs, test_hyps)
    test_wer = compute_wer(test_refs, test_hyps)
    test_acc = compute_exact_accuracy(test_refs, test_hyps)
    avg_conf = float(sum(test_confs) / max(1, len(test_confs)))

    print("\n--- FINAL TEST SET OCR METRICS (FROM SCRATCH) ---")
    print(f"Test Samples: {len(test_refs)}")
    print(f"Character Error Rate (CER): {test_cer:.4f}")
    print(f"Word Error Rate (WER): {test_wer:.4f}")
    print(f"Exact Sequence Accuracy: {test_acc * 100:.2f}%")
    print(f"Mean Confidence Estimate: {avg_conf:.4f}")

    # Save model checkpoint
    save_p = Path(output_model_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "num_classes": tokenizer.num_classes,
        "test_cer": test_cer,
        "test_wer": test_wer,
        "test_acc": test_acc,
    }, save_p)
    print(f"[OCR] Model saved to {save_p}")

    return {
        "model_path": str(save_p),
        "test_cer": test_cer,
        "test_wer": test_wer,
        "test_acc": test_acc,
        "mean_confidence": avg_conf,
        "history": history,
    }


if __name__ == "__main__":
    train_ocr_model(num_samples=300, epochs=4)
