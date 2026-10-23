"""CTC Decoder, character tokenizer, and confidence estimator."""

from typing import List, Tuple
import torch
import numpy as np


class CTCTokenizer:
    """
    Bi-directional character-to-integer mapping for OCR.
    Index 0 is reserved for CTC blank token.
    """

    def __init__(self, vocab: str = "0123456789.,-/: ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        self.vocab = vocab
        self.blank_idx = 0
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.num_classes = len(vocab) + 1

    def encode(self, text: str) -> List[int]:
        """Convert string to list of integer tokens."""
        text = text.upper()
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode_greedy(
        self,
        log_probs: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Greedy CTC decoding for a batch of predictions.
        log_probs: (Time, Batch, num_classes)
        Returns: list of (decoded_text, confidence_score) for each batch element.
        """
        # (Time, Batch, num_classes) -> (Batch, Time, num_classes)
        probs = torch.exp(log_probs).permute(1, 0, 2).detach().cpu().numpy()
        batch_size, time_steps, _ = probs.shape
        results = []

        for b in range(batch_size):
            b_probs = probs[b]  # (Time, num_classes)
            max_indices = np.argmax(b_probs, axis=1)  # (Time,)
            max_confs = np.max(b_probs, axis=1)

            # Collapse duplicate tokens and remove blank (0)
            decoded_chars = []
            emitted_confs = []
            prev_idx = -1

            for t in range(time_steps):
                idx = max_indices[t]
                if idx != prev_idx and idx != self.blank_idx:
                    char = self.idx_to_char.get(idx, "")
                    if char:
                        decoded_chars.append(char)
                        emitted_confs.append(float(max_confs[t]))
                prev_idx = idx

            text = "".join(decoded_chars)
            # Confidence is mean of emitted character probabilities, or overall mean if empty
            conf = float(np.mean(emitted_confs)) if emitted_confs else 0.0
            results.append((text, round(conf, 3)))

        return results
