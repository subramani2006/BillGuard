"""OCR Evaluation Metrics: Character Error Rate (CER) and Word Error Rate (WER)."""

from typing import List, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    CER = (Substitutions + Insertions + Deletions) / Total Reference Characters
    """
    total_dist = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_clean = ref.strip().upper()
        hyp_clean = hyp.strip().upper()
        total_dist += levenshtein_distance(ref_clean, hyp_clean)
        total_chars += max(1, len(ref_clean))

    return round(total_dist / max(1, total_chars), 4)


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Word Error Rate (WER).
    WER = Edit Distance on Words / Total Reference Words
    """
    total_dist = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().upper().split()
        hyp_words = hyp.strip().upper().split()

        # Word-level Levenshtein
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,       # deletion
                    d[i][j - 1] + 1,       # insertion
                    d[i - 1][j - 1] + cost # substitution
                )

        total_dist += d[len(ref_words)][len(hyp_words)]
        total_words += max(1, len(ref_words))

    return round(total_dist / max(1, total_words), 4)


def compute_exact_accuracy(references: List[str], hypotheses: List[str]) -> float:
    """Compute exact string match accuracy."""
    matches = sum(1 for r, h in zip(references, hypotheses) if r.strip().upper() == h.strip().upper())
    return round(matches / max(1, len(references)), 4)
