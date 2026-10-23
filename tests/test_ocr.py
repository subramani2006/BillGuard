"""Unit tests for OCR model, CTC decoding, and CER/WER metrics."""

import torch
import pytest
from PIL import Image

from src.ocr.model import CRNNOcrModel
from src.ocr.decoder import CTCTokenizer
from src.ocr.metrics import compute_cer, compute_wer, compute_exact_accuracy
from src.ocr.extractor import BillOcrExtractor


def test_ctc_tokenizer_roundtrip():
    tokenizer = CTCTokenizer()
    text = "CONSUMER: 09-245-001"
    tokens = tokenizer.encode(text)
    assert len(tokens) == len(text)
    assert all(t > 0 for t in tokens)


def test_crnn_forward_shape():
    tokenizer = CTCTokenizer()
    model = CRNNOcrModel(num_classes=tokenizer.num_classes)
    # Batch=2, Channel=1, Height=32, Width=256
    dummy_input = torch.randn(2, 1, 32, 256)
    out = model(dummy_input)
    # Output shape should be (Time=64, Batch=2, num_classes)
    assert out.shape == (64, 2, tokenizer.num_classes)


def test_cer_and_wer_metrics():
    refs = ["TANGEDCO", "100 UNITS"]
    hyps = ["TANGEDC0", "100 UNITS"]  # 1 char substitution ('O' -> '0') in first word

    cer = compute_cer(refs, hyps)
    wer = compute_wer(refs, hyps)
    acc = compute_exact_accuracy(refs, hyps)

    assert 0.0 < cer < 0.20
    assert wer == 0.3333  # 1 word wrong out of 3 words (1/3 = 0.3333)
    assert acc == 0.50    # 1 sentence exact match out of 2


def test_ocr_extractor_fallback():
    extractor = BillOcrExtractor(confidence_threshold=0.85)
    dummy_bill = Image.new("RGB", (600, 750), color=(250, 250, 250))
    res = extractor.extract_from_bill_image(dummy_bill)

    assert "fields" in res
    assert "confidences" in res
    assert "needs_manual_verification" in res
    assert "crop_images" in res
    # With a blank image, extractor must gracefully flag for manual verification
    assert res["needs_manual_verification"] is True
