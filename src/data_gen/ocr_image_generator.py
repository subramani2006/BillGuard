"""Synthetic Bill & Field-Crop Image Generator for OCR Training from Scratch."""

import os
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np


class BillImageGenerator:
    """
    Renders realistic synthetic Indian (TANGEDCO) electricity bills and field crops.
    Applies synthetic paper textures, noise, rotations, and thermal/dot-matrix effects.
    Strictly self-contained: relies solely on Pillow and NumPy.
    """

    DEFAULT_VOCAB = "0123456789.,-/: ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self, vocab: Optional[str] = None, seed: int = 42):
        self.vocab = vocab or self.DEFAULT_VOCAB
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _get_font(self, size: int = 16):
        """Retrieve default or system monospace/truetype font gracefully."""
        # Try common Windows/system fonts, fallback to default bitmap
        candidate_fonts = [
            "consola.ttf", "arial.ttf", "cour.ttf", "calibri.ttf",
            "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arial.ttf"
        ]
        for f in candidate_fonts:
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def generate_field_crop(
        self,
        text: str,
        target_height: int = 32,
        target_width: int = 256,
        apply_augmentation: bool = True
    ) -> Image.Image:
        """
        Generate a single line/field image crop (e.g., '09-245-001', '4520', '1330.00').
        Ideal for CRNN CTC sequence recognition training.
        """
        text = text.upper().strip()
        # Filter characters to vocabulary
        text = "".join([c for c in text if c in self.vocab])
        if not text:
            text = "0"

        # Background color: off-white paper tones
        bg_val = random.randint(230, 255) if apply_augmentation else 255
        img = Image.new("L", (target_width, target_height), color=bg_val)
        draw = ImageDraw.Draw(img)

        # Select font size appropriate for box
        font_size = random.randint(14, 20) if apply_augmentation else 18
        font = self._get_font(font_size)

        # Compute text position
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(text) * 10
            text_h = 16

        # Position with slight random padding
        x_pad = max(2, (target_width - text_w) // 2) if text_w < target_width else 4
        if apply_augmentation:
            x = random.randint(4, max(5, x_pad))
            y = max(1, random.randint(2, max(3, (target_height - text_h) // 2)))
        else:
            x = x_pad
            y = max(2, (target_height - text_h) // 2)

        # Ink color: dark grey or black with variations
        ink_color = random.randint(0, 45) if apply_augmentation else 0
        draw.text((x, y), text, fill=ink_color, font=font)

        # Optional augmentations: noise, slight blur, faint horizontal ruled lines
        if apply_augmentation:
            # Add salt & pepper noise
            arr = np.array(img)
            noise_prob = random.uniform(0.002, 0.015)
            mask_salt = np.random.rand(*arr.shape) < noise_prob
            mask_pepper = np.random.rand(*arr.shape) < noise_prob
            arr[mask_salt] = 255
            arr[mask_pepper] = 0
            img = Image.fromarray(arr)

            # Random slight blur
            if random.random() < 0.25:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

            # Faint background rule line
            if random.random() < 0.3:
                draw_aug = ImageDraw.Draw(img)
                line_y = random.choice([2, target_height - 3])
                draw_aug.line([(0, line_y), (target_width, line_y)], fill=random.randint(180, 220), width=1)

        return img

    def generate_full_bill_image(
        self,
        consumer_no: str,
        consumer_name: str,
        bill_date: str,
        prev_date: str,
        prev_reading: float,
        curr_reading: float,
        units: float,
        net_amount: float,
        arrears: float = 0.0,
        due_date: str = "2024-08-30",
        tariff_name: str = "TN DOMESTIC LT-1A",
    ) -> Tuple[Image.Image, Dict[str, Tuple[int, int, int, int]]]:
        """
        Generate a complete synthetic TANGEDCO receipt bill with field bounding boxes.
        Returns the rendered Image and a dict of bounding boxes {field_name: (x1, y1, x2, y2)}.
        """
        w, h = 600, 750
        img = Image.new("RGB", (w, h), color=(250, 249, 245))
        draw = ImageDraw.Draw(img)

        # Fonts
        title_font = self._get_font(18)
        header_font = self._get_font(14)
        body_font = self._get_font(13)
        bold_font = self._get_font(14)
        small_font = self._get_font(11)

        # Outer border
        draw.rectangle([(15, 15), (w - 15, h - 15)], outline=(60, 60, 60), width=2)
        draw.rectangle([(18, 18), (w - 18, h - 18)], outline=(180, 180, 180), width=1)

        # Header Banner
        draw.text((w // 2 - 210, 30), "TAMIL NADU GENERATION AND DISTRIBUTION CORP LTD", fill=(10, 30, 80), font=title_font)
        draw.text((w // 2 - 130, 55), "TANGEDCO - DOMESTIC BILL CARD", fill=(20, 20, 20), font=header_font)
        draw.text((w // 2 - 110, 75), f"TARIFF: {tariff_name}", fill=(80, 80, 80), font=small_font)
        draw.line([(25, 95), (w - 25, 95)], fill=(100, 100, 100), width=2)

        # Store bounding boxes of key fields for OCR cropping & ground truth
        boxes: Dict[str, Tuple[int, int, int, int]] = {}

        # Consumer Details Section
        y = 115
        draw.text((35, y), "Consumer No:", fill=(30, 30, 30), font=bold_font)
        draw.text((180, y), consumer_no, fill=(0, 0, 0), font=bold_font)
        boxes["consumer_number"] = (175, y - 2, 360, y + 22)

        draw.text((380, y), "Bill Date:", fill=(30, 30, 30), font=body_font)
        draw.text((470, y), bill_date, fill=(0, 0, 0), font=body_font)
        boxes["billing_date"] = (465, y - 2, 570, y + 20)

        y += 30
        draw.text((35, y), "Name:", fill=(30, 30, 30), font=body_font)
        draw.text((180, y), consumer_name, fill=(0, 0, 0), font=body_font)

        draw.text((380, y), "Due Date:", fill=(30, 30, 30), font=body_font)
        draw.text((470, y), due_date, fill=(180, 0, 0), font=bold_font)
        boxes["due_date"] = (465, y - 2, 570, y + 20)

        # Meter Reading Table
        y += 45
        draw.line([(25, y), (w - 25, y)], fill=(160, 160, 160), width=1)
        y += 8
        draw.text((35, y), "METER READING DETAILS (BI-MONTHLY CYCLE)", fill=(10, 30, 80), font=bold_font)
        y += 24

        # Table Header
        draw.rectangle([(25, y), (w - 25, y + 28)], fill=(230, 235, 245), outline=(150, 150, 150))
        draw.text((35, y + 6), "Prev Date", fill=(20, 20, 20), font=bold_font)
        draw.text((140, y + 6), "Curr Date", fill=(20, 20, 20), font=bold_font)
        draw.text((250, y + 6), "Prev Read", fill=(20, 20, 20), font=bold_font)
        draw.text((360, y + 6), "Curr Read", fill=(20, 20, 20), font=bold_font)
        draw.text((480, y + 6), "Units", fill=(20, 20, 20), font=bold_font)

        # Table Row
        y += 28
        draw.rectangle([(25, y), (w - 25, y + 36)], fill=(255, 255, 255), outline=(150, 150, 150))
        draw.text((35, y + 8), prev_date, fill=(0, 0, 0), font=body_font)
        boxes["previous_reading_date"] = (30, y + 2, 135, y + 30)

        draw.text((140, y + 8), bill_date, fill=(0, 0, 0), font=body_font)
        boxes["current_reading_date"] = (138, y + 2, 242, y + 30)

        prev_read_str = f"{prev_reading:.0f}"
        draw.text((255, y + 8), prev_read_str, fill=(0, 0, 0), font=body_font)
        boxes["previous_meter_reading"] = (250, y + 2, 345, y + 30)

        curr_read_str = f"{curr_reading:.0f}"
        draw.text((365, y + 8), curr_read_str, fill=(0, 0, 0), font=body_font)
        boxes["current_meter_reading"] = (360, y + 2, 460, y + 30)

        units_str = f"{units:.0f}"
        draw.text((490, y + 8), units_str, fill=(0, 0, 120), font=bold_font)
        boxes["billed_units"] = (480, y + 2, 565, y + 30)

        # Charges Breakdown Section
        y += 55
        draw.line([(25, y), (w - 25, y)], fill=(160, 160, 160), width=1)
        y += 10
        draw.text((35, y), "ASSESSMENT BREAKDOWN", fill=(10, 30, 80), font=bold_font)

        y += 26
        items = [
            ("Energy Charges (Net of Subsidy):", f"INR {net_amount:.2f}", "billed_energy_charges"),
            ("Fixed / Service Charges:", "INR 0.00", "billed_fixed_charges"),
            ("Electricity Tax / Cess:", "INR 0.00", "billed_tax"),
            ("Arrears / Past Dues:", f"INR {arrears:.2f}", "arrears_amount"),
        ]

        for label, val, key in items:
            draw.text((50, y), label, fill=(40, 40, 40), font=body_font)
            draw.text((420, y), val, fill=(0, 0, 0), font=body_font)
            boxes[key] = (410, y - 2, 560, y + 20)
            y += 24

        # Total Payable Box
        y += 15
        grand_total = net_amount + arrears
        draw.rectangle([(35, y), (w - 35, y + 50)], fill=(235, 245, 235), outline=(50, 120, 50), width=2)
        draw.text((50, y + 14), "NET PAYABLE AMOUNT:", fill=(10, 90, 20), font=bold_font)
        draw.text((390, y + 14), f"INR {grand_total:.2f}", fill=(10, 90, 20), font=title_font)
        boxes["billed_amount"] = (385, y + 6, 560, y + 44)

        # Footer Notes
        y += 75
        draw.text((35, y), "Note: Government of Tamil Nadu provides 100 units free for all domestic consumers.", fill=(80, 80, 80), font=small_font)
        draw.text((35, y + 18), "Please verify meter readings before making online / e-seva payments.", fill=(80, 80, 80), font=small_font)
        draw.text((35, y + 36), "This is a computer-generated bill receipt for LT Tariff 1-A.", fill=(100, 100, 100), font=small_font)

        # Add subtle paper texture & scan noise
        arr = np.array(img)
        noise = np.random.normal(0, 3.5, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        return img, boxes

    def generate_ocr_training_dataset(
        self,
        num_samples: int = 1200,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic text line crops spanning numbers, dates, codes, and amounts.
        Splits into train (70%), val (15%), test (15%).
        """
        crops_data: List[Dict[str, str]] = []

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            img_dir = out_p / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            # Category selection
            cat = random.choice(["number", "currency", "date", "code", "units", "word"])

            if cat == "number":
                val = str(random.randint(0, 999999))
            elif cat == "currency":
                val = f"{random.uniform(0.0, 9999.0):.2f}"
            elif cat == "date":
                day = str(random.randint(1, 28)).zfill(2)
                month = str(random.randint(1, 12)).zfill(2)
                year = random.choice(["2022", "2023", "2024", "2025"])
                sep = random.choice(["-", "/"])
                val = f"{year}{sep}{month}{sep}{day}" if random.random() < 0.5 else f"{day}{sep}{month}{sep}{year}"
            elif cat == "code":
                val = f"09-245-{str(random.randint(1, 999)).zfill(3)}-{str(random.randint(1, 99)).zfill(2)}"
            elif cat == "units":
                val = f"{random.randint(10, 1800)}"
            else:
                val = random.choice([
                    "UNITS", "TANGEDCO", "READING", "PREVIOUS", "CURRENT",
                    "AMOUNT", "CONSUMER", "TARIFF", "ARREARS", "DOMESTIC"
                ])

            # Split assignment
            if i < int(num_samples * 0.70):
                split = "train"
            elif i < int(num_samples * 0.85):
                split = "val"
            else:
                split = "test"

            crop_img = self.generate_field_crop(val, apply_augmentation=(split == "train"))
            filename = f"crop_{i:05d}.png"

            if output_dir:
                crop_img.save(img_dir / filename)

            crops_data.append({
                "filename": filename,
                "label": val,
                "split": split,
                "category": cat
            })

        if output_dir:
            import pandas as pd
            df = pd.DataFrame(crops_data)
            df.to_csv(out_p / "labels.csv", index=False)
            print(f"[OCR Synth] Generated {len(crops_data)} crops in {output_dir}")

        return crops_data
