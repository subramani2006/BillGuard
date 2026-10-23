"""CRNN (CNN + Bidirectional GRU + CTC) Architecture initialized strictly from scratch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNOcrModel(nn.Module):
    """
    CRNN architecture for sequence recognition trained strictly from random initialization.
    No pretrained weights or external checkpoints used.
    """

    def __init__(self, num_classes: int, hidden_size: int = 64):
        super().__init__()
        self.num_classes = num_classes  # Includes blank token at index 0

        # CNN Feature Extractor
        # Input shape: (Batch, 1, Height=32, Width=256)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, 16, 128)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 8, 64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 128, 4, 64)

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # (B, 128, 1, 64)
        )

        # Bidirectional Recurrent Layer
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

        # Linear classification head to character vocabulary
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # Explicit Random Weight Initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Strict random weight initialization (Kaiming Normal for CNN, Orthogonal for RNN)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, 1, 32, 256)
        Returns log_probs: (Time=64, Batch, num_classes) suitable for nn.CTCLoss
        """
        features = self.cnn(x)  # (Batch, 128, 1, 64)
        features = features.squeeze(2)  # (Batch, 128, 64)
        features = features.permute(0, 2, 1)  # (Batch, Time=64, Channels=128)

        rnn_out, _ = self.rnn(features)  # (Batch, 64, 128)
        logits = self.fc(rnn_out)  # (Batch, 64, num_classes)

        # PyTorch CTCLoss expects (Time, Batch, num_classes)
        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        return log_probs
