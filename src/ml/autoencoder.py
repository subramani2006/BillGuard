"""Deep Autoencoder Anomaly Detector in PyTorch initialized strictly from scratch."""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from .features import FEATURE_NAMES


class PyTorchAutoencoderNet(nn.Module):
    """Deep bottleneck Autoencoder initialized strictly with random weights."""

    def __init__(self, input_dim: int = 8, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Strict random weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class BillAutoencoderDetector:
    """
    Autoencoder anomaly detector trained solely on normal consumption patterns.
    Reconstruction error serves as the anomaly score and feature-level explanation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        latent_dim: int = 4,
        threshold_percentile: float = 94.0,
        device_str: str = "cpu",
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.device = torch.device(device_str)
        self.net = PyTorchAutoencoderNet(input_dim, latent_dim).to(self.device)
        self.threshold: float = 1.5
        self.is_fitted: bool = False

    def fit(
        self,
        X_train_normal: np.ndarray,
        X_val_normal: Optional[np.ndarray] = None,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.005,
    ) -> List[float]:
        """
        Train autoencoder to minimize reconstruction error on normal samples.
        """
        self.net.train()
        tensor_x = torch.from_numpy(X_train_normal).float()
        ds = TensorDataset(tensor_x)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        loss_history = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.net(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(loader))
            loss_history.append(avg_loss)

        # Calibrate anomaly threshold on validation (or train) set
        eval_x = X_val_normal if X_val_normal is not None else X_train_normal
        recon_errors = self.get_reconstruction_errors(eval_x)
        self.threshold = float(np.percentile(recon_errors, self.threshold_percentile))
        self.is_fitted = True

        return loss_history

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error per sample."""
        self.net.eval()
        tensor_x = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            recon = self.net(tensor_x)
            diff = (tensor_x - recon) ** 2
            mse = torch.mean(diff, dim=1).cpu().numpy()
        return mse

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized anomaly score in [0, 1] relative to calibrated threshold."""
        errors = self.get_reconstruction_errors(X)
        norm_scores = 1.0 / (1.0 + np.exp(-3.0 * (errors - self.threshold) / max(0.1, self.threshold)))
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly prediction: 1 for anomaly, 0 for normal."""
        errors = self.get_reconstruction_errors(X)
        return (errors > self.threshold).astype(np.int32)

    def explain_instance(
        self,
        x_single: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Explain anomaly via per-feature squared reconstruction errors.
        Features with the largest reconstruction error contributed most to the anomaly verdict.
        """
        self.net.eval()
        names = feature_names or FEATURE_NAMES
        tensor_x = torch.from_numpy(x_single.reshape(1, -1)).float().to(self.device)

        with torch.no_grad():
            recon = self.net(tensor_x)
            sq_err = ((tensor_x - recon) ** 2).squeeze(0).cpu().numpy()

        total_err = float(np.sum(sq_err)) + 1e-6
        attributions = []

        for name, err in zip(names, sq_err):
            pct = (float(err) / total_err) * 100.0
            attributions.append({
                "feature": name,
                "reconstruction_error": round(float(err), 4),
                "contribution_pct": round(pct, 1),
            })

        attributions.sort(key=lambda x: x["contribution_pct"], reverse=True)
        return attributions

    def save(self, filepath: str) -> None:
        """Save model weights and threshold."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }, p)

    def load(self, filepath: str) -> None:
        """Load trained autoencoder weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.threshold = checkpoint.get("threshold", 1.5)
        self.net.eval()
        self.is_fitted = True
