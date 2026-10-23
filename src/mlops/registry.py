"""Model Registry, Candidate-vs-Production Evaluation Gate, and Rollback System."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class ModelVersionMetadata(BaseModel):
    version: int
    model_name: str
    artifact_path: str
    stage: str = "DEVELOPMENT"  # DEVELOPMENT, CANDIDATE, PRODUCTION, ARCHIVED
    metrics: Dict[str, float] = Field(default_factory=dict)
    registered_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""
    promoted_at: Optional[str] = None
    notes: str = ""


class ModelRegistry:
    """
    Manages model versioning, safety gate promotion criteria, and automated rollback.
    """

    def __init__(self, registry_file: str = "./models/registry.json"):
        self.registry_file = Path(registry_file)
        self.versions: List[ModelVersionMetadata] = []
        self.load_registry()

    def load_registry(self) -> None:
        """Load registry metadata from JSON."""
        if not self.registry_file.exists():
            self._initialize_default_registry()
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.versions = [ModelVersionMetadata(**v) for v in data.get("versions", [])]
        except Exception as e:
            print(f"[Warning] Failed to read registry: {e}. Reinitializing.")
            self._initialize_default_registry()

    def _initialize_default_registry(self) -> None:
        """Initialize registry with baseline Production v1."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        v1 = ModelVersionMetadata(
            version=1,
            model_name="IsolationForest_Scratch",
            artifact_path="./models/isolation_forest.joblib",
            stage="PRODUCTION",
            metrics={"precision": 0.88, "recall": 0.82, "f1_score": 0.85, "roc_auc": 0.91},
            description="Initial production baseline Isolation Forest trained from scratch.",
            promoted_at=datetime.now().isoformat(),
        )
        self.versions = [v1]
        self.save_registry()

    def save_registry(self) -> None:
        """Persist registry metadata to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"versions": [v.model_dump() for v in self.versions]}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_production_model(self) -> Optional[ModelVersionMetadata]:
        """Return the active PRODUCTION model metadata."""
        for v in self.versions:
            if v.stage == "PRODUCTION":
                return v
        return None

    def register_candidate(
        self,
        model_name: str,
        artifact_path: str,
        metrics: Dict[str, float],
        description: str = "",
    ) -> ModelVersionMetadata:
        """Register a new candidate model version."""
        next_version = (max([v.version for v in self.versions], default=0)) + 1
        candidate = ModelVersionMetadata(
            version=next_version,
            model_name=model_name,
            artifact_path=artifact_path,
            stage="CANDIDATE",
            metrics=metrics,
            description=description,
        )
        self.versions.append(candidate)
        self.save_registry()
        return candidate

    def evaluate_promotion_gate(
        self,
        candidate_version: int,
        min_f1_threshold: float = 0.75,
        min_precision_threshold: float = 0.70,
    ) -> Tuple[bool, str]:
        """
        Automated Gatekeeper:
        1. Candidate F1 must satisfy minimum threshold.
        2. Candidate Precision must satisfy minimum threshold to avoid alert fatigue.
        3. Candidate F1 must meet or exceed current production model's F1.
        """
        cand = next((v for v in self.versions if v.version == candidate_version), None)
        if not cand:
            return False, f"Version {candidate_version} not found in registry."

        cand_f1 = cand.metrics.get("f1_score", 0.0)
        cand_prec = cand.metrics.get("precision", 0.0)

        if cand_f1 < min_f1_threshold:
            return False, f"Gate Failed: Candidate F1 ({cand_f1:.3f}) < required threshold ({min_f1_threshold})."

        if cand_prec < min_precision_threshold:
            return False, f"Gate Failed: Candidate Precision ({cand_prec:.3f}) < required threshold ({min_precision_threshold})."

        prod = self.get_production_model()
        if prod:
            prod_f1 = prod.metrics.get("f1_score", 0.0)
            if cand_f1 < prod_f1:
                return False, f"Gate Failed: Candidate F1 ({cand_f1:.3f}) does not match or exceed current production F1 ({prod_f1:.3f})."

        return True, f"Gate Passed: Candidate meets quality criteria (F1: {cand_f1:.3f}, Precision: {cand_prec:.3f})."

    def promote_candidate(self, candidate_version: int) -> Tuple[bool, str]:
        """Promote candidate to PRODUCTION and archive previous production model."""
        can_promote, reason = self.evaluate_promotion_gate(candidate_version)
        if not can_promote:
            return False, reason

        # Archive current production model
        for v in self.versions:
            if v.stage == "PRODUCTION":
                v.stage = "ARCHIVED"
                v.notes += f" Archived on promotion of v{candidate_version}."

        # Promote candidate
        for v in self.versions:
            if v.version == candidate_version:
                v.stage = "PRODUCTION"
                v.promoted_at = datetime.now().isoformat()
                v.notes += f" Promoted to production ({reason})."

        self.save_registry()
        return True, f"Successfully promoted v{candidate_version} to PRODUCTION."

    def rollback_production(self) -> Tuple[bool, str]:
        """
        Roll back active production model to the immediately preceding ARCHIVED model.
        """
        current_prod = self.get_production_model()
        if not current_prod:
            return False, "No active production model found."

        # Find candidate for rollback (highest version among ARCHIVED)
        archived_candidates = [v for v in self.versions if v.stage == "ARCHIVED"]
        if not archived_candidates:
            return False, "No archived model version available for rollback."

        # Sort by version descending
        archived_candidates.sort(key=lambda x: x.version, reverse=True)
        target = archived_candidates[0]

        # Demote current prod
        current_prod.stage = "ARCHIVED"
        current_prod.notes += f" Demoted via rollback on {datetime.now().isoformat()}."

        # Restore target
        target.stage = "PRODUCTION"
        target.promoted_at = datetime.now().isoformat()
        target.notes += f" Restored via rollback from v{current_prod.version}."

        self.save_registry()
        return True, f"Rolled back from v{current_prod.version} to v{target.version}."
