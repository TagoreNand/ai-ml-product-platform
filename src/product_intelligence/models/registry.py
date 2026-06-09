"""A lightweight, file-based model registry.

Gives the project the parts of a model registry that matter for correctness and
auditability without standing up MLflow: immutable, content-addressed versions;
a JSON index; stage promotion (``staging`` -> ``production``); and co-located
metadata + explainer background data. The serving layer resolves
``production`` (or ``latest`` / an explicit version) at load time, so retraining
is a register-and-promote operation rather than a file overwrite.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass
class ModelRecord:
    name: str
    version: str
    stage: str
    created_at: str
    model_type: str
    metrics: dict = field(default_factory=dict)
    threshold: float | None = None
    band_thresholds: dict | None = None
    feature_names: list[str] | None = None
    data_version: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.index_path = self.root / "registry.json"
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- index helpers ----------
    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"models": {}}

    def _save_index(self, index: dict) -> None:
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    # ---------- registration ----------
    def register(
        self,
        name: str,
        pipeline: Any,
        *,
        model_type: str,
        metrics: dict,
        threshold: float | None = None,
        band_thresholds: dict | None = None,
        feature_names: list[str] | None = None,
        data_version: str | None = None,
        extra: dict | None = None,
        background: pd.DataFrame | None = None,
        promote: bool = True,
    ) -> ModelRecord:
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()[:10]
        version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{digest}"

        version_dir = self.models_dir / name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "model.joblib").write_bytes(payload)

        record = ModelRecord(
            name=name,
            version=version,
            stage="production" if promote else "staging",
            created_at=datetime.now(timezone.utc).isoformat(),
            model_type=model_type,
            metrics=metrics,
            threshold=threshold,
            band_thresholds=band_thresholds,
            feature_names=feature_names,
            data_version=data_version,
            extra=extra or {},
        )
        (version_dir / "metadata.json").write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )
        if background is not None:
            background.to_csv(version_dir / "background.csv", index=False)

        index = self._load_index()
        model_entry = index["models"].setdefault(name, {"production": None, "versions": {}})
        model_entry["versions"][version] = record.to_dict()
        if promote:
            model_entry["production"] = version
        self._save_index(index)
        return record

    # ---------- resolution / loading ----------
    def resolve_version(self, name: str, selector: str = "production") -> str:
        index = self._load_index()
        entry = index["models"].get(name)
        if not entry or not entry["versions"]:
            raise FileNotFoundError(f"No registered versions for model '{name}'")
        if selector == "production":
            version = entry.get("production")
            if not version:
                raise FileNotFoundError(f"No production version promoted for '{name}'")
            return version
        if selector == "latest":
            return sorted(entry["versions"])[-1]
        if selector in entry["versions"]:
            return selector
        raise FileNotFoundError(f"Version '{selector}' not found for model '{name}'")

    def _version_dir(self, name: str, version: str) -> Path:
        return self.models_dir / name / version

    def load(self, name: str, selector: str = "production") -> tuple[Any, ModelRecord]:
        version = self.resolve_version(name, selector)
        version_dir = self._version_dir(name, version)
        pipeline = joblib.load(version_dir / "model.joblib")
        record = ModelRecord(**json.loads((version_dir / "metadata.json").read_text("utf-8")))
        return pipeline, record

    def load_background(self, name: str, selector: str = "production") -> pd.DataFrame | None:
        version = self.resolve_version(name, selector)
        path = self._version_dir(name, version) / "background.csv"
        return pd.read_csv(path) if path.exists() else None

    def get_record(self, name: str, selector: str = "production") -> ModelRecord:
        version = self.resolve_version(name, selector)
        version_dir = self._version_dir(name, version)
        return ModelRecord(**json.loads((version_dir / "metadata.json").read_text("utf-8")))

    def list_versions(self, name: str) -> list[dict]:
        index = self._load_index()
        entry = index["models"].get(name, {"versions": {}})
        return sorted(entry["versions"].values(), key=lambda r: r["created_at"], reverse=True)

    def list_models(self) -> list[str]:
        return list(self._load_index()["models"].keys())

    def promote(self, name: str, version: str, stage: str = "production") -> None:
        index = self._load_index()
        entry = index["models"][name]
        entry["versions"][version]["stage"] = stage
        if stage == "production":
            entry["production"] = version
        self._save_index(index)

    def prune(self, name: str, keep: int = 5) -> list[str]:
        """Delete old non-production versions, keeping the newest ``keep``."""
        index = self._load_index()
        entry = index["models"].get(name, {"versions": {}, "production": None})
        versions = sorted(entry["versions"])
        removed = []
        for version in versions[:-keep]:
            if version == entry.get("production"):
                continue
            shutil.rmtree(self._version_dir(name, version), ignore_errors=True)
            entry["versions"].pop(version, None)
            removed.append(version)
        self._save_index(index)
        return removed
