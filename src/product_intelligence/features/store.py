"""A minimal, point-in-time-correct feature store (extension).

Captures the two properties that make a feature store more than a table:
* **Online retrieval** - latest feature vector per entity for low-latency serving.
* **Point-in-time-correct historical retrieval** - an as-of join that returns the
  feature values *as they were known* at each label timestamp, preventing the
  data leakage that naive joins cause in training sets.

File-backed (CSV) so it runs with no infra; the interface mirrors Feast closely
enough that swapping in a real store is mechanical.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from product_intelligence.core.logging import get_logger

logger = get_logger(__name__)


class FeatureStore:
    def __init__(
        self, root: str | Path, entity_key: str = "account_id", ts_key: str = "event_timestamp"
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entity_key = entity_key
        self.ts_key = ts_key
        self.offline_path = self.root / "offline.csv"
        self.online_path = self.root / "online.csv"

    def materialize(self, df: pd.DataFrame) -> int:
        """Persist the offline history and refresh the online (latest) snapshot."""
        frame = df.copy()
        if self.ts_key not in frame.columns:
            frame[self.ts_key] = pd.Timestamp.utcnow().isoformat()
        frame.to_csv(self.offline_path, index=False)

        ordered = frame.sort_values(self.ts_key)
        online = ordered.groupby(self.entity_key, as_index=False).tail(1)
        online.to_csv(self.online_path, index=False)
        logger.info("materialized %d rows; %d online entities", len(frame), len(online))
        return len(frame)

    def get_online_features(
        self, entity_ids: list[str], features: list[str] | None = None
    ) -> pd.DataFrame:
        if not self.online_path.exists():
            raise FileNotFoundError("Online store empty; call materialize() first.")
        online = pd.read_csv(self.online_path)
        result = online[online[self.entity_key].isin(entity_ids)]
        if features:
            cols = [self.entity_key, *features]
            result = result[[c for c in cols if c in result.columns]]
        return result.reset_index(drop=True)

    def get_historical_features(
        self, entity_df: pd.DataFrame, features: list[str] | None = None
    ) -> pd.DataFrame:
        """As-of join: for each (entity, timestamp) return the latest feature row
        with ``event_timestamp <= timestamp`` (no future leakage)."""
        if not self.offline_path.exists():
            raise FileNotFoundError("Offline store empty; call materialize() first.")
        offline = pd.read_csv(self.offline_path)
        offline[self.ts_key] = pd.to_datetime(offline[self.ts_key], utc=True, errors="coerce")
        entity_df = entity_df.copy()
        entity_df[self.ts_key] = pd.to_datetime(entity_df[self.ts_key], utc=True, errors="coerce")

        left = entity_df.sort_values(self.ts_key)
        right = offline.sort_values(self.ts_key)
        joined = pd.merge_asof(
            left, right, on=self.ts_key, by=self.entity_key, direction="backward"
        )
        if features:
            keep = [self.entity_key, self.ts_key, *features]
            joined = joined[[c for c in keep if c in joined.columns]]
        return joined.reset_index(drop=True)
