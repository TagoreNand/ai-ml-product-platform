"""Pluggable data sources (extension: swap synthetic data for warehouse extracts).

A single ``DataSource`` interface decouples the training pipeline from *where*
account data comes from. The default ``synthetic`` source keeps the repo runnable
offline; ``file`` (CSV/Parquet/JSON) and ``sql`` (SQLite/warehouse) sources are
the drop-in path to real data. Selection is config-driven (``DATA_SOURCE``), so
moving from demo to production is an environment change, not a code change.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from product_intelligence.core.config import Settings, settings
from product_intelligence.core.logging import get_logger
from product_intelligence.data.synthetic import generate_accounts

logger = get_logger(__name__)


class DataSource(ABC):
    """Returns a labelled account snapshot as a DataFrame."""

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


class SyntheticDataSource(DataSource):
    def __init__(self, n_samples: int = 5000, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def load(self) -> pd.DataFrame:
        logger.info("loading synthetic data n=%s seed=%s", self.n_samples, self.seed)
        return generate_accounts(n_samples=self.n_samples, seed=self.seed)


class FileDataSource(DataSource):
    """Read a snapshot from CSV / Parquet / JSON (auto-detected by extension)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        suffix = self.path.suffix.lower()
        logger.info("loading file data %s", self.path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(self.path)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(self.path, lines=suffix == ".jsonl")
        return pd.read_csv(self.path)


class SQLDataSource(DataSource):
    """Read a snapshot from a SQL warehouse. SQLite (stdlib) by default.

    For Postgres/Snowflake/BigQuery, pass a SQLAlchemy/DBAPI connection instead -
    the ``query`` contract is unchanged.
    """

    def __init__(self, sqlite_path: str, query: str = "SELECT * FROM accounts") -> None:
        self.sqlite_path = sqlite_path
        self.query = query

    def load(self) -> pd.DataFrame:
        logger.info("loading sql data %s", self.sqlite_path)
        with sqlite3.connect(self.sqlite_path) as conn:
            return pd.read_sql_query(self.query, conn)


def get_data_source(cfg: Settings | None = None, **synthetic_kwargs) -> DataSource:
    cfg = cfg or settings
    if cfg.data_source == "file":
        if not cfg.data_path:
            raise ValueError("DATA_PATH must be set when DATA_SOURCE=file")
        return FileDataSource(cfg.data_path)
    if cfg.data_source == "sql":
        if not cfg.data_sql_path:
            raise ValueError("DATA_SQL_PATH must be set when DATA_SOURCE=sql")
        return SQLDataSource(cfg.data_sql_path, cfg.data_sql_query)
    return SyntheticDataSource(**synthetic_kwargs)
