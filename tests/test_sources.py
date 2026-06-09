import sqlite3

from product_intelligence.data.sources import (
    FileDataSource,
    SQLDataSource,
    SyntheticDataSource,
    get_data_source,
)


def test_synthetic_source():
    df = SyntheticDataSource(150, 3).load()
    assert len(df) == 150
    assert "will_churn" in df.columns


def test_file_source_roundtrip(tmp_path):
    df = SyntheticDataSource(80, 1).load()
    p = tmp_path / "accounts.csv"
    df.to_csv(p, index=False)
    assert len(FileDataSource(p).load()) == 80


def test_sql_source(tmp_path):
    df = SyntheticDataSource(60, 1).load()
    db = tmp_path / "w.db"
    with sqlite3.connect(db) as c:
        df.to_sql("accounts", c, index=False)
    assert len(SQLDataSource(str(db)).load()) == 60


def test_factory_defaults_to_synthetic():
    assert isinstance(get_data_source(n_samples=10, seed=1), SyntheticDataSource)
