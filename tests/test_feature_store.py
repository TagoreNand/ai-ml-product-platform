import pandas as pd

from product_intelligence.features.store import FeatureStore


def test_online_and_point_in_time(tmp_path):
    fs = FeatureStore(tmp_path / "fs")
    history = pd.DataFrame(
        {
            "account_id": ["x", "x", "y"],
            "event_timestamp": ["2024-01-01", "2024-02-01", "2024-01-15"],
            "nps": [10, 40, 5],
        }
    )
    fs.materialize(history)

    online = fs.get_online_features(["x", "y"], ["nps"])
    latest = dict(zip(online.account_id, online.nps, strict=False))
    assert latest["x"] == 40 and latest["y"] == 5

    # As-of: x at 2024-01-20 must see 10 (not the future 40) -> no leakage.
    entity = pd.DataFrame({"account_id": ["x"], "event_timestamp": ["2024-01-20"]})
    pit = fs.get_historical_features(entity, ["nps"])
    assert int(pit["nps"].iloc[0]) == 10
