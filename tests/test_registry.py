import pandas as pd
from sklearn.dummy import DummyClassifier

from product_intelligence.models.registry import ModelRegistry


def _toy_model():
    X = pd.DataFrame({"a": [0, 1, 0, 1]})
    y = [0, 1, 0, 1]
    return DummyClassifier(strategy="most_frequent").fit(X, y)


def test_register_and_load(tmp_path):
    reg = ModelRegistry(tmp_path)
    rec = reg.register("toy", _toy_model(), model_type="dummy", metrics={"acc": 1.0}, threshold=0.5)
    assert rec.stage == "production"
    pipeline, loaded = reg.load("toy", "production")
    assert loaded.version == rec.version
    assert loaded.metrics["acc"] == 1.0
    assert reg.resolve_version("toy", "latest") == rec.version


def test_promote_and_list(tmp_path):
    reg = ModelRegistry(tmp_path)
    v1 = reg.register("toy", _toy_model(), model_type="dummy", metrics={}, promote=True)
    v2 = reg.register("toy", _toy_model(), model_type="dummy", metrics={}, promote=False)
    assert reg.resolve_version("toy", "production") == v1.version
    reg.promote("toy", v2.version, "production")
    assert reg.resolve_version("toy", "production") == v2.version
    assert len(reg.list_versions("toy")) == 2
