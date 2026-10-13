"""Smoke tests: rule engine, saved models and every dashboard page."""
import joblib, pandas as pd, pytest
from streamlit.testing.v1 import AppTest
from src.config import ARTIFACTS, ROOT
from src.preprocessing import ENG_FEATURES, add_engineered_features
from src.reasoning import forward_chain, make_rules, prove

RULES = make_rules(["dec", "mar", "oct", "sep"], ["retired", "student"], 1430.25, 5)
YES = dict(age=34, job="student", marital="single", education="tertiary", default="no", balance=3200, housing="no", loan="no",
           contact="cellular", day=15, month="mar", campaign=2, pdays=90, previous=2, poutcome="success")
NO = dict(YES, job="management", housing="yes", month="may", poutcome="unknown", pdays=-1, previous=0)


def test_forward_chaining():
    f, fired = forward_chain(RULES, add_engineered_features(pd.DataFrame([YES])).iloc[0].to_dict())
    assert "Recommend" in f and "PriorSuccess" in f and fired[-1] == "R13"
    assert "Recommend" not in forward_chain(RULES, NO)[0]


def test_backward_chaining_agrees_with_forward():
    for row in (YES, NO):
        assert prove(RULES, "Recommend", row)["ok"] == ("Recommend" in forward_chain(RULES, row)[0])


def test_saved_models_rank_yes_above_no():
    for name, m in joblib.load(ARTIFACTS / "models.joblib").items():
        p = lambda r: m["pipeline"].predict_proba(add_engineered_features(pd.DataFrame([r]))[ENG_FEATURES])[0, 1]
        assert p(YES) > m["threshold"] > p(NO), name


@pytest.mark.parametrize("page", ["overview", "explorer", "performance", "predict", "batch", "logic", "about"])
def test_page_renders(page):
    at = AppTest.from_string(f"from src.dashboard import pages\npages.{page}()", default_timeout=180).run()
    assert not at.exception


def test_full_app_with_menu():
    assert not AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run().exception


def test_neural_network_saved_and_valid():
    import numpy as np
    m = joblib.load(ARTIFACTS / "models.joblib")["Neural Network (MLP)"]
    pipe = m["pipeline"]; X = add_engineered_features(pd.DataFrame([YES, NO]))[ENG_FEATURES]
    p = pipe.predict_proba(X)
    assert p.shape == (2, 2) and np.allclose(p.sum(axis=1), 1) and ((p >= 0) & (p <= 1)).all()
    assert len(pipe.named_steps["model"].models_) == 5
