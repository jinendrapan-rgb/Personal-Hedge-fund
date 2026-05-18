"""Layer 8 — attribution, JARVIS, dashboard loaders, page compile smoke."""
import json
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import dashboard_data as dd
from src.reporting.attribution import attribute
from src.reporting.jarvis import Jarvis


# ---- attribution ------------------------------------------------------
def test_attribution_recovers_known_loadings():
    rng = np.random.default_rng(0)
    n = 500
    f1 = rng.normal(0.0004, 0.01, n)
    f2 = rng.normal(0.0002, 0.008, n)
    y = 1.5 * f1 - 0.5 * f2 + rng.normal(0, 1e-5, n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    y_ser = pd.Series(y, index=idx)
    res = attribute(y_ser, pd.DataFrame({"f1": f1, "f2": f2}, index=idx))
    assert res["r_squared"] > 0.99
    # Invariant: factor contributions + alpha reconstruct annualised return.
    recon = res["f1"] + res["f2"] + res["alpha"]
    assert recon == pytest.approx(y_ser.mean() * 252, rel=1e-6)
    assert abs(res["alpha"]) < 0.02   # near-zero true alpha recovered


def test_attribution_needs_overlap():
    with pytest.raises(ValueError):
        attribute(pd.Series([0.1, 0.2]),
                  pd.DataFrame({"a": [0.1, 0.2], "b": [0.0, 0.1]}))


# ---- JARVIS -----------------------------------------------------------
class FakeLLM:
    def __init__(self): self.seen = None

    def complete_text(self, system, user):
        self.seen = (system, user)
        return "  Sharpe is 0.82.  "


def test_jarvis_grounds_prompt_in_state_and_strips():
    llm = FakeLLM()
    state = {"metrics": {"sharpe": 0.82}, "portfolio": {"n_positions": 50}}
    out = Jarvis(llm=llm).ask("What is the Sharpe?", state)
    assert out == "Sharpe is 0.82."
    sys_prompt, user = llm.seen
    assert '"sharpe": 0.82' in sys_prompt and "n_positions" in sys_prompt
    assert user == "What is the Sharpe?"
    assert "do not" in sys_prompt.lower() or "never invent" in sys_prompt.lower()


# ---- dashboard loaders -------------------------------------------------
def test_loaders_graceful_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_BT", tmp_path / "none")
    assert dd.list_backtests() == []
    bt = dd.load_backtest("missing")
    assert bt["metrics"] == {} and bt["returns"].empty
    assert dd.portfolio_snapshot(pd.DataFrame())["n_positions"] == 0
    assert dd.equity_curve(pd.DataFrame()).empty


def test_loaders_read_real_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_BT", tmp_path)
    run = tmp_path / "r1"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"sharpe": 0.9}))
    rets = pd.DataFrame({"total": [0.01, -0.005, 0.02]},
                        index=pd.bdate_range("2024-01-02", periods=3))
    rets.to_parquet(run / "returns.parquet")
    pos = pd.DataFrame({"AAA": [0.04], "BBB": [-0.04]},
                       index=pd.to_datetime(["2024-01-31"]))
    pos.to_parquet(run / "positions.parquet")

    assert dd.list_backtests() == ["r1"]
    bt = dd.load_backtest("r1")
    assert bt["metrics"]["sharpe"] == 0.9
    assert dd.equity_curve(bt["returns"]).iloc[-1] == pytest.approx(
        (1.01) * (0.995) * (1.02))
    snap = dd.portfolio_snapshot(bt["positions"])
    assert snap["n_long"] == 1 and snap["n_short"] == 1
    assert snap["gross"] == pytest.approx(0.08)


# ---- dashboard pages compile ------------------------------------------
def test_dashboard_pages_compile():
    root = Path(__file__).resolve().parents[1] / "dashboard"
    files = [root / "app.py", *sorted((root / "pages").glob("*.py"))]
    assert len(files) == 8  # Overview + 7 pages
    for f in files:
        py_compile.compile(str(f), doraise=True)
