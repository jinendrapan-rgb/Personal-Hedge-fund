"""End-to-end integration: the daily pipeline across all layers."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from src.factors.factor_engine import FACTORS
from src.factors.sectors import SectorProvider
from src.pipeline import DailyPipeline, preflight
from src.portfolio.optimiser import Optimiser
from src.reporting import dashboard_data as dd

TICKERS = [f"S{i:02d}" for i in range(24)]


class FakeDM:
    def get_universe(self, as_of):
        return TICKERS

    def get_prices(self, t, start, end):
        rng = np.random.default_rng(abs(hash(t)) % 2**32)
        idx = pd.bdate_range("2022-01-03", "2024-12-31")
        px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx)))
        df = pd.DataFrame({"close": px}, index=idx)
        return df.loc[df.index <= pd.Timestamp(end)]


class FakeFactorEngine:
    def compute(self, as_of, universe=None):
        rng = np.random.default_rng(42)
        z = rng.normal(0, 1, len(TICKERS))
        df = pd.DataFrame({"date": as_of, "ticker": TICKERS})
        for f in FACTORS:
            df[f] = rng.normal(0, 1, len(TICKERS))
        df["composite"] = z
        return df


@pytest.fixture
def patched_data_dir(tmp_path, monkeypatch):
    fake = types.SimpleNamespace(paths=types.SimpleNamespace(data=tmp_path))
    monkeypatch.setattr("src.pipeline.settings", fake)
    monkeypatch.setattr(dd, "settings",
                        types.SimpleNamespace(
                            paths=types.SimpleNamespace(
                                root=tmp_path, data=tmp_path)))
    return tmp_path


def _sectors():
    return pd.Series({t: ["Tech", "Energy", "Fin", "Staples"][i % 4]
                      for i, t in enumerate(TICKERS)})


def test_preflight_shape():
    pf = preflight().as_dict()
    assert set(pf) == {"sec", "polygon_key", "ai_key", "alpaca", "notes"}
    assert isinstance(pf["notes"], list) and pf["notes"]


def test_pipeline_runs_end_to_end_and_writes_artifacts(patched_data_dir):
    sec = _sectors()
    betas = pd.Series(1.0, index=TICKERS)
    pipe = DailyPipeline(FakeDM(), FakeFactorEngine(), Optimiser(),
                         sectors=sec, betas=betas)
    res = pipe.run(as_of="2024-12-31", universe=TICKERS)

    assert res.stages["universe"] == "ok"
    assert res.stages["factors"] == "ok"
    assert res.stages["ai_overlay"] == "off (default)"   # spec default
    assert res.stages["portfolio"] == "ok"
    assert res.stages["risk"] == "ok"
    assert res.stages["execution"].startswith("dry-run")
    assert res.ok()

    for fn in ("factor_scores.parquet", "target_portfolio.parquet",
               "pipeline_status.json"):
        assert (patched_data_dir / fn).exists()

    tgt = pd.read_parquet(patched_data_dir / "target_portfolio.parquet")
    assert {"ticker", "weight", "date"} <= set(tgt.columns)
    assert tgt["weight"].abs().sum() <= 2.0 + 1e-4          # gross respected

    status = json.loads((patched_data_dir / "pipeline_status.json").read_text())
    assert status["as_of"] == "2024-12-31"


def test_pipeline_halts_when_factors_empty(patched_data_dir):
    class EmptyFE:
        def compute(self, as_of, universe=None):
            return pd.DataFrame(columns=["date", "ticker", *FACTORS, "composite"])

    pipe = DailyPipeline(FakeDM(), EmptyFE(), Optimiser(),
                         sectors=_sectors(),
                         betas=pd.Series(1.0, index=TICKERS))
    res = pipe.run(as_of="2024-12-31", universe=TICKERS)
    assert "empty" in res.stages["factors"]
    assert not res.ok()
    assert "portfolio" not in res.stages          # halted before construction


def test_pipeline_blocks_on_degraded_factor_coverage(patched_data_dir):
    """Only quality+value populated (e.g. Polygon plan blocks price
    history) -> must refuse to build a portfolio, not silently succeed."""
    class PartialFE:
        def compute(self, as_of, universe=None):
            df = pd.DataFrame({"date": as_of, "ticker": TICKERS})
            for f in FACTORS:
                df[f] = np.nan
            df["quality"] = np.random.default_rng(0).normal(0, 1, len(TICKERS))
            df["value"] = np.random.default_rng(1).normal(0, 1, len(TICKERS))
            df["composite"] = df[["quality", "value"]].mean(axis=1)
            return df

    pipe = DailyPipeline(FakeDM(), PartialFE(), Optimiser(),
                         sectors=_sectors(),
                         betas=pd.Series(1.0, index=TICKERS))
    res = pipe.run(as_of="2024-12-31", universe=TICKERS)
    assert res.stages["factors"].startswith("blocked")
    assert "momentum" in res.stages["factors"]
    assert not res.ok()
    assert "portfolio" not in res.stages          # did NOT build a book


def test_ai_overlay_applied_when_enabled(patched_data_dir):
    calls = {"n": 0}

    def overlay(df):
        calls["n"] += 1
        return df

    pipe = DailyPipeline(FakeDM(), FakeFactorEngine(), Optimiser(),
                         ai_overlay=overlay, sectors=_sectors(),
                         betas=pd.Series(1.0, index=TICKERS),
                         enable_ai_overlay=True)
    res = pipe.run(as_of="2024-12-31", universe=TICKERS)
    assert calls["n"] == 1 and res.stages["ai_overlay"] == "ok (applied)"
