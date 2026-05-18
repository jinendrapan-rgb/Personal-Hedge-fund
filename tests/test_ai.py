"""Layer 4 — forensic AI layer, fully offline (LLM mocked)."""
import pandas as pd
import pytest

from src.ai import cache
from src.ai.aggregate import apply_overlay, combine_reports
from src.ai.forensic_analyzer import ForensicAnalyzer
from src.ai.insider_analyzer import InsiderAnalyzer
from src.ai.llm import _coerce_json
from src.ai.models import AnalysisReport


class FakeLLM:
    name = "fake"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        return dict(self.payload)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_DIR", tmp_path / "ai")


# ---- models / validation ----------------------------------------------
def test_score_clamped_and_severity_normalised():
    r = AnalysisReport.parse(
        {"flags": [{"concern": "x", "severity": "CRITICAL", "evidence": "q"}],
         "score_adjustment": -9.0, "confidence": "totally"},
        analyzer="forensic", ticker="aapl", filing_date="2020-01-01")
    assert r.score_adjustment == -2.0          # clamped
    assert r.flags[0].severity == "low"        # unknown -> low
    assert r.confidence == "low"               # unknown -> low
    assert r.ticker == "AAPL"


def test_no_flags_forces_zero_score():
    r = AnalysisReport.parse(
        {"flags": [], "score_adjustment": -1.7},
        analyzer="forensic", ticker="MSFT", filing_date="2023-07-01")
    assert r.flags == [] and r.score_adjustment == 0.0


def test_flag_without_concern_dropped():
    r = AnalysisReport.parse(
        {"flags": [{"severity": "high", "evidence": "q"}], "score_adjustment": -2},
        analyzer="forensic", ticker="X", filing_date="d")
    assert r.flags == [] and r.score_adjustment == 0.0


# ---- coerce json -------------------------------------------------------
def test_coerce_json_strips_fences_and_prose():
    assert _coerce_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _coerce_json('Here you go: {"b": 2} thanks') == {"b": 2}


# ---- analyzer + cache --------------------------------------------------
def test_forensic_analyzer_parses_and_caches():
    llm = FakeLLM({
        "flags": [{"concern": "channel stuffing", "severity": "high",
                   "evidence": "shipped product to distributors with right of return"}],
        "score_adjustment": -1.5, "summary": "aggressive rev rec",
        "confidence": "high"})
    fa = ForensicAnalyzer(llm=llm)

    r1 = fa.analyze("ENRN", "2001-04-02", "...10-K text...")
    assert r1.flags[0].concern == "channel stuffing"
    assert r1.score_adjustment == -1.5 and llm.calls == 1

    # Second call: served from cache, LLM NOT hit again.
    r2 = fa.analyze("ENRN", "2001-04-02", "...10-K text...")
    assert r2.score_adjustment == -1.5 and llm.calls == 1

    # use_cache=False forces a fresh call.
    fa.analyze("ENRN", "2001-04-02", "...10-K text...", use_cache=False)
    assert llm.calls == 2


def test_clean_filing_yields_no_flags():
    llm = FakeLLM({"flags": [], "score_adjustment": 0.0,
                   "summary": "nothing unusual", "confidence": "high"})
    r = ForensicAnalyzer(llm=llm).analyze("MSFT", "2023-07-27", "clean 10-K")
    assert r.flags == [] and r.score_adjustment == 0.0


def test_insider_analyzer_serialises_txns():
    llm = FakeLLM({"flags": [{"concern": "cluster buying", "severity": "med",
                              "evidence": "3 insiders bought on 2024-05-01"}],
                   "score_adjustment": 1.0, "confidence": "med"})
    r = InsiderAnalyzer(llm=llm).analyze(
        "ABC", "2024-06-01",
        [{"insider": "CEO", "type": "buy", "shares": 1000, "plan": "open"}])
    assert r.score_adjustment == 1.0 and r.analyzer == "insider"


# ---- aggregate / overlay ----------------------------------------------
def _rep(score):
    return AnalysisReport.parse(
        {"flags": [{"concern": "c", "severity": "low", "evidence": "e"}],
         "score_adjustment": score},
        analyzer="a", ticker="T", filing_date="d")


def test_combine_reports_mean_and_clip():
    assert combine_reports([]) == 0.0
    assert combine_reports([_rep(-2), _rep(-1)]) == pytest.approx(-1.5)
    assert combine_reports([_rep(2), _rep(2), _rep(2)]) == 2.0


def test_overlay_off_by_default_does_not_change_composite():
    fs = pd.DataFrame({"ticker": ["A", "B", "C"],
                       "composite": [1.0, 0.0, -1.0]})
    out = apply_overlay(fs, {"A": -2.0, "B": 2.0}, enabled=False)
    assert "ai_score" in out
    assert out["composite"].tolist() == [1.0, 0.0, -1.0]   # unchanged
    assert out.loc[out.ticker == "A", "ai_score"].iloc[0] == -2.0


def test_overlay_enabled_applies_clipped_adjustment():
    fs = pd.DataFrame({"ticker": ["A", "B", "C", "D"],
                       "composite": [2.0, 1.0, -1.0, -2.0]})
    out = apply_overlay(fs, {"A": -5.0}, enabled=True)
    a = out.loc[out.ticker == "A"]
    # A's huge -5 AI score is clipped to -2*sigma, not applied raw.
    sigma = pd.Series([2.0, 1.0, -1.0, -2.0]).std(ddof=0)
    assert a["composite"].iloc[0] == pytest.approx(2.0 - 2 * sigma)
