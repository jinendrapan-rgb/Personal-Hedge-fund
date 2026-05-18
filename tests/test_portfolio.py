"""Layer 5 — covariance, MVO constraints, Black-Litterman, optimiser."""
import numpy as np
import pandas as pd
import pytest

from src.portfolio.black_litterman import black_litterman, implied_returns
from src.portfolio.constraints import Constraints
from src.portfolio.covariance import ledoit_wolf_cov
from src.portfolio.mvo import InfeasibleError, MVOConfig, optimise
from src.portfolio.optimiser import Optimiser

N = 24
TICKERS = [f"S{i:02d}" for i in range(N)]
SECTORS = pd.Series({t: ["Tech", "Energy", "Fin", "Staples"][i % 4]
                     for i, t in enumerate(TICKERS)})
BETAS = pd.Series({t: 1.0 + 0.1 * ((i % 5) - 2) for i, t in enumerate(TICKERS)})


def _returns(seed=1, days=320):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=days)
    # common factor + idiosyncratic
    f = rng.normal(0, 0.01, days)
    data = {t: 0.5 * f + rng.normal(0, 0.012, days) for t in TICKERS}
    return pd.DataFrame(data, index=idx)


def _alpha(seed=2):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, N), index=TICKERS, name="composite")


# ---- covariance --------------------------------------------------------
def test_ledoit_wolf_well_formed_and_conditioned():
    r = _returns()
    cov = ledoit_wolf_cov(r)
    assert cov.shape == (N, N)
    assert np.allclose(cov.values, cov.values.T)
    eig = np.linalg.eigvalsh(cov.values)
    assert (eig > 0).all()                                  # PSD / invertible
    sample = r.cov().values * 252
    assert np.linalg.cond(cov.values) < np.linalg.cond(sample)  # better conditioned


# ---- MVO constraints ---------------------------------------------------
def _check_constraints(w, c: Constraints, tol=1e-5):
    assert w.abs().sum() <= c.gross_max + tol
    assert c.net_min - tol <= w.sum() <= c.net_max + tol
    assert (w.abs() <= c.pos_max + tol).all()
    for s in SECTORS.unique():
        net = w[SECTORS.reindex(w.index) == s].sum()
        assert abs(net) <= c.sector_net_tol + tol
    assert abs((BETAS.reindex(w.index) * w).sum()) <= c.beta_max + tol


def test_mvo_respects_all_constraints_and_tilts_to_alpha():
    cov = ledoit_wolf_cov(_returns())
    a = _alpha()
    w = optimise(a, cov, SECTORS, BETAS)
    _check_constraints(w, Constraints())
    assert np.corrcoef(w.values, a.reindex(w.index).values)[0, 1] > 0.3


def test_turnover_penalty_reduces_trading():
    cov = ledoit_wolf_cov(_returns())
    a = _alpha()
    prev = optimise(_alpha(seed=99), cov, SECTORS, BETAS)
    free = optimise(a, cov, SECTORS, BETAS, prev_w=prev)
    penal = optimise(a, cov, SECTORS, BETAS, prev_w=prev,
                     config=MVOConfig(turnover_penalty=0.5))
    to_free = (free - prev.reindex(free.index).fillna(0)).abs().sum()
    to_penal = (penal - prev.reindex(penal.index).fillna(0)).abs().sum()
    assert to_penal < to_free


def test_infeasible_raises():
    cov = ledoit_wolf_cov(_returns())
    impossible = Constraints(gross_max=0.0, net_min=0.5, net_max=0.6)
    with pytest.raises(InfeasibleError):
        optimise(_alpha(), cov, SECTORS, BETAS, constraints=impossible)


# ---- Black-Litterman ---------------------------------------------------
def test_bl_posterior_between_prior_and_views():
    cov = ledoit_wolf_cov(_returns())
    mw = pd.Series(1.0 / N, index=TICKERS)
    pi = implied_returns(cov, mw)
    views = pd.Series(np.linspace(-0.05, 0.05, N), index=TICKERS)

    hi = black_litterman(cov, mw, views, view_confidence=0.95)
    lo = black_litterman(cov, mw, views, view_confidence=0.05)
    # Higher confidence pulls the posterior further from the prior.
    assert (hi - pi).abs().mean() > (lo - pi).abs().mean()
    assert hi.shape == (N,)


# ---- optimiser end-to-end ---------------------------------------------
def test_optimiser_mvo_and_bl_valid_books():
    r, a = _returns(), _alpha()
    opt = Optimiser()
    for mode in ("mvo", "bl"):
        w = opt.build(a, r, SECTORS, BETAS, mode=mode)
        nz = w[w != 0]
        _check_constraints(nz, Constraints())
        # 0.5% floor enforced post-process
        assert (nz.abs() >= Constraints().pos_min - 1e-9).all()


def test_optimiser_mode_validation():
    with pytest.raises(ValueError):
        Optimiser().build(_alpha(), _returns(), SECTORS, BETAS, mode="xxx")
