"""Layer-5 verification: real Ledoit-Wolf + real cvxpy solve on a
synthetic universe (alpha/returns synthetic; the optimiser itself is the
production code). Shows MVO vs Black-Litterman overlap heavily but differ
in the tails, and that all constraints hold.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.portfolio.constraints import Constraints
from src.portfolio.optimiser import Optimiser

N = 40
T = [f"S{i:02d}" for i in range(N)]
SEC = pd.Series({t: ["Tech", "Energy", "Fin", "Staples", "Health"][i % 5]
                 for i, t in enumerate(T)})
BETA = pd.Series({t: 1.0 + 0.08 * ((i % 6) - 3) for i, t in enumerate(T)})


def main() -> int:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-06-01", periods=400)
    f = rng.normal(0, 0.01, len(idx))
    rets = pd.DataFrame(
        {t: 0.5 * f + rng.normal(0, 0.012, len(idx)) for t in T}, index=idx
    )
    alpha = pd.Series(rng.normal(0, 1, N), index=T, name="composite")

    opt = Optimiser()
    wm = opt.build(alpha, rets, SEC, BETA, mode="mvo")
    wb = opt.build(alpha, rets, SEC, BETA, mode="bl")
    c = Constraints()

    def stats(w, name):
        nz = w[w != 0]
        print(f"\n{name}: {len(nz)} positions | gross={w.abs().sum():.2f} "
              f"net={w.sum():+.3f} "
              f"beta={(BETA.reindex(w.index)*w).sum():+.3f}")
        top = w.sort_values(ascending=False).head(3)
        bot = w.sort_values().head(3)
        print("  top longs :", ", ".join(f"{k}={v:+.3f}" for k, v in top.items()))
        print("  top shorts:", ", ".join(f"{k}={v:+.3f}" for k, v in bot.items()))
        ok = (w.abs().sum() <= c.gross_max + 1e-4
              and abs(w.sum()) <= c.net_max + 1e-4
              and (w.abs() <= c.pos_max + 1e-4).all())
        print(f"  constraints OK: {ok}")

    stats(wm, "MVO")
    stats(wb, "Black-Litterman")

    common = (wm[wm != 0].index).intersection(wb[wb != 0].index)
    union = (wm[wm != 0].index).union(wb[wb != 0].index)
    jacc = len(common) / max(len(union), 1)
    same_side = sum(np.sign(wm[t]) == np.sign(wb[t]) for t in common)
    print(f"\nMVO vs BL: name overlap (Jaccard)={jacc:.0%}, "
          f"same-side on {same_side}/{len(common)} shared names")
    print("Expected: heavy overlap, differing in tail names — confirmed."
          if jacc > 0.4 else "Lower overlap than expected — inspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
