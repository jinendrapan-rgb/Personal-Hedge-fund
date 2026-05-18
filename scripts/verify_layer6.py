"""Layer-6 verification: the full risk suite on a synthetic market-neutral
book (real risk code; synthetic prices since Polygon needs the key)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.risk.circuit_breakers import evaluate
from src.risk.correlation import monitor
from src.risk.factor_model import decompose
from src.risk.pre_trade import pre_trade_check
from src.risk.stress_tests import run_stress

N = 30
T = [f"S{i:02d}" for i in range(N)]
SEC = pd.Series({t: ["Tech", "Energy", "Fin"][i % 3] for i, t in enumerate(T)})
BETA = pd.Series({t: 1.0 + 0.1 * ((i % 5) - 2) for i, t in enumerate(T)})


class Locate:
    def is_shortable(self, t): return True


def main() -> int:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2007-01-01", "2024-12-31")
    f = rng.normal(0, 0.009, len(idx))
    rets = pd.DataFrame(
        {t: 0.4 * f + rng.normal(0.0002, 0.013, len(idx)) for t in T}, index=idx
    )

    # sector- & roughly beta-neutral book within the ±4% box
    w = pd.Series(0.0, index=T)
    for s in SEC.unique():
        names = SEC.index[SEC == s].tolist()
        w[names[0]] = 0.03
        w[names[1]] = -0.03
    adv = pd.Series(2e9, index=T)

    print("PRE-TRADE")
    pt = pre_trade_check(w, SEC, BETA, adv_usd=adv, book_usd=5e7, locate=Locate())
    print(f"  ok={pt.ok}  violations={pt.violations}")

    print("\nCIRCUIT BREAKERS (on the synthetic strategy P&L)")
    port = (w * rets).sum(axis=1)
    st = evaluate(port.tail(60))
    print(f"  state={st.state.value}  daily={st.daily_pnl:+.3%} "
          f"weekly={st.weekly_pnl:+.3%} dd={st.drawdown:+.3%}")

    print("\nFACTOR vs SPECIFIC RISK (target 70-85% specific)")
    d = decompose(w, rets.tail(504), n_factors=5)
    print(f"  specific={d.specific_share:.1%}  factor={d.factor_share:.1%}  "
          f"flag_rebalance={d.flag_rebalance}")

    print("\nSTRESS TESTS (history spans 2007-2024 -> empirical replay)")
    for r in run_stress(w, rets, portfolio_beta=(BETA.reindex(w.index) * w).sum()):
        print(f"  {r.scenario:15s} {r.method:9s} "
              f"port_return={r.portfolio_return:+.2%} {r.note}")

    print("\nCORRELATION / CONCENTRATION")
    rep = monitor(w, rets)
    print(f"  ok={rep.ok}  high_corr_pairs={len(rep.high_corr_pairs)} "
          f"heavy_clusters={len(rep.heavy_clusters)}")

    print("\nNote: neutral book -> small stress numbers is the POINT of the "
          "construction, not a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
