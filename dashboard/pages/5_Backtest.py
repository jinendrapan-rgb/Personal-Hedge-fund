import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.reporting.dashboard_data import (  # noqa: E402
    equity_curve, list_backtests, load_backtest,
)

st.title("Backtest — the trust-building tab")
st.caption("Walk-forward results. Factor-only vs factor+AI is an A/B you "
           "run by executing two backtests; the AI overlay is OFF by "
           "default (spec) so the comparison is honest.")

runs = list_backtests()
if not runs:
    st.info("No walk-forward runs yet. Build one with Layer 3.")
else:
    chosen = st.multiselect("Compare runs", runs, default=runs[-1:])
    for run in chosen:
        bt = load_backtest(run)
        m = bt["metrics"]
        st.subheader(run)
        c = st.columns(4)
        c[0].metric("Sharpe", m.get("sharpe", "—"))
        c[1].metric("Ann.", f'{m.get("ann_return", 0):.2%}')
        c[2].metric("IR vs SPY", m.get("information_ratio_vs_bench", "—"))
        c[3].metric("Rebalances", m.get("n_rebalances", "—"))
        eq = equity_curve(bt["returns"])
        if not eq.empty:
            st.line_chart(eq, height=240)
