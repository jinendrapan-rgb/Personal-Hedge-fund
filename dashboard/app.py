"""GreyMatter dashboard — entry (Overview). Run: streamlit run dashboard/app.py

Pages degrade gracefully when the pipeline hasn't produced data yet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.reporting.dashboard_data import (  # noqa: E402
    equity_curve, list_backtests, load_backtest, portfolio_snapshot,
)

st.set_page_config(page_title="GreyMatter", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data
def _runs():
    return list_backtests()


@st.cache_data
def _bt(run_id):
    return load_backtest(run_id)


st.title("GREYMATTER — AI Hedge Fund")
st.caption("L/S equity · S&P 500 · 5-factor · forensic AI · walk-forward")

runs = _runs()
if not runs:
    st.info("No backtest artifacts yet. Run a walk-forward backtest "
            "(Layer 3) — the dashboard reads `backtests/<run_id>/`.")
else:
    run = st.sidebar.selectbox("Backtest run", runs, index=len(runs) - 1)
    bt = _bt(run)
    m = bt["metrics"]
    snap = portfolio_snapshot(bt["positions"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe", m.get("sharpe", "—"))
    c2.metric("Ann. return", f'{m.get("ann_return", 0):.2%}')
    c3.metric("Max DD", f'{m.get("max_drawdown", 0):.2%}')
    c4.metric("Positions", snap["n_positions"])

    eq = equity_curve(bt["returns"])
    if not eq.empty:
        st.line_chart(eq, height=320)
    st.caption(f"gross {snap['gross']:.2f} · net {snap['net']:+.3f} · "
               f"{snap['n_long']}L / {snap['n_short']}S")
