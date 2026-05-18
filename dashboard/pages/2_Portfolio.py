import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.reporting.dashboard_data import (  # noqa: E402
    list_backtests, load_backtest, portfolio_snapshot,
)

st.title("Portfolio")
runs = list_backtests()
if not runs:
    st.info("No backtest artifacts yet.")
else:
    run = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
    bt = load_backtest(run)
    snap = portfolio_snapshot(bt["positions"])
    c = st.columns(4)
    c[0].metric("Positions", snap["n_positions"])
    c[1].metric("Gross", f'{snap["gross"]:.2f}')
    c[2].metric("Net", f'{snap["net"]:+.3f}')
    c[3].metric("L / S", f'{snap["n_long"]} / {snap["n_short"]}')
    if not bt["positions"].empty:
        st.subheader("Latest target weights")
        latest = bt["positions"].iloc[-1].dropna()
        st.bar_chart(latest[latest != 0].sort_values())
