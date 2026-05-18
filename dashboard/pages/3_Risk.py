import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.reporting.dashboard_data import list_backtests, load_backtest  # noqa: E402
from src.risk.circuit_breakers import evaluate  # noqa: E402

st.title("Risk")
runs = list_backtests()
if not runs:
    st.info("No backtest artifacts yet — risk panel needs a returns stream.")
else:
    bt = load_backtest(st.sidebar.selectbox("Run", runs, index=len(runs) - 1))
    r = bt["returns"]
    if r.empty or "total" not in r:
        st.info("Run has no returns series.")
    else:
        s = evaluate(r["total"])
        color = {"green": "🟢"}.get(s.state.value, "🔴")
        st.subheader(f"{color} Circuit breaker: {s.state.value}")
        c = st.columns(3)
        c[0].metric("Daily", f"{s.daily_pnl:+.2%}")
        c[1].metric("Weekly", f"{s.weekly_pnl:+.2%}")
        c[2].metric("Drawdown", f"{s.drawdown:+.2%}")
        st.caption(s.reason)
        st.line_chart((1 + r["total"].fillna(0)).cumprod(), height=260)
