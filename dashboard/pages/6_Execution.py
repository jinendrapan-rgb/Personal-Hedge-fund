import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.config import settings  # noqa: E402
from src.reporting.dashboard_data import list_backtests, load_backtest  # noqa: E402

st.title("Execution")
mode = "LIVE (confirm required)" if settings.alpaca_live else "paper"
keys = bool(settings.alpaca_api_key and settings.alpaca_secret_key)
st.write(f"Alpaca keys present: **{keys}** · mode: **{mode}**")
st.caption("Live trading needs ALPACA_LIVE=true AND explicit confirm_live.")

runs = list_backtests()
if not runs:
    st.info("No runs. Live execution logs appear here once trading.")
else:
    bt = load_backtest(st.sidebar.selectbox("Run", runs, index=len(runs) - 1))
    tr = bt["trades"]
    if tr.empty:
        st.info("No trades recorded for this run.")
    else:
        st.subheader("Rebalance trades")
        st.dataframe(tr.tail(200), use_container_width=True)
        st.caption(f"{len(tr)} trade rows · slippage tracked at execution time")
