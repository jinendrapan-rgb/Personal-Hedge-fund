import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.config import settings  # noqa: E402
from src.reporting.dashboard_data import (  # noqa: E402
    list_backtests, load_backtest, portfolio_snapshot,
)
from src.reporting.jarvis import Jarvis  # noqa: E402

st.title("JARVIS")
if not (settings.openai_api_key or settings.anthropic_api_key):
    st.warning("No AI key set (OPENAI_API_KEY or ANTHROPIC_API_KEY). "
               "JARVIS needs one to answer.")
    st.stop()

runs = list_backtests()
state = {"note": "no backtest loaded"}
if runs:
    bt = load_backtest(runs[-1])
    state = {"latest_run": runs[-1], "metrics": bt["metrics"],
             "portfolio": portfolio_snapshot(bt["positions"])}

q = st.text_input("Ask about the portfolio / risk / performance")
if q:
    with st.spinner("Thinking…"):
        st.markdown(Jarvis().ask(q, state))
st.caption("Grounded in current backtest state — JARVIS is told not to "
           "invent positions or numbers.")
