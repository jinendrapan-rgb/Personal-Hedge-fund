import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.reporting.dashboard_data import list_backtests, load_backtest  # noqa: E402

st.title("Performance — tear sheet")
runs = list_backtests()
if not runs:
    st.info("No backtest artifacts yet.")
else:
    bt = load_backtest(st.sidebar.selectbox("Run", runs, index=len(runs) - 1))
    m = bt["metrics"]
    if not m:
        st.info("No metrics.json for this run.")
    else:
        st.dataframe(pd.Series(m).rename("value").to_frame(),
                     use_container_width=True)
        r = bt["returns"]
        if not r.empty and "total" in r:
            monthly = ((1 + r["total"]).resample("ME").prod() - 1)
            st.subheader("Monthly returns")
            st.bar_chart(monthly)
