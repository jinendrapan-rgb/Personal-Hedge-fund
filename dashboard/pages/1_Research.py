import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.reporting.dashboard_data import load_factor_scores  # noqa: E402

st.title("Research — factor candidates")
fs = load_factor_scores()
if fs.empty:
    st.info("No factor_scores.parquet yet. Run the Factor Engine (Layer 2).")
else:
    n = st.slider("Names per side", 10, 50, 30)
    st.subheader("Top longs")
    st.dataframe(fs.sort_values("composite", ascending=False).head(n),
                 use_container_width=True)
    st.subheader("Top shorts")
    st.dataframe(fs.sort_values("composite").head(n),
                 use_container_width=True)
    st.caption("Approve/Reject would write to a decisions table "
               "(wired when a live pipeline is running).")
