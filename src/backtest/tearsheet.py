"""Self-contained HTML tear sheet (no external deps; inline SVG curve)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _equity_svg(total: pd.Series, bench: pd.Series, w=720, h=240) -> str:
    eq = (1 + total.fillna(0)).cumprod()
    bq = (1 + bench.reindex(total.index).fillna(0)).cumprod()
    lo = float(min(eq.min(), bq.min()))
    hi = float(max(eq.max(), bq.max()))
    rng = (hi - lo) or 1.0

    def path(s: pd.Series) -> str:
        n = len(s)
        pts = []
        for i, v in enumerate(s.values):
            x = (i / max(n - 1, 1)) * w
            y = h - ((float(v) - lo) / rng) * (h - 20) - 10
            pts.append(f"{'M' if i == 0 else 'L'}{x:.1f} {y:.1f}")
        return " ".join(pts)

    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'preserveAspectRatio="none" style="background:#0a0f15;border-radius:8px">'
        f'<path d="{path(bq)}" fill="none" stroke="#5f7d86" '
        f'stroke-width="1.4" stroke-dasharray="4 3"/>'
        f'<path d="{path(eq)}" fill="none" stroke="#37e8d8" stroke-width="2"/>'
        f"</svg>"
    )


def write_tearsheet(
    path: Path, returns: pd.DataFrame, bench: pd.Series, metrics: dict
) -> None:
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items()
    )
    monthly = (
        (1 + returns["total"]).resample("ME").prod() - 1
        if not returns.empty else pd.Series(dtype=float)
    )
    mrows = "".join(
        f"<tr><td>{ix.date()}</td><td "
        f'style="color:{"#54f0a8" if v>=0 else "#ff5d6c"}">{v:+.2%}</td></tr>'
        for ix, v in monthly.items()
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>GreyMatter Tear Sheet</title>
<style>
body{{background:#070a0f;color:#cfe8ec;font-family:ui-monospace,Menlo,monospace;
padding:28px;max-width:840px;margin:auto}}
h1{{color:#37e8d8;letter-spacing:3px;font-size:20px}}
h2{{color:#ffb454;font-size:12px;letter-spacing:2px;margin-top:26px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
td{{border-bottom:1px solid #16323c;padding:6px 8px}}
td:first-child{{color:#5f7d86}} .legend{{font-size:11px;color:#5f7d86}}
</style></head><body>
<h1>GREYMATTER — WALK-FORWARD TEAR SHEET</h1>
<p class="legend">cyan = strategy &nbsp;·&nbsp; dashed grey = benchmark</p>
{_equity_svg(returns["total"], bench)}
<h2>METRICS</h2><table>{rows}</table>
<h2>MONTHLY RETURNS</h2><table>{mrows}</table>
</body></html>"""
    path.write_text(html)
