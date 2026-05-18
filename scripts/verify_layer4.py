"""Layer-4 live verification (uses whichever AI key is set).

Cheap by design: two short synthetic filing excerpts, temperature 0. The
spec's real cases (Enron/Luckin 10-Ks light up; Microsoft stays clean) use
the same analyzer on full SEC filing text from Layer 1 — this just proves
the wiring and that the prompt discriminates fraud from clean without
false-positiving.
"""
from __future__ import annotations

import sys

from src.config import settings

BAD = """
In Q4 we recognised $180M of revenue on bill-and-hold arrangements where
product remained in our warehouses and title had not transferred. We also
shipped significant inventory to distributors near quarter-end with
generous right-of-return terms to meet guidance. Days sales outstanding
rose from 52 to 119 days year over year. The auditor was replaced in
March. Management has substantial doubt about the Company's ability to
continue as a going concern absent additional financing.
"""

CLEAN = """
Net revenue grew 11% driven by cloud services. Operating cash flow of
$28.4B exceeded net income of $24.1B. Days sales outstanding were broadly
unchanged. There were no changes in auditors and no related-party
transactions outside the ordinary course. The Company maintains a strong
liquidity position with $34B of cash and marketable securities.
"""


def main() -> int:
    if not (settings.openai_api_key or settings.anthropic_api_key):
        print("SKIPPED: no AI key in .env (OPENAI_API_KEY or ANTHROPIC_API_KEY).")
        return 0

    from src.ai.forensic_analyzer import ForensicAnalyzer

    fa = ForensicAnalyzer()
    print(f"Provider: {fa.llm.name}\n")

    print("=" * 60, "\nFRAUDULENT EXCERPT (expect flags, negative score)\n", "=" * 60)
    r = fa.analyze("FRAUD", "2001-04-02", BAD, use_cache=False)
    for f in r.flags:
        print(f"  [{f.severity.upper():4}] {f.concern}\n         ev: {f.evidence[:90]}")
    print(f"  score_adjustment={r.score_adjustment}  confidence={r.confidence}")
    print(f"  summary: {r.summary}")
    bad_ok = len(r.flags) > 0 and r.score_adjustment < 0

    print("\n" + "=" * 60, "\nCLEAN EXCERPT (expect ~no flags, score 0)\n", "=" * 60)
    c = fa.analyze("CLEAN", "2023-07-27", CLEAN, use_cache=False)
    print(f"  flags={len(c.flags)}  score_adjustment={c.score_adjustment}"
          f"  confidence={c.confidence}")
    print(f"  summary: {c.summary}")
    clean_ok = len(c.flags) == 0 and c.score_adjustment == 0.0

    print("\nRESULT:",
          "PASS — discriminates fraud from clean ✓"
          if (bad_ok and clean_ok) else
          f"CHECK — bad_ok={bad_ok} clean_ok={clean_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
