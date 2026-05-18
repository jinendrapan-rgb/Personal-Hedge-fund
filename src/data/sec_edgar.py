"""SEC EDGAR access via the structured data APIs.

We deliberately use ``data.sec.gov`` JSON endpoints rather than scraping
filing HTML:

* ``company_tickers.json``        -> ticker -> CIK map
* ``submissions/CIK##########``   -> filing index (10-K/10-Q/8-K/4/13F)
* ``api/xbrl/companyfacts/CIK..`` -> ALL XBRL facts, each tagged with the
  ``filed`` date of the filing that reported it.

The ``filed`` date is the point-in-time key: it lets us retrieve a
fundamental exactly as it was known on a given date, not as later restated.
SEC's fair-access policy requires a descriptive User-Agent.
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.data._http import RateLimiter, get_json

log = logging.getLogger("greymatter.data.sec")

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

FILING_FORMS = ("10-K", "10-Q", "8-K", "4", "13F-HR")


class SecEdgarClient:
    def __init__(self, user_agent: str | None = None, rate_per_sec: float | None = None) -> None:
        self._headers = {
            "User-Agent": user_agent or settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._limiter = RateLimiter(rate_per_sec or settings.sec_rate_per_sec)
        self._ticker_map: dict[str, str] | None = None

    # ---- ticker -> CIK -------------------------------------------------
    def ticker_to_cik(self, ticker: str) -> str:
        """Return the zero-padded 10-digit CIK for ``ticker``."""
        if self._ticker_map is None:
            self._ticker_map = self._load_ticker_map()
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise KeyError(f"No CIK found for ticker {ticker!r}")
        return cik

    def _load_ticker_map(self) -> dict[str, str]:
        raw = get_json(_TICKERS_URL, limiter=self._limiter, headers=self._headers)
        # raw is {"0": {"cik_str": int, "ticker": str, "title": str}, ...}
        return {
            row["ticker"].upper(): f"{int(row['cik_str']):010d}"
            for row in raw.values()
        }

    # ---- filings index -------------------------------------------------
    def get_filings(
        self, ticker: str, forms: tuple[str, ...] = FILING_FORMS
    ) -> list[dict[str, Any]]:
        """Recent filings for ``ticker`` filtered to ``forms``.

        Each item: ``{form, filed, accession, primary_doc, period}``.
        """
        cik = self.ticker_to_cik(ticker)
        payload = get_json(
            _SUBMISSIONS_URL.format(cik10=cik),
            limiter=self._limiter,
            headers=self._headers,
        )
        recent = payload.get("filings", {}).get("recent", {})
        out: list[dict[str, Any]] = []
        for form, filed, accn, doc, period in zip(
            recent.get("form", []),
            recent.get("filingDate", []),
            recent.get("accessionNumber", []),
            recent.get("primaryDocument", []),
            recent.get("reportDate", []),
        ):
            if form in forms:
                out.append(
                    {
                        "form": form,
                        "filed": filed,
                        "accession": accn,
                        "primary_doc": doc,
                        "period": period,
                    }
                )
        return out

    # ---- XBRL company facts -------------------------------------------
    def get_company_facts(self, ticker: str) -> dict[str, Any]:
        """Raw ``companyfacts`` JSON (structured XBRL, all history)."""
        cik = self.ticker_to_cik(ticker)
        return get_json(
            _FACTS_URL.format(cik10=cik),
            limiter=self._limiter,
            headers=self._headers,
        )
