"""Central configuration. Reads .env via python-dotenv.

All paths and credentials flow through here so the rest of the codebase never
touches os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    prices: Path = PROJECT_ROOT / "data" / "prices"
    fundamentals: Path = PROJECT_ROOT / "data" / "fundamentals"
    filings: Path = PROJECT_ROOT / "data" / "filings"
    universe: Path = PROJECT_ROOT / "data" / "universe"
    cache: Path = PROJECT_ROOT / "data" / "cache"

    def ensure(self) -> None:
        for p in (self.prices, self.fundamentals, self.filings, self.universe, self.cache):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    polygon_api_key: str = field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    alpaca_api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    alpaca_base_url: str = field(
        default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("SEC_USER_AGENT", "GreyMatter Research contact@example.com")
    )
    # Polygon Starter allows 5 req/sec. Keep a margin.
    polygon_rate_per_sec: float = 5.0
    sec_rate_per_sec: float = 5.0

    paths: Paths = field(default_factory=Paths)

    def require(self, *names: str) -> None:
        """Raise a clear error if a required credential is missing.

        Call this from code paths that actually hit the network so that
        offline unit tests (which never call it) keep passing without keys.
        """
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"Missing required credential(s): {', '.join(missing)}. "
                f"Add them to {PROJECT_ROOT / '.env'} (see .env.example)."
            )


settings = Settings()
settings.paths.ensure()
