"""
data_fetcher.py — Fetches price and fundamental data for the NSE universe.
Falls back to realistic mock data when yfinance is unavailable or blocked.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .config import (
    NSE_UNIVERSE, HISTORY_YEARS, NIFTY50_TICKER, REBALANCE_FREQ
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class UniverseDataFetcher:
    """
    Fetches and caches:
      - Monthly adjusted-close price matrix  (stocks × months)
      - Monthly fundamental snapshots        (P/B, P/E, EV/EBITDA, ROE, etc.)
      - Nifty 50 benchmark returns
    """

    def __init__(self, use_mock: bool = False) -> None:
        self.use_mock   = use_mock
        self.start      = datetime.today() - timedelta(weeks=52 * HISTORY_YEARS + 4)
        self.end        = datetime.today()
        self._prices:    Optional[pd.DataFrame] = None
        self._monthly:   Optional[pd.DataFrame] = None
        self._funds:     Optional[pd.DataFrame] = None
        self._benchmark: Optional[pd.Series]    = None

    # ── Public interface ──────────────────────────────────────────────────────

    def get_monthly_prices(self) -> pd.DataFrame:
        """Monthly (month-end) adjusted close prices — stocks as columns."""
        if self._monthly is not None:
            return self._monthly
        if self.use_mock:
            prices = self._mock_prices()
        else:
            prices = self._fetch_prices()
        self._monthly = prices.resample(REBALANCE_FREQ).last()
        logger.info(f"Monthly prices: {self._monthly.shape} — "
                    f"{self._monthly.index[0].date()} → {self._monthly.index[-1].date()}")
        return self._monthly

    def get_fundamentals(self) -> pd.DataFrame:
        """
        Fundamental snapshot per stock (latest available).
        Columns: book_to_price, earnings_yield, ev_to_ebitda, roe, roce, asset_growth
        """
        if self._funds is not None:
            return self._funds
        if self.use_mock:
            self._funds = self._mock_fundamentals()
        else:
            self._funds = self._fetch_fundamentals()
        return self._funds

    def get_benchmark_returns(self) -> pd.Series:
        """Monthly Nifty 50 returns."""
        if self._benchmark is not None:
            return self._benchmark
        if self.use_mock:
            bm = self._mock_benchmark()
        else:
            try:
                raw = yf.download(NIFTY50_TICKER, start=self.start,
                                  end=self.end, progress=False)["Close"]
                monthly = raw.resample(REBALANCE_FREQ).last()
                bm = monthly.pct_change().dropna()
                bm.name = "Nifty50"
                logger.info(f"Benchmark fetched: {len(bm)} monthly returns")
            except Exception as e:
                logger.warning(f"Benchmark fetch failed: {e}. Using mock.")
                bm = self._mock_benchmark()
        self._benchmark = bm
        return self._benchmark

    # ── Live fetch ────────────────────────────────────────────────────────────

    def _fetch_prices(self) -> pd.DataFrame:
        """Fetch daily adjusted prices for the full universe."""
        tickers = [t for _, t in NSE_UNIVERSE]
        logger.info(f"Fetching prices for {len(tickers)} stocks…")
        try:
            raw = yf.download(tickers, start=self.start, end=self.end,
                              progress=False, auto_adjust=True)["Close"]
            # Flatten MultiIndex if needed
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(0)
            # Rename columns to short names
            rename = {t: n for n, t in NSE_UNIVERSE if t in raw.columns}
            raw = raw.rename(columns=rename)
            logger.info(f"Prices fetched: {raw.shape}, "
                        f"coverage: {raw.notna().mean().mean():.1%}")
            return raw
        except Exception as e:
            logger.warning(f"Price fetch failed: {e}. Switching to mock.")
            return self._mock_prices()

    def _fetch_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental ratios from yfinance .info for each stock."""
        rows = []
        for name, ticker in NSE_UNIVERSE:
            try:
                info = yf.Ticker(ticker).info or {}
                bp  = 1 / info.get("priceToBook", np.nan) if info.get("priceToBook") else np.nan
                ey  = 1 / info.get("trailingPE",  np.nan) if info.get("trailingPE")  else np.nan
                eveb= info.get("enterpriseToEbitda", np.nan)
                roe = info.get("returnOnEquity",  np.nan)
                ag  = info.get("revenueGrowth",   np.nan)    # proxy for asset growth
                rows.append({
                    "stock": name,
                    "book_to_price":  bp,
                    "earnings_yield": ey,
                    "ev_to_ebitda":   eveb,
                    "roe":            roe,
                    "roce":           np.nan,  # not directly available in yfinance
                    "asset_growth":   ag,
                })
                logger.info(f"  {name}: B/P={'N/A' if np.isnan(bp) else f'{bp:.3f}'}")
            except Exception as e:
                logger.warning(f"  {name} fundamentals failed: {e}")
                rows.append({"stock": name, **{k: np.nan for k in
                             ["book_to_price","earnings_yield","ev_to_ebitda","roe","roce","asset_growth"]}})
        df = pd.DataFrame(rows).set_index("stock")
        logger.info(f"Fundamentals fetched for {len(df)} stocks | "
                    f"coverage: {df.notna().mean().mean():.1%}")
        return df

    # ── Mock generators ───────────────────────────────────────────────────────

    def _mock_prices(self) -> pd.DataFrame:
        """
        Generate realistic correlated mock prices for the full universe.
        Uses a factor model: return = market_beta * market_ret + sector_ret + idio
        """
        np.random.seed(42)
        names = [n for n, _ in NSE_UNIVERSE]
        n_stocks = len(names)
        n_days   = HISTORY_YEARS * 252

        dates = pd.bdate_range(end=datetime.today(), periods=n_days)

        # Market factor: ~10% annual return, 16% vol
        market_daily = np.random.normal(0.04/252, 0.16/np.sqrt(252), n_days)

        # Sector groupings — derive n_sectors dynamically from universe size
        sector_ids = np.array([i // 6 for i in range(n_stocks)])
        n_sectors  = int(sector_ids.max()) + 1

        prices = np.zeros((n_days, n_stocks))

        # Starting prices — vary between ₹100 and ₹5000
        np.random.seed(43)
        start_prices = np.exp(np.random.uniform(np.log(100), np.log(5000), n_stocks))
        prices[0] = start_prices

        for t in range(1, n_days):
            sector_rets = np.random.normal(0, 0.008, n_sectors)
            idio_rets   = np.random.normal(0, 0.012, n_stocks)
            betas       = np.random.uniform(0.6, 1.4, n_stocks)
            stock_rets  = betas * market_daily[t] + sector_rets[sector_ids] + idio_rets
            prices[t]   = prices[t-1] * np.exp(stock_rets)

        df = pd.DataFrame(prices, index=dates, columns=names)
        # Introduce some missing data (stocks that weren't listed early on)
        for i in range(0, n_stocks, 8):
            gap = np.random.randint(0, 60)
            if gap > 0:
                df.iloc[:gap, i] = np.nan

        logger.info(f"Mock prices generated: {df.shape}")
        return df

    def _mock_fundamentals(self) -> pd.DataFrame:
        """
        Generate mock fundamental data with realistic cross-sectional dispersion.
        Sector-level means with stock-level noise — mimics real NSE fundamentals.
        """
        np.random.seed(100)
        names = [n for n, _ in NSE_UNIVERSE]
        n = len(names)

        sector_id  = np.array([i // 6 for i in range(n)])
        n_sectors_ = int(sector_id.max()) + 1   # dynamic, matches universe size

        # Sector-level fundamental characteristics — extended to n_sectors_ entries
        _bp   = [0.15, 0.10, 0.12, 0.08, 0.20, 0.25, 0.09, 0.18, 0.22, 0.11, 0.16]
        _ey   = [0.04, 0.03, 0.04, 0.02, 0.07, 0.06, 0.03, 0.05, 0.07, 0.04, 0.05]
        _roe  = [0.15, 0.22, 0.18, 0.12, 0.10, 0.08, 0.16, 0.20, 0.12, 0.14, 0.13]
        _eveb = [18.0, 25.0, 22.0, 30.0, 12.0, 10.0, 28.0, 20.0, 15.0, 22.0, 19.0]
        _ag   = [0.12, 0.15, 0.08, 0.20, 0.05, 0.03, 0.18, 0.10, 0.07, 0.12, 0.09]
        sector_bp_mean   = np.array(_bp[:n_sectors_])
        sector_ey_mean   = np.array(_ey[:n_sectors_])
        sector_roe_mean  = np.array(_roe[:n_sectors_])
        sector_eveb_mean = np.array(_eveb[:n_sectors_])
        sector_ag_mean   = np.array(_ag[:n_sectors_])

        rows = []
        for i, name in enumerate(names):
            s = sector_id[i]
            rows.append({
                "stock":          name,
                "book_to_price":  max(0.01, sector_bp_mean[s]  + np.random.normal(0, 0.05)),
                "earnings_yield": max(0.00, sector_ey_mean[s]  + np.random.normal(0, 0.015)),
                "ev_to_ebitda":   max(2.0,  sector_eveb_mean[s]+ np.random.normal(0, 5.0)),
                "roe":            max(0.00, sector_roe_mean[s] + np.random.normal(0, 0.04)),
                "roce":           max(0.00, sector_roe_mean[s] * 0.85 + np.random.normal(0, 0.03)),
                "asset_growth":   max(-0.2, sector_ag_mean[s]  + np.random.normal(0, 0.08)),
            })

        df = pd.DataFrame(rows).set_index("stock")
        logger.info(f"Mock fundamentals generated for {len(df)} stocks")
        return df

    def _mock_benchmark(self) -> pd.Series:
        """Generate synthetic Nifty 50 monthly returns."""
        np.random.seed(99)
        dates = pd.date_range(end=datetime.today(), periods=HISTORY_YEARS * 12, freq="ME")
        n_months = len(dates)   # actual length after date_range construction
        rets  = np.random.normal(0.012, 0.045, n_months)   # ~14% annual, 16% vol
        return pd.Series(rets, index=dates, name="Nifty50")
