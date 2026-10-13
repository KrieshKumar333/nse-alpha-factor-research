"""
config.py — Alpha Factor Research: NSE 500 Universe
Everything configurable for the systematic factor study lives here.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  UNIVERSE — NSE 500 representative stocks (large + mid cap)
#  Using a curated, liquid 60-stock subset representative of NSE 500 sectors
# ─────────────────────────────────────────────────────────────────────────────
NSE_UNIVERSE: List[Tuple[str, str]] = [
    # Financials
    ("HDFCBANK",   "HDFCBANK.NS"),  ("ICICIBANK", "ICICIBANK.NS"),
    ("SBIN",       "SBIN.NS"),      ("KOTAKBANK", "KOTAKBANK.NS"),
    ("AXISBANK",   "AXISBANK.NS"),  ("BAJFINANCE","BAJFINANCE.NS"),
    # IT
    ("TCS",        "TCS.NS"),       ("INFY",      "INFY.NS"),
    ("WIPRO",      "WIPRO.NS"),     ("HCLTECH",   "HCLTECH.NS"),
    ("TECHM",      "TECHM.NS"),     ("LTIM",      "LTIM.NS"),
    # Consumer Staples
    ("HINDUNILVR", "HINDUNILVR.NS"),("NESTLEIND",  "NESTLEIND.NS"),
    ("BRITANNIA",  "BRITANNIA.NS"), ("DABUR",      "DABUR.NS"),
    ("MARICO",     "MARICO.NS"),    ("COLPAL",     "COLPAL.NS"),
    # Industrials / Infra
    ("LT",         "LT.NS"),        ("SIEMENS",    "SIEMENS.NS"),
    ("ABB",        "ABB.NS"),       ("HAVELLS",    "HAVELLS.NS"),
    ("BHEL",       "BHEL.NS"),      ("CUMMINSIND", "CUMMINSIND.NS"),
    # Energy
    ("RELIANCE",   "RELIANCE.NS"),  ("ONGC",       "ONGC.NS"),
    ("BPCL",       "BPCL.NS"),      ("IOC",        "IOC.NS"),
    ("POWERGRID",  "POWERGRID.NS"), ("NTPC",       "NTPC.NS"),
    # Materials / Chemicals
    ("TATASTEEL",  "TATASTEEL.NS"), ("HINDALCO",   "HINDALCO.NS"),
    ("JSWSTEEL",   "JSWSTEEL.NS"),  ("SRF",        "SRF.NS"),
    ("PIDILITIND", "PIDILITIND.NS"),("AARTIIND",   "AARTIIND.NS"),
    # Pharma / Healthcare
    ("SUNPHARMA",  "SUNPHARMA.NS"), ("DRREDDY",    "DRREDDY.NS"),
    ("CIPLA",      "CIPLA.NS"),     ("DIVISLAB",   "DIVISLAB.NS"),
    ("APOLLOHOSP", "APOLLOHOSP.NS"),("TORNTPHARM", "TORNTPHARM.NS"),
    # Auto
    ("MARUTI",     "MARUTI.NS"),    ("TATAMOTORS", "TATAMOTORS.NS"),
    ("BAJAJ-AUTO", "BAJAJ-AUTO.NS"),("HEROMOTOCO", "HEROMOTOCO.NS"),
    ("EICHERMOT",  "EICHERMOT.NS"), ("M&M",        "M&M.NS"),
    # Telecom / Media
    ("BHARTIARTL", "BHARTIARTL.NS"),("INDUSINDBK", "INDUSINDBK.NS"),
    # Consumer Discretionary
    ("TITAN",      "TITAN.NS"),     ("TATACONSUM", "TATACONSUM.NS"),
    ("DMART",      "DMART.NS"),     ("NYKAA",      "NYKAA.NS"),
    # Cement
    ("ULTRACEMCO", "ULTRACEMCO.NS"),("AMBUJACEM",  "AMBUJACEM.NS"),
    ("ACC",        "ACC.NS"),       ("SHREECEM",   "SHREECEM.NS"),
    # Diversified
    ("ADANIENT",   "ADANIENT.NS"),  ("ADANIPORTS", "ADANIPORTS.NS"),
    ("GRASIM",     "GRASIM.NS"),    ("ITC",        "ITC.NS"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  FACTOR DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FactorDef:
    name:        str
    category:    str
    description: str       # economic rationale
    lookback:    int        # lookback in trading days (for price-based)


FACTORS: List[FactorDef] = [
    # ── Momentum ────────────────────────────────────────────────────────────
    FactorDef("mom_1m",   "Momentum",
              "1-month price return. Short-term momentum; captures recent trend strength.",
              21),
    FactorDef("mom_3m",   "Momentum",
              "3-month price return. Medium-term momentum; industry standard lookback.",
              63),
    FactorDef("mom_6m",   "Momentum",
              "6-month price return. Classic Jegadeesh-Titman factor (1993).",
              126),
    FactorDef("mom_12m",  "Momentum",
              "12-month return excluding last month (skip-1 to avoid reversal contamination).",
              252),

    # ── Value ────────────────────────────────────────────────────────────────
    FactorDef("book_to_price",  "Value",
              "Book value / Market cap. Classic value factor — Fama-French HML.",
              0),
    FactorDef("earnings_yield", "Value",
              "EPS (TTM) / Price. Inverse of P/E; higher = cheaper relative to earnings.",
              0),
    FactorDef("ev_to_ebitda",   "Value",
              "EV / EBITDA. Capital-structure neutral valuation — preferred in India research.",
              0),

    # ── Quality ──────────────────────────────────────────────────────────────
    FactorDef("roe",            "Quality",
              "Return on Equity (TTM). Measures management's efficiency in generating returns.",
              0),
    FactorDef("roce",           "Quality",
              "Return on Capital Employed. Capital-structure neutral quality measure.",
              0),
    FactorDef("asset_growth",   "Quality",
              "YoY asset growth (inverted). Abnormal asset growth predicts poor future returns — Cooper et al. (2008).",
              0),

    # ── Low Volatility ───────────────────────────────────────────────────────
    FactorDef("vol_1y",         "Low Volatility",
              "1-year realised daily return volatility (annualised). Low-vol anomaly.",
              252),
    FactorDef("beta_1y",        "Low Volatility",
              "Market beta over trailing 1 year. Low-beta stocks tend to outperform (Black 1972).",
              252),
]

FACTOR_NAMES: List[str] = [f.name for f in FACTORS]

# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
REBALANCE_FREQ:    str = "ME"          # Monthly rebalance (month-end)
FORWARD_RETURN_M:  int = 1             # 1-month forward return as target
HISTORY_YEARS:     int = 5             # years of history to fetch
N_DECILES:         int = 10
LONG_DECILE:       int = 1             # Decile 1 = bottom (sorted ascending; depends on factor sign)
SHORT_DECILE:      int = 10
TRANSACTION_COST:  float = 0.0015      # 15bps round-trip (realistic for NSE large-cap)
NIFTY50_TICKER:    str = "^NSEI"       # Benchmark

# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
IC_SIGNIFICANCE_THRESHOLD: float = 0.03  # |IC| > 0.03 considered meaningful
ICIR_SIGNIFICANCE:         float = 0.5   # IC / IC_std > 0.5 for production-grade factor
P_VALUE_THRESHOLD:         float = 0.10  # p-value for t-test on IC series

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
EXCEL_OUTPUT = "outputs/Alpha_Factor_Research.xlsx"
PDF_REPORT   = "outputs/Alpha_Factor_Report.pdf"
