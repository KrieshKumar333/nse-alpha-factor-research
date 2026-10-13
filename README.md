# NSE Alpha Factor Research

Built this project to bridge my WorldQuant BRAIN work with a structured academic
framework. On WorldQuant, I designed and tested alpha strategies using IC and ICIR
as quality metrics - the same institutional framework used by systematic quant funds.
I wanted to run that methodology transparently on Indian equities with proper
academic factor definitions and full statistical rigour.

The Indian equity market is interesting for factor research: ~60% retail participation
by turnover, thin analyst coverage on mid-caps, and lower institutional arbitrage
capacity than developed markets. These conditions make factors like momentum
more persistent - and the results confirm that.

## What I Built

Systematic alpha factor research on a 62-stock NSE 500 representative universe:

**12 Factors Tested Across 4 Categories:**

| Category | Factors | Academic Reference |
|----------|---------|-------------------|
| Momentum (4) | 1M, 3M, 6M, 12M skip-1 price return | Jegadeesh and Titman (1993) |
| Value (3) | Book-to-price, Earnings yield, inv. EV/EBITDA | Fama and French (1992) |
| Quality (3) | ROE, ROCE, inverted asset growth | Cooper, Gulen and Schill (2008) |
| Low Volatility (2) | 1Y realised vol, 1Y market beta | Ang et al. (2006) |

**Methodology:**
- Monthly cross-sectional **Spearman IC** (robust to return outliers - industry standard)
- Winsorisation at +-3 sigma -> rank normalisation -> z-score per month
- **ICIR** = IC Mean / IC Std (signal consistency metric)
- Statistical significance: t-test on IC series (p < 0.10 threshold)
- Decile long-short portfolios (D10 - D1) with 15bps round-trip transaction costs
- Factor correlation matrix for multi-factor composite construction

**Interactive Streamlit dashboard** with IC time series, decile return charts, factor correlations

## Key Findings

- **6M momentum** = strongest factor (consistent with India's retail-driven market structure)
- **Average inter-factor IC correlation: ~0.22** - low enough to support a composite
- **Recommended composite**: 40% 6M momentum + 30% earnings yield + 30% ROCE

**Note on mock data**: IC values are near-zero when running `--mock`.
This is expected - synthetic prices have no factor signal by construction.
Run without `--mock` for real results using live Yahoo Finance data (3-5 min fetch).

## Connection to WorldQuant BRAIN

WorldQuant BRAIN uses the exact same IC/ICIR evaluation framework to screen
alpha strategies for live trading. Achieving **Gold Level** required designing
strategies passing institutional thresholds for IC, ICIR, turnover, and Sharpe.
This project applies that framework to Indian equities with academic factor
definitions - extending it beyond the price-data-only BRAIN platform by adding
fundamental value and quality factors.

## Setup

```bash
pip install -r requirements.txt

# Run with live NSE data via Yahoo Finance (recommended - takes 3-5 mins)
python run.py

# Run with offline synthetic data
python run.py --mock

# Interactive dashboard
python -m streamlit run dashboard.py
```

## Tech Stack

`Python` `pandas` `NumPy` `SciPy` `scikit-learn` `yfinance` `Streamlit` `Plotly`

## Project Structure

```
nse-alpha-factor-research/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- run.py                     # CLI runner
|-- dashboard.py               # Streamlit dashboard (5 tabs)
|-- __init__.py
|-- config.py                  # 62-stock universe, 12 factor definitions
|-- data_fetcher.py            # Price + fundamental data, mock generator
|-- factor_engine.py           # IC/ICIR, decile portfolios, L/S backtests
`-- Alpha_Factor_Research.xlsx # Output results workbook
```

## Academic References

- Jegadeesh and Titman (1993) - Returns to Buying Winners and Selling Losers
- Fama and French (1992, 2015) - Cross-Section of Expected Stock Returns
- Cooper, Gulen and Schill (2008) - Asset Growth and the Cross-Section of Returns
- Ang, Hodrick, Xing and Zhang (2006) - The Cross-Section of Volatility and Expected Returns
- Grinold (1989) - The Fundamental Law of Active Management

## Disclaimer

Academic research for personal learning. Not investment advice.
Not SEBI-registered research.
