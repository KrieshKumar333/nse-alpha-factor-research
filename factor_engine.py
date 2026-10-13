"""
factor_engine.py — Constructs, scores, and analyses all alpha factors.

Factor construction follows institutional conventions:
  - Cross-sectional z-score normalisation each period (mean=0, std=1)
  - Winsorisation at ±3σ before scoring (removes outlier contamination)
  - Factors are SIGNED so that positive score = expected to outperform
    (momentum factors: higher return = positive score)
    (value factors: higher B/P = positive score)
    (quality factors: higher ROE = positive score)
    (low-vol factors: lower volatility = positive score → inverted)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import FACTORS, FACTOR_NAMES, N_DECILES, REBALANCE_FREQ, IC_SIGNIFICANCE_THRESHOLD

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def _winsorise(s: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """Winsorise at ±n_sigma cross-sectionally."""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return s
    return s.clip(lower=mu - n_sigma * sigma, upper=mu + n_sigma * sigma)


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; returns NaN series if all values identical."""
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sigma


def _rank_normalise(s: pd.Series) -> pd.Series:
    """Rank-based z-score: robust to non-normal distributions."""
    ranked = s.rank(pct=True)   # uniform [0, 1]
    return _zscore(ranked)


# ─────────────────────────────────────────────────────────────────────────────
class FactorEngine:
    """
    Builds and analyses systematic alpha factors for the NSE equity universe.

    Inputs:
      monthly_prices  — stocks × months price matrix
      fundamentals    — stocks × fundamental columns (latest snapshot)
      benchmark_rets  — monthly benchmark return series
    """

    def __init__(
        self,
        monthly_prices:  pd.DataFrame,
        fundamentals:    pd.DataFrame,
        benchmark_rets:  pd.Series,
    ) -> None:
        self.prices     = monthly_prices
        self.funds      = fundamentals
        self.bench_rets = benchmark_rets
        self._raw_factors: Optional[Dict[str, pd.DataFrame]] = None
        self._scored:      Optional[Dict[str, pd.DataFrame]] = None

    # ── Factor construction ───────────────────────────────────────────────────

    def build_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Returns dict of factor_name → (months × stocks) DataFrame of raw scores.
        """
        if self._raw_factors is not None:
            return self._raw_factors

        prices = self.prices
        f      = self.funds   # single-period snapshot; applied uniformly across history

        raw: Dict[str, pd.DataFrame] = {}

        # ── Momentum factors (price-based, time-varying) ─────────────────────
        logger.info("Building momentum factors…")

        # 1-month momentum
        raw["mom_1m"]  = prices.pct_change(1)

        # 3-month momentum
        raw["mom_3m"]  = prices.pct_change(3)

        # 6-month momentum
        raw["mom_6m"]  = prices.pct_change(6)

        # 12-month momentum: skip the last month (industry standard)
        # Return[t-12 → t-1] — avoids short-term reversal contamination
        raw["mom_12m"] = prices.shift(1).pct_change(11)

        # ── Volatility & beta factors (rolling, time-varying) ────────────────
        logger.info("Building low-volatility factors…")

        # Realised 12-month daily volatility → monthly (approximate from monthly returns)
        monthly_rets = prices.pct_change(1)

        # 12-month rolling volatility (annualised)
        raw["vol_1y"] = monthly_rets.rolling(12, min_periods=6).std() * np.sqrt(12)

        # 12-month rolling beta vs equally-weighted portfolio
        mkt_proxy = monthly_rets.mean(axis=1)  # cross-sectional mean as market proxy
        betas: Dict[str, pd.Series] = {}
        for col in prices.columns:
            s = monthly_rets[col]
            cov_df = pd.concat([s, mkt_proxy], axis=1).dropna()
            if len(cov_df) < 12:
                betas[col] = pd.Series(1.0, index=monthly_rets.index)
                continue
            rolling_cov = cov_df.rolling(12, min_periods=6).cov().unstack()
            # .cov() returns pairs; we want Cov(stock, market) / Var(market)
            try:
                cov_sm  = cov_df.rolling(12, min_periods=6).cov().unstack().iloc[:, 1]
                var_mkt = mkt_proxy.rolling(12, min_periods=6).var()
                beta_s  = (cov_sm / var_mkt.replace(0, np.nan)).reindex(monthly_rets.index)
                betas[col] = beta_s.clip(-3, 5)  # sanity bounds
            except Exception:
                betas[col] = pd.Series(1.0, index=monthly_rets.index)

        raw["beta_1y"] = pd.DataFrame(betas, index=monthly_rets.index)

        # ── Value & quality factors (fundamental snapshot, static) ───────────
        logger.info("Building fundamental factors…")
        monthly_idx = prices.index
        stock_cols  = prices.columns

        def _broadcast(series_name: str) -> pd.DataFrame:
            """Broadcast a single-period fundamental vector across all months."""
            col = self.funds.get(series_name, pd.Series(dtype=float)) \
                  if isinstance(self.funds, pd.DataFrame) else pd.Series(dtype=float)
            if isinstance(self.funds, pd.DataFrame) and series_name in self.funds.columns:
                s = self.funds[series_name].reindex(stock_cols)
            else:
                s = pd.Series(np.nan, index=stock_cols)
            mat = pd.DataFrame(
                np.tile(s.values, (len(monthly_idx), 1)),
                index=monthly_idx, columns=stock_cols,
            )
            return mat

        raw["book_to_price"]  = _broadcast("book_to_price")
        raw["earnings_yield"] = _broadcast("earnings_yield")
        raw["roe"]            = _broadcast("roe")
        raw["roce"]           = _broadcast("roce")

        # ev_to_ebitda: lower = cheaper → invert for positive sign convention
        raw["ev_to_ebitda"] = -_broadcast("ev_to_ebitda")

        # asset_growth: high growth → typically poor returns → invert
        raw["asset_growth"] = -_broadcast("asset_growth")

        self._raw_factors = raw
        logger.info(f"Built {len(raw)} raw factor matrices")
        return raw

    def score_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Returns dict of factor_name → (months × stocks) DataFrame of
        cross-sectionally z-scored, winsorised scores.

        Low-volatility factors are INVERTED before scoring (low vol = positive score).
        """
        if self._scored is not None:
            return self._scored

        raw = self.build_factors()
        scored: Dict[str, pd.DataFrame] = {}

        # Low-vol factors: invert so that LOW volatility = HIGH (positive) score
        invert_set = {"vol_1y", "beta_1y", "ev_to_ebitda", "asset_growth"}

        for name, mat in raw.items():
            sign = -1 if name in invert_set else 1
            scored_rows = []
            for dt in mat.index:
                row = mat.loc[dt].dropna()
                if len(row) < 10:   # need at least 10 stocks for meaningful cross-section
                    scored_rows.append(pd.Series(np.nan, index=mat.columns))
                    continue
                row_w  = _winsorise(row)
                row_z  = _rank_normalise(row_w) * sign
                scored_rows.append(row_z.reindex(mat.columns))

            scored[name] = pd.DataFrame(scored_rows, index=mat.index)

        self._scored = scored
        logger.info("Factor scoring complete (cross-sectional z-score + rank normalisation)")
        return scored

    # ── Information Coefficient analysis ─────────────────────────────────────

    def compute_ic_series(self) -> Dict[str, pd.Series]:
        """
        Compute monthly Spearman IC (factor score vs 1-month forward return).
        Spearman IC preferred over Pearson: robust to outliers in returns.
        """
        scored   = self.score_factors()
        fwd_ret  = self.prices.pct_change(1).shift(-1)  # 1-month forward

        ic_dict: Dict[str, pd.Series] = {}

        for name, mat in scored.items():
            monthly_ic = []
            for dt in mat.index:
                scores_t = mat.loc[dt].dropna()
                fwd_t    = fwd_ret.loc[dt] if dt in fwd_ret.index else pd.Series(dtype=float)
                common   = scores_t.index.intersection(fwd_t.dropna().index)
                if len(common) < 10:
                    monthly_ic.append(np.nan)
                    continue
                ic_val, _ = stats.spearmanr(scores_t[common], fwd_t[common])
                monthly_ic.append(ic_val)

            ic_s = pd.Series(monthly_ic, index=mat.index, name=name).dropna()
            ic_dict[name] = ic_s

        logger.info("IC series computed for all factors")
        return ic_dict

    def ic_summary(self) -> pd.DataFrame:
        """
        Returns a summary table of IC statistics for all factors.
        Columns: Mean IC, IC Std, ICIR, t-stat, p-value, Hit Rate, % positive.
        """
        ic_dict = self.compute_ic_series()
        rows = []

        for name, ic_s in ic_dict.items():
            if len(ic_s) < 6:
                continue
            mean_ic = ic_s.mean()
            ic_std  = ic_s.std()
            icir    = mean_ic / ic_std if ic_std > 0 else np.nan
            t_stat  = mean_ic / (ic_std / np.sqrt(len(ic_s))) if ic_std > 0 else np.nan
            p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ic_s)-1)) if not np.isnan(t_stat) else np.nan
            hit_rate= (ic_s > 0).mean()

            # Find factor category
            cat = next((f.category for f in FACTORS if f.name == name), "Unknown")
            desc= next((f.description[:60] for f in FACTORS if f.name == name), "")

            rows.append({
                "Factor":       name,
                "Category":     cat,
                "Mean IC":      round(mean_ic, 4),
                "IC Std":       round(ic_std, 4),
                "ICIR":         round(icir, 3)   if not np.isnan(icir) else np.nan,
                "t-statistic":  round(t_stat, 2) if not np.isnan(t_stat) else np.nan,
                "p-value":      round(p_val, 4)  if not np.isnan(p_val) else np.nan,
                "Hit Rate":     round(hit_rate, 3),
                "Significant":  bool(abs(mean_ic) > IC_SIGNIFICANCE_THRESHOLD and
                                     p_val is not np.nan and p_val < 0.10),
                "Description":  desc,
            })

        df = pd.DataFrame(rows).set_index("Factor")
        df = df.sort_values("ICIR", ascending=False, key=abs)
        logger.info(f"IC summary: {df['Significant'].sum()} / {len(df)} factors significant")
        return df

    # ── Decile portfolio construction ─────────────────────────────────────────

    def decile_returns(self, factor_name: str) -> pd.DataFrame:
        """
        Compute monthly equal-weighted returns for each decile portfolio.
        Returns DataFrame (months × 10 deciles).
        """
        scored  = self.score_factors()
        mat     = scored[factor_name]
        fwd_ret = self.prices.pct_change(1).shift(-1)

        decile_rets: List[Dict] = []
        for dt in mat.index:
            scores_t = mat.loc[dt].dropna()
            fwd_t    = fwd_ret.loc[dt] if dt in fwd_ret.index else pd.Series(dtype=float)
            common   = scores_t.index.intersection(fwd_t.dropna().index)
            if len(common) < N_DECILES * 2:
                decile_rets.append({f"D{d}": np.nan for d in range(1, N_DECILES+1)})
                continue
            df_t    = pd.DataFrame({"score": scores_t[common], "ret": fwd_t[common]})
            df_t["decile"] = pd.qcut(df_t["score"].rank(method="first"), N_DECILES, labels=False) + 1
            d_mean  = df_t.groupby("decile")["ret"].mean()
            decile_rets.append({f"D{int(d)}": v for d, v in d_mean.items()})

        return pd.DataFrame(decile_rets, index=mat.index)

    def long_short_returns(
        self, factor_name: str, tc: float = 0.0015
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Returns (long_short_returns, cumulative_ls_returns) after transaction costs.
        Long: top decile (D10 = highest score = expected outperformers)
        Short: bottom decile (D1)
        """
        d_rets = self.decile_returns(factor_name).dropna()
        ls     = d_rets["D10"] - d_rets["D1"] - 2 * tc   # both legs bear tc
        cumls  = (1 + ls).cumprod() - 1
        return ls, cumls

    def decile_cumulative_returns(self, factor_name: str) -> pd.DataFrame:
        """Cumulative returns for all 10 deciles."""
        d_rets = self.decile_returns(factor_name).dropna()
        return (1 + d_rets).cumprod() - 1

    def factor_correlation_matrix(self) -> pd.DataFrame:
        """Cross-factor correlation using IC series — shows diversification potential."""
        ic_dict = self.compute_ic_series()
        ic_df   = pd.DataFrame(ic_dict).dropna(how="all")
        return ic_df.corr(method="pearson")
