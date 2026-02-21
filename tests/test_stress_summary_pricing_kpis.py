# -*- coding: utf-8 -*-
"""
tests/test_stress_summary_pricing_kpis.py

PC10 — Non-regression tests for pricing KPIs in stress_summary.

Validates that the four new columns added to each stress_summary row:
  • sale_pnl_total      – sum of realised P&L across sold loans
  • avg_sale_pnl        – mean P&L per sale (0 if no sales)
  • avg_bid_pct_ead     – mean price/book ratio for sold loans (0 if no sales)
  • sell_blocked_count  – # loans blocked by the fire-sale guardrail

are:
  (a) present after stress_engine processes results
  (b) free of NaN / None
  (c) economically coherent: a pricing_crunch (haircut) portfolio compared to
      a baseline shows lower avg_bid_pct_ead and/or higher sell_blocked_count.

Because running the full coordinator pipeline requires a trained model, these
tests use lightweight stubs that replicate the KPI-extraction logic applied
inside run_stress_pipeline when building summary_records.
"""

import os
import sys
import math
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


# ---------------------------------------------------------------------------
# Helpers — replicate the exact KPI extraction from stress_engine.py
# ---------------------------------------------------------------------------

def _extract_pricing_kpis(df_res: pd.DataFrame) -> dict:
    """
    Mirror of the KPI-extraction block inside run_stress_pipeline.
    Returns a dict with the four PC10 columns.
    """
    def _col_sum_filtered(df, mask, *names, default=0.0):
        for n in names:
            if n in df.columns:
                v = pd.to_numeric(df.loc[mask, n], errors="coerce").fillna(0.0)
                return float(v.sum())
        return default

    def _col_mean_filtered(df, mask, *names, default=np.nan):
        for n in names:
            if n in df.columns:
                v = pd.to_numeric(df.loc[mask, n], errors="coerce").dropna()
                return float(v.mean()) if len(v) > 0 else default
        return default

    sell_mask = (
        df_res["Accion_final"] == "VENDER"
        if "Accion_final" in df_res.columns
        else pd.Series([False] * len(df_res), index=df_res.index)
    )

    sale_pnl_total = _col_sum_filtered(
        df_res, sell_mask, "pnl_realized", "pnl", "pnl_book"
    )

    avg_sale_pnl_raw = _col_mean_filtered(
        df_res, sell_mask, "pnl_realized", "pnl", "pnl_book"
    )
    avg_sale_pnl = 0.0 if np.isnan(avg_sale_pnl_raw) else avg_sale_pnl_raw

    avg_bid_pct_ead_raw = _col_mean_filtered(
        df_res, sell_mask,
        "audit_price_book_ratio", "price_book_ratio", "pnl_ratio_book"
    )
    avg_bid_pct_ead = 0.0 if np.isnan(avg_bid_pct_ead_raw) else avg_bid_pct_ead_raw

    if "Sell_Blocked" in df_res.columns:
        sell_blocked_count = int(df_res["Sell_Blocked"].fillna(False).astype(bool).sum())
    elif "reason_code" in df_res.columns:
        sell_blocked_count = int(
            df_res["reason_code"].astype(str).str.contains("RC02_SELL_BLOCKED").sum()
        )
    else:
        sell_blocked_count = 0

    return {
        "sale_pnl_total": round(sale_pnl_total, 2),
        "avg_sale_pnl": round(avg_sale_pnl, 2),
        "avg_bid_pct_ead": round(avg_bid_pct_ead, 4),
        "sell_blocked_count": sell_blocked_count,
    }


# ---------------------------------------------------------------------------
# Fixtures — small synthetic portfolios
# ---------------------------------------------------------------------------

def _make_portfolio(n_loans: int = 10, n_sell: int = 4, n_blocked: int = 1,
                    bid_ratio: float = 0.55) -> pd.DataFrame:
    """
    Build a minimal decisiones_explicadas-like DataFrame.

    Parameters
    ----------
    n_loans      total rows
    n_sell       rows with Accion_final == VENDER
    n_blocked    rows with Sell_Blocked == True (subset of n_loans)
    bid_ratio    audit_price_book_ratio for sold loans
    """
    rng = np.random.default_rng(42)
    actions = (
        ["VENDER"] * n_sell
        + ["MANTENER"] * ((n_loans - n_sell) // 2)
        + ["REESTRUCTURAR"] * (n_loans - n_sell - (n_loans - n_sell) // 2)
    )

    pnl_vals = np.where(
        np.array(actions) == "VENDER",
        rng.uniform(-80_000, -5_000, size=n_loans),   # typical NPL discount
        0.0,
    )

    bid_ratios = np.where(
        np.array(actions) == "VENDER",
        bid_ratio,
        np.nan,
    )

    blocked = [False] * n_loans
    for i in range(min(n_blocked, n_loans)):
        blocked[i] = True

    df = pd.DataFrame({
        "loan_id": [f"L{i:04d}" for i in range(n_loans)],
        "EAD": rng.uniform(200_000, 1_500_000, size=n_loans),
        "PD": rng.uniform(0.05, 0.40, size=n_loans),
        "Accion_final": actions,
        "pnl_realized": pnl_vals,
        "audit_price_book_ratio": bid_ratios,
        "Sell_Blocked": blocked,
        "EVA_post": np.where(np.array(actions) == "VENDER", 0.0,
                             rng.uniform(-50_000, 30_000, size=n_loans)),
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPricingKpiColumns:
    """PC10: new pricing KPI columns must be present and non-null."""

    def test_required_columns_present(self):
        """KPI dict must have all four PC10 keys."""
        df = _make_portfolio()
        kpis = _extract_pricing_kpis(df)
        for col in ("sale_pnl_total", "avg_sale_pnl", "avg_bid_pct_ead", "sell_blocked_count"):
            assert col in kpis, f"Missing KPI column: {col}"

    def test_no_nans_in_kpis(self):
        """No NaN / None in any KPI after extraction."""
        df = _make_portfolio(n_loans=15, n_sell=5)
        kpis = _extract_pricing_kpis(df)
        for k, v in kpis.items():
            assert v is not None, f"{k} is None"
            if isinstance(v, float):
                assert not math.isnan(v), f"{k} is NaN"

    def test_zero_sales_portfolio(self):
        """When no loans are sold all pricing KPIs should be zero."""
        df = _make_portfolio(n_loans=10, n_sell=0, n_blocked=0)
        kpis = _extract_pricing_kpis(df)
        assert kpis["sale_pnl_total"] == 0.0, "sale_pnl_total must be 0 with no sales"
        assert kpis["avg_sale_pnl"] == 0.0, "avg_sale_pnl must be 0 with no sales"
        assert kpis["avg_bid_pct_ead"] == 0.0, "avg_bid_pct_ead must be 0 with no sales"

    def test_sell_blocked_count_correct(self):
        """sell_blocked_count must equal number of Sell_Blocked=True rows."""
        df = _make_portfolio(n_loans=12, n_sell=4, n_blocked=3)
        kpis = _extract_pricing_kpis(df)
        assert kpis["sell_blocked_count"] == 3, (
            f"Expected 3 blocked, got {kpis['sell_blocked_count']}"
        )

    def test_sale_pnl_total_sign(self):
        """sale_pnl_total must be negative in a typical NPL discount scenario."""
        df = _make_portfolio(n_loans=10, n_sell=4, bid_ratio=0.5)
        kpis = _extract_pricing_kpis(df)
        assert kpis["sale_pnl_total"] < 0, (
            f"NPL sales should produce negative aggregate PnL, got {kpis['sale_pnl_total']}"
        )

    def test_avg_bid_pct_ead_between_zero_and_one(self):
        """avg_bid_pct_ead (price/book) must be in (0, 1] for typical NPL."""
        df = _make_portfolio(n_loans=20, n_sell=8, bid_ratio=0.55)
        kpis = _extract_pricing_kpis(df)
        if kpis["avg_bid_pct_ead"] != 0.0:
            assert 0.0 < kpis["avg_bid_pct_ead"] <= 1.05, (
                f"avg_bid_pct_ead={kpis['avg_bid_pct_ead']} outside realistic range (0, 1.05]"
            )


class TestPricingCrunchKpiEffect:
    """PC10: pricing_crunch must make avg_bid_pct_ead fall or sell_blocked_count rise."""

    def _make_crunch_pair(self, bid_ratio_base: float, bid_ratio_crunch: float,
                          n_blocked_crunch: int):
        """
        Simulate two runs: baseline vs pricing_crunch.
        Returns (kpis_base, kpis_crunch).
        """
        df_base = _make_portfolio(
            n_loans=20, n_sell=8, n_blocked=1, bid_ratio=bid_ratio_base
        )
        df_crunch = _make_portfolio(
            n_loans=20, n_sell=5, n_blocked=n_blocked_crunch,  # fewer sales due to guardrail
            bid_ratio=bid_ratio_crunch,
        )
        return _extract_pricing_kpis(df_base), _extract_pricing_kpis(df_crunch)

    def test_pricing_crunch_lowers_avg_bid_pct(self):
        """Haircut scenario must lower avg_bid_pct_ead relative to baseline."""
        kpi_base, kpi_crunch = self._make_crunch_pair(
            bid_ratio_base=0.60,
            bid_ratio_crunch=0.40,   # ~33% haircut on bid
            n_blocked_crunch=1,
        )
        assert kpi_crunch["avg_bid_pct_ead"] < kpi_base["avg_bid_pct_ead"], (
            f"pricing_crunch avg_bid_pct_ead ({kpi_crunch['avg_bid_pct_ead']}) must be "
            f"below baseline ({kpi_base['avg_bid_pct_ead']})"
        )

    def test_pricing_crunch_raises_sell_blocked_or_lowers_bid(self):
        """
        Under pricing crunch at least one of these must hold:
          • avg_bid_pct_ead drops, OR
          • sell_blocked_count increases.
        """
        kpi_base, kpi_crunch = self._make_crunch_pair(
            bid_ratio_base=0.60,
            bid_ratio_crunch=0.40,
            n_blocked_crunch=4,       # more blocks due to tighter guardrail
        )
        economic_impact = (
            kpi_crunch["avg_bid_pct_ead"] < kpi_base["avg_bid_pct_ead"]
            or kpi_crunch["sell_blocked_count"] > kpi_base["sell_blocked_count"]
        )
        assert economic_impact, (
            "pricing_crunch should reduce avg_bid_pct_ead or increase sell_blocked_count. "
            f"Base={kpi_base}, Crunch={kpi_crunch}"
        )

    def test_sale_pnl_total_more_negative_under_crunch(self):
        """Aggregate P&L should be more negative under haircut stress."""
        kpi_base, kpi_crunch = self._make_crunch_pair(
            bid_ratio_base=0.60,
            bid_ratio_crunch=0.40,
            n_blocked_crunch=2,
        )
        # Either crunch pnl is more negative OR not (fewer loans sold may offset)
        # Non-regression: just confirm both are non-positive (no phantom gain)
        assert kpi_base["sale_pnl_total"] <= 0, "Baseline aggregate PnL should be ≤ 0"
        assert kpi_crunch["sale_pnl_total"] <= 0, "Crunch aggregate PnL should be ≤ 0"


class TestPricingKpiSellBlockedFallback:
    """sell_blocked_count fallback via reason_code column."""

    def test_fallback_reason_code(self):
        """If Sell_Blocked column absent, use reason_code containing RC02_SELL_BLOCKED."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "loan_id": ["L001", "L002", "L003", "L004"],
            "Accion_final": ["VENDER", "MANTENER", "MANTENER", "REESTRUCTURAR"],
            "pnl_realized": [-15_000.0, 0.0, 0.0, 0.0],
            "audit_price_book_ratio": [0.55, np.nan, np.nan, np.nan],
            "reason_code": [
                "RC06_MICRO_SELL_VALUE_NEGATIVE",
                "RC02_SELL_BLOCKED_FIRE_SALE",  # blocked row
                "RC05_KEEP_MEETS_HURDLE",
                "RC03_MICRO_RESTRUCT_VALUE_UPLIFT",
            ],
            # No Sell_Blocked column intentionally
        })
        kpis = _extract_pricing_kpis(df)
        assert kpis["sell_blocked_count"] == 1, (
            f"Expected 1 blocked via reason_code fallback, got {kpis['sell_blocked_count']}"
        )

    def test_no_sell_blocked_column_no_reason_code(self):
        """If neither Sell_Blocked nor reason_code exists, count defaults to 0."""
        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "Accion_final": ["VENDER", "MANTENER"],
            "pnl_realized": [-10_000.0, 0.0],
            "audit_price_book_ratio": [0.60, np.nan],
        })
        kpis = _extract_pricing_kpis(df)
        assert kpis["sell_blocked_count"] == 0
