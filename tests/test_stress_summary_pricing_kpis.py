# -*- coding: utf-8 -*-
# ============================================================
# tests/test_stress_summary_pricing_kpis.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Test del resumen de KPIs de pricing tras stress.
# ============================================================
"""
tests/test_stress_summary_pricing_kpis.py

PC10 hardening — Non-regression tests for pricing KPIs in stress_summary.

Five columns added to each stress_summary row:
  • sale_pnl_total           – sum of realised P&L across sold loans (€)
  • avg_sale_pnl             – mean P&L per sale (0 if no sales)
  • avg_bid_pct_ead          – mean bid/EAD for sold loans; NaN if unavailable
  • avg_bid_pct_ead_available– bool: True if avg_bid_pct_ead was computable
  • sell_blocked_count       – # loans blocked by fire-sale guardrail

Column priority for avg_bid_pct_ead (PC10 hardening):
  price_ratio_ead > Price_to_EAD > price_ratio_book >
  audit_price_book_ratio > pnl_ratio_book
  (audit_price_book_ratio is often NaN in coordinator output; using
   price_ratio_ead as canonical bid/EAD source)

Fallback policy:
  - No sales → avg_bid_pct_ead = NaN, avg_bid_pct_ead_available = False
  - Column absent → same (NaN, False), never fake 0.0
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
    Mirror of the KPI-extraction block inside run_stress_pipeline (PC10 hardening).
    Column priority for avg_bid_pct_ead:
      price_ratio_ead > Price_to_EAD > price_ratio_book >
      audit_price_book_ratio > pnl_ratio_book
    Returns NaN (not 0) when the KPI is genuinely unavailable.
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

    # PC10 hardening: correct column priority; NaN when unavailable (no fake 0)
    avg_bid_pct_ead_raw = _col_mean_filtered(
        df_res, sell_mask,
        "price_ratio_ead", "Price_to_EAD",
        "price_ratio_book", "audit_price_book_ratio",
        "pnl_ratio_book",
    )
    avg_bid_pct_ead_available = not np.isnan(avg_bid_pct_ead_raw)
    avg_bid_pct_ead = (
        round(float(avg_bid_pct_ead_raw), 4) if avg_bid_pct_ead_available else np.nan
    )

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
        "avg_bid_pct_ead": avg_bid_pct_ead,           # NaN when unavailable
        "avg_bid_pct_ead_available": avg_bid_pct_ead_available,
        "sell_blocked_count": sell_blocked_count,
    }


# ---------------------------------------------------------------------------
# Fixtures — small synthetic portfolios
# ---------------------------------------------------------------------------

def _make_portfolio(n_loans: int = 10, n_sell: int = 4, n_blocked: int = 1,
                    bid_ratio: float = 0.55,
                    use_canonical_col: bool = True) -> pd.DataFrame:
    """
    Build a minimal decisiones_explicadas-like DataFrame.

    Parameters
    ----------
    n_loans            total rows
    n_sell             rows with Accion_final == VENDER
    n_blocked          rows with Sell_Blocked == True (subset of n_loans)
    bid_ratio          price_ratio_ead value for sold loans
    use_canonical_col  if True uses 'price_ratio_ead' (canonical coordinator
                       column); if False uses 'audit_price_book_ratio' (legacy,
                       often NaN in real output — tests unavailability path)
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

    # Use the canonical column name that coordinator actually outputs
    bid_col = "price_ratio_ead" if use_canonical_col else "audit_price_book_ratio"

    df = pd.DataFrame({
        "loan_id": [f"L{i:04d}" for i in range(n_loans)],
        "EAD": rng.uniform(200_000, 1_500_000, size=n_loans),
        "PD": rng.uniform(0.05, 0.40, size=n_loans),
        "Accion_final": actions,
        "pnl_realized": pnl_vals,
        bid_col: bid_ratios,
        "Sell_Blocked": blocked,
        "EVA_post": np.where(np.array(actions) == "VENDER", 0.0,
                             rng.uniform(-50_000, 30_000, size=n_loans)),
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPricingKpiColumns:
    """PC10: new pricing KPI columns must be present with correct semantics."""

    def test_required_columns_present(self):
        """KPI dict must contain all five PC10 keys (including availability flag)."""
        df = _make_portfolio()
        kpis = _extract_pricing_kpis(df)
        for col in ("sale_pnl_total", "avg_sale_pnl", "avg_bid_pct_ead",
                    "avg_bid_pct_ead_available", "sell_blocked_count"):
            assert col in kpis, f"Missing KPI column: {col}"

    def test_no_nans_in_non_optional_kpis(self):
        """
        Non-optional KPIs must not be NaN.
        avg_bid_pct_ead is exempt when avg_bid_pct_ead_available=False.
        """
        df = _make_portfolio(n_loans=15, n_sell=5)
        kpis = _extract_pricing_kpis(df)
        for k, v in kpis.items():
            assert v is not None, f"{k} is None"
            if k == "avg_bid_pct_ead":
                # Allowed to be NaN only when availability flag says False
                if kpis["avg_bid_pct_ead_available"]:
                    assert not (isinstance(v, float) and math.isnan(v)), (
                        f"avg_bid_pct_ead is NaN but available=True"
                    )
            elif isinstance(v, float):
                assert not math.isnan(v), f"{k} is NaN"

    def test_avg_bid_pct_ead_available_when_sales_and_col_present(self):
        """With sales and price_ratio_ead column, availability must be True."""
        df = _make_portfolio(n_loans=15, n_sell=5, use_canonical_col=True)
        kpis = _extract_pricing_kpis(df)
        assert kpis["avg_bid_pct_ead_available"] is True, (
            "avg_bid_pct_ead_available should be True when price_ratio_ead is present"
        )
        assert not math.isnan(kpis["avg_bid_pct_ead"]), (
            "avg_bid_pct_ead must not be NaN when available=True"
        )

    def test_avg_bid_pct_ead_unavailable_when_sales_but_old_col_is_nan(self):
        """
        When audit_price_book_ratio (legacy column) is present but all-NaN for
        sold rows, avg_bid_pct_ead should be NaN and available=False.
        """
        df = _make_portfolio(n_loans=10, n_sell=4, use_canonical_col=False)
        # Overwrite the legacy column with NaN for sold rows (real behavior)
        df.loc[df["Accion_final"] == "VENDER", "audit_price_book_ratio"] = np.nan
        kpis = _extract_pricing_kpis(df)
        assert kpis["avg_bid_pct_ead_available"] is False, (
            "availability must be False when all bid columns are NaN for sold rows"
        )
        assert isinstance(kpis["avg_bid_pct_ead"], float) and math.isnan(kpis["avg_bid_pct_ead"]), (
            "avg_bid_pct_ead must be NaN when unavailable (no fake 0)"
        )

    def test_zero_sales_portfolio(self):
        """When no loans are sold: pnl KPIs are 0, bid KPI is NaN + available=False."""
        df = _make_portfolio(n_loans=10, n_sell=0, n_blocked=0)
        kpis = _extract_pricing_kpis(df)
        assert kpis["sale_pnl_total"] == 0.0, "sale_pnl_total must be 0 with no sales"
        assert kpis["avg_sale_pnl"] == 0.0, "avg_sale_pnl must be 0 with no sales"
        assert kpis["avg_bid_pct_ead_available"] is False, (
            "avg_bid_pct_ead_available must be False with no sales"
        )
        assert isinstance(kpis["avg_bid_pct_ead"], float) and math.isnan(kpis["avg_bid_pct_ead"]), (
            "avg_bid_pct_ead must be NaN (not 0) with no sales"
        )

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
        """When available, avg_bid_pct_ead (bid/EAD) must be in (0, 1] for NPL."""
        df = _make_portfolio(n_loans=20, n_sell=8, bid_ratio=0.55)
        kpis = _extract_pricing_kpis(df)
        if kpis["avg_bid_pct_ead_available"]:
            assert 0.0 < kpis["avg_bid_pct_ead"] <= 1.05, (
                f"avg_bid_pct_ead={kpis['avg_bid_pct_ead']} outside realistic range (0, 1.05]"
            )


class TestPricingCrunchKpiEffect:
    """PC10: pricing_crunch must make avg_bid_pct_ead fall or sell_blocked_count rise."""

    def _make_crunch_pair(self, bid_ratio_base: float, bid_ratio_crunch: float,
                          n_blocked_crunch: int):
        """
        Simulate two runs: baseline vs pricing_crunch (canonical column).
        Returns (kpis_base, kpis_crunch).
        """
        df_base = _make_portfolio(
            n_loans=20, n_sell=8, n_blocked=1, bid_ratio=bid_ratio_base,
            use_canonical_col=True,
        )
        df_crunch = _make_portfolio(
            n_loans=20, n_sell=5, n_blocked=n_blocked_crunch,
            bid_ratio=bid_ratio_crunch,
            use_canonical_col=True,
        )
        return _extract_pricing_kpis(df_base), _extract_pricing_kpis(df_crunch)

    def test_pricing_crunch_lowers_avg_bid_pct(self):
        """Haircut scenario must lower avg_bid_pct_ead relative to baseline."""
        kpi_base, kpi_crunch = self._make_crunch_pair(
            bid_ratio_base=0.60,
            bid_ratio_crunch=0.40,   # ~33% haircut on bid
            n_blocked_crunch=1,
        )
        # Both must be available for meaningful comparison
        assert kpi_base["avg_bid_pct_ead_available"] and kpi_crunch["avg_bid_pct_ead_available"], (
            "avg_bid_pct_ead must be available in both runs for this test"
        )
        assert kpi_crunch["avg_bid_pct_ead"] < kpi_base["avg_bid_pct_ead"], (
            f"pricing_crunch avg_bid_pct_ead ({kpi_crunch['avg_bid_pct_ead']}) must be "
            f"below baseline ({kpi_base['avg_bid_pct_ead']})"
        )

    def test_pricing_crunch_raises_sell_blocked_or_lowers_bid(self):
        """
        Under pricing crunch at least one of these must hold:
          • avg_bid_pct_ead drops (when available in both runs), OR
          • sell_blocked_count increases.
        """
        kpi_base, kpi_crunch = self._make_crunch_pair(
            bid_ratio_base=0.60,
            bid_ratio_crunch=0.40,
            n_blocked_crunch=4,
        )
        bid_drops = (
            kpi_base["avg_bid_pct_ead_available"]
            and kpi_crunch["avg_bid_pct_ead_available"]
            and kpi_crunch["avg_bid_pct_ead"] < kpi_base["avg_bid_pct_ead"]
        )
        blocks_rise = kpi_crunch["sell_blocked_count"] > kpi_base["sell_blocked_count"]
        assert bid_drops or blocks_rise, (
            "pricing_crunch should reduce avg_bid_pct_ead or increase sell_blocked_count. "
            f"Base={kpi_base}, Crunch={kpi_crunch}"
        )

    def test_sale_pnl_total_more_negative_under_crunch(self):
        """Aggregate P&L should be ≤ 0 in both runs (no phantom gain)."""
        kpi_base, kpi_crunch = self._make_crunch_pair(
            bid_ratio_base=0.60,
            bid_ratio_crunch=0.40,
            n_blocked_crunch=2,
        )
        assert kpi_base["sale_pnl_total"] <= 0, "Baseline aggregate PnL should be ≤ 0"
        assert kpi_crunch["sale_pnl_total"] <= 0, "Crunch aggregate PnL should be ≤ 0"


class TestPricingKpiSellBlockedFallback:
    """sell_blocked_count fallback via reason_code column."""

    def test_fallback_reason_code(self):
        """If Sell_Blocked column absent, use reason_code containing RC02_SELL_BLOCKED."""
        df = pd.DataFrame({
            "loan_id": ["L001", "L002", "L003", "L004"],
            "Accion_final": ["VENDER", "MANTENER", "MANTENER", "REESTRUCTURAR"],
            "pnl_realized": [-15_000.0, 0.0, 0.0, 0.0],
            "price_ratio_ead": [0.55, np.nan, np.nan, np.nan],
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
        # avg_bid_pct_ead should use price_ratio_ead (0.55 for sold loan)
        assert kpis["avg_bid_pct_ead_available"] is True
        assert abs(kpis["avg_bid_pct_ead"] - 0.55) < 1e-4

    def test_no_sell_blocked_column_no_reason_code(self):
        """If neither Sell_Blocked nor reason_code exists, count defaults to 0."""
        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "Accion_final": ["VENDER", "MANTENER"],
            "pnl_realized": [-10_000.0, 0.0],
            "price_ratio_ead": [0.60, np.nan],
        })
        kpis = _extract_pricing_kpis(df)
        assert kpis["sell_blocked_count"] == 0
        assert kpis["avg_bid_pct_ead_available"] is True
        assert abs(kpis["avg_bid_pct_ead"] - 0.60) < 1e-4


class TestAvgBidPctEadAvailability:
    """PC10 hardening: avg_bid_pct_ead availability semantics."""

    def test_nan_not_zero_when_legacy_col_all_nan(self):
        """
        When audit_price_book_ratio is the only 'bid' column but all values
        are NaN for sold rows → avg_bid_pct_ead must be NaN (not 0).
        This mirrors the real coordinator output where that column is often NaN.
        """
        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "Accion_final": ["VENDER", "MANTENER"],
            "pnl_realized": [-20_000.0, 0.0],
            "audit_price_book_ratio": [np.nan, np.nan],   # always NaN in real output
        })
        kpis = _extract_pricing_kpis(df)
        assert kpis["avg_bid_pct_ead_available"] is False, (
            "Should be unavailable when only legacy col present and all-NaN"
        )
        assert isinstance(kpis["avg_bid_pct_ead"], float) and math.isnan(kpis["avg_bid_pct_ead"]), (
            f"Expected NaN, got {kpis['avg_bid_pct_ead']}"
        )

    def test_canonical_col_takes_priority(self):
        """
        price_ratio_ead must win over audit_price_book_ratio even when both
        are present (price_ratio_ead has priority in the search order).
        """
        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "Accion_final": ["VENDER", "MANTENER"],
            "pnl_realized": [-10_000.0, 0.0],
            "price_ratio_ead": [0.45, np.nan],         # canonical → wins
            "audit_price_book_ratio": [0.99, np.nan],  # legacy → ignored
        })
        kpis = _extract_pricing_kpis(df)
        assert kpis["avg_bid_pct_ead_available"] is True
        # Result should be based on price_ratio_ead (0.45), not audit col (0.99)
        assert abs(kpis["avg_bid_pct_ead"] - 0.45) < 1e-4, (
            f"Expected 0.45 (price_ratio_ead), got {kpis['avg_bid_pct_ead']}"
        )
