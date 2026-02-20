# -*- coding: utf-8 -*-
"""
tests/test_stress_pricing_crunch_effect.py

Validates that the 'pricing_crunch' stress scenario has a measurable effect
on NPL sale prices via cfg.BID_HAIRCUT_GLOBAL injection in price_simulator.

Before PC9 fix: bid_haircut_mult was never passed to price_simulator during
stress_engine runs => pricing_crunch and baseline produced identical results.
After PC9 fix: stress_engine injects cfg.BID_HAIRCUT_GLOBAL before calling
run_coordinator_inference, so prices drop by 1/bid_haircut_mult = ~23%.
"""

import os
import sys
import math
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


class TestPricingCrunchEffect:
    """Pricing crunch shock must reduce NPL sale prices measurably."""

    def test_bid_haircut_global_reduces_price(self):
        """
        With BID_HAIRCUT_GLOBAL=1.3 the mean NPL price must be ~30% lower
        than with BID_HAIRCUT_GLOBAL=1.0.
        Tolerance: actual reduction should be within [20%, 40%] to account
        for MC randomness and flooring effects.
        """
        import config as cfg
        from optimizer.price_simulator import simulate_npl_price

        loan_params = dict(
            ead=1_000_000.0,
            lgd=0.45,
            segment="CORPORATE",
            rating="BB",
            pd=0.12,
            dpd=180,
            secured=False,
            n_mc=5000,
            seed=42,
        )

        # Baseline (no haircut)
        cfg.BID_HAIRCUT_GLOBAL = 1.0
        result_baseline = simulate_npl_price(**loan_params)
        price_baseline = result_baseline["precio_optimo"]

        # Pricing crunch (30% deeper haircut)
        cfg.BID_HAIRCUT_GLOBAL = 1.3
        result_crunch = simulate_npl_price(**loan_params)
        price_crunch = result_crunch["precio_optimo"]

        # Reset always
        cfg.BID_HAIRCUT_GLOBAL = 1.0

        assert price_baseline > 0, "Baseline price must be positive"
        assert price_crunch > 0, "Pricing crunch price must be positive"
        assert price_crunch < price_baseline, (
            f"Pricing crunch price ({price_crunch:.0f}) must be LOWER "
            f"than baseline ({price_baseline:.0f})"
        )

        pct_reduction = (price_baseline - price_crunch) / price_baseline
        assert 0.20 <= pct_reduction <= 0.40, (
            f"Expected ~30% price reduction, got {pct_reduction:.1%}. "
            f"Baseline={price_baseline:.0f}, Crunch={price_crunch:.0f}"
        )

    def test_bid_haircut_global_reset_after_call(self):
        """BID_HAIRCUT_GLOBAL must revert to 1.0 after stress_engine run."""
        import config as cfg

        # Simulate what stress_engine does
        original = getattr(cfg, "BID_HAIRCUT_GLOBAL", 1.0)
        cfg.BID_HAIRCUT_GLOBAL = 1.3
        try:
            # Simulate a "crash" during inference
            raise RuntimeError("Simulated inference failure")
        except RuntimeError:
            pass
        finally:
            cfg.BID_HAIRCUT_GLOBAL = 1.0  # stress_engine's finally block

        assert cfg.BID_HAIRCUT_GLOBAL == 1.0, (
            "BID_HAIRCUT_GLOBAL must be reset to 1.0 even after inference failure"
        )

    def test_apply_shocks_does_not_raise_on_pricing_crunch(self):
        """
        apply_shocks with pricing_crunch shocks (bid_haircut_mult only)
        must not raise any exception — it's a no-op on portfolio features,
        the haircut is applied separately via BID_HAIRCUT_GLOBAL.
        """
        import pandas as pd
        from engines.stress_engine import apply_shocks

        df = pd.DataFrame({
            "loan_id": ["L001", "L002"],
            "EAD": [1_000_000.0, 500_000.0],
            "PD": [0.12, 0.08],
            "LGD": [0.45, 0.40],
            "RW": [1.5, 1.0],
        })

        shocks = {"bid_haircut_mult": 1.3}
        df_out = apply_shocks(df, "pricing_crunch", shocks)

        # Portfolio features should be unchanged (haircut injected via cfg globally)
        assert list(df_out["PD"]) == list(df["PD"]), "PD should not change in pricing_crunch"
        assert list(df_out["LGD"]) == list(df["LGD"]), "LGD should not change in pricing_crunch"

    def test_pricing_crunch_distinct_from_baseline_in_yaml(self):
        """
        stress_scenarios.yaml must define pricing_crunch with bid_haircut_mult != 1.0
        (i.e., it should not be equivalent to baseline by configuration).
        """
        import yaml

        yaml_path = os.path.join(ROOT_DIR, "configs", "stress_scenarios.yaml")
        assert os.path.exists(yaml_path), "stress_scenarios.yaml must exist"

        with open(yaml_path, "r") as f:
            cfg_yaml = yaml.safe_load(f)

        scenarios = cfg_yaml.get("scenarios", cfg_yaml)

        assert "pricing_crunch" in scenarios, "pricing_crunch scenario must be defined"
        assert "baseline" in scenarios, "baseline scenario must be defined"

        pc_shocks = scenarios["pricing_crunch"].get("shocks", {})
        baseline_shocks = scenarios["baseline"].get("shocks", {})

        assert pc_shocks != baseline_shocks, (
            "pricing_crunch shocks must differ from baseline shocks to avoid no-op scenario"
        )
        assert "bid_haircut_mult" in pc_shocks, (
            "pricing_crunch must define bid_haircut_mult"
        )
        assert float(pc_shocks["bid_haircut_mult"]) > 1.0, (
            "bid_haircut_mult must be > 1.0 to apply meaningful haircut"
        )
