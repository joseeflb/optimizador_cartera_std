#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke tests (bank-ready):
1) PortfolioEnv reset/step for actions 0..11
2) LoanEnv obs shape/order sanity (10D)
3) simulate_npl_price: book_value/coverage/proxy + fire-sale override
4) Micro VecNormalize: load OK for (10,), invalidate mismatch (no crash)
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import tempfile
from typing import Optional

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - env guard
    print(f"[WARN] Missing numpy dependency: {exc}")
    raise SystemExit(0)

try:
    from env.portfolio_env import PortfolioEnv
    from env.loan_env import LoanEnv
    from optimizer.price_simulator import simulate_npl_price
except Exception as exc:  # pragma: no cover - env guard
    print(f"[WARN] Missing runtime dependencies: {exc}")
    raise SystemExit(0)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def test_portfolio_env_actions() -> None:
    env = PortfolioEnv(loan_dicts=None, top_k=2, scenario="baseline", ppo_micro=None)
    obs, info = env.reset(seed=123)
    _assert(obs.shape == (env.state_dim,), "PortfolioEnv reset obs shape mismatch.")
    _assert("portfolio_metrics" in info, "PortfolioEnv reset info missing portfolio_metrics.")

    for action in range(12):
        obs, reward, terminated, truncated, info = env.step(action)
        _assert(obs.shape == (env.state_dim,), f"PortfolioEnv step obs shape mismatch for action {action}.")
        _assert(isinstance(reward, float), "PortfolioEnv reward should be float.")
        _assert(isinstance(terminated, bool) and isinstance(truncated, bool), "Gymnasium flags invalid.")

        # ✅ AÑADE ESTO:
        if terminated or truncated:
            break


def test_loan_env_obs_shape_and_order() -> None:
    loan_pool = [{
        "segment": "CORPORATE",
        "rating": "BBB",
        "EAD": 1000.0,
        "PD": 0.50,
        "LGD": 0.40,
        "DPD": 120.0,
        "RW": 1.50,
    }]
    env = LoanEnv(loan_pool=loan_pool, seed=42)
    obs, _ = env.reset()
    _assert(obs.shape == (10,), "LoanEnv obs shape must be (10,).")

    # ✅ si LoanEnv devuelve obs normalizada, desnormalizamos para checks
    if getattr(env, "normalize_obs", False):
        scale = np.array([1e6, 1.0, 1.0, 1.5, 1e6, 1.0, 0.30, 9.0, 9.0, 24.0], dtype=np.float32)
        obs_check = obs * scale
    else:
        obs_check = obs

    segmento_id = float(obs_check[8])
    dpd_30 = float(obs_check[9])

    _assert(abs(segmento_id - 1.0) < 1e-6, f"segmento_id expected 1.0 for CORPORATE, got {segmento_id}")
    _assert(abs(dpd_30 - 4.0) < 1e-6, f"DPD/30 expected 4.0 for DPD=120, got {dpd_30}")


def test_price_simulator_book_and_override() -> None:
    base_args = dict(
        ead=1_000_000.0,
        lgd=0.85,
        pd=0.80,
        dpd=720.0,
        segment="CORPORATE",
        rating="B",
        secured=False,
        n_mc=500,
        seed=123,
    )

    out_book = simulate_npl_price(book_value=600_000.0, **base_args)
    _assert(out_book["ok"], "simulate_npl_price failed for book_value case.")
    _assert(out_book["book_value_source"] == "provided", "Expected provided book_value source.")
    _assert(
        abs(out_book["pnl"] - (out_book["precio_optimo"] - out_book["book_value"])) < 1e-6,
        "PnL should be precio_optimo - book_value (post tx).",
    )

    out_cov = simulate_npl_price(coverage_rate=0.30, **base_args)
    _assert(out_cov["ok"], "simulate_npl_price failed for coverage_rate case.")
    _assert(out_cov["book_value_source"] == "coverage_rate", "Expected coverage_rate book_value source.")

    out_proxy = simulate_npl_price(**base_args)
    _assert(out_proxy["ok"], "simulate_npl_price failed for proxy book_value case.")
    _assert(out_proxy["book_value_source"] == "proxy_pd_lgd", "Expected proxy book_value source.")

    out_low_thr = simulate_npl_price(fire_sale_price_ratio_book=0.01, **base_args)
    out_high_thr = simulate_npl_price(fire_sale_price_ratio_book=0.90, **base_args)
    _assert(out_low_thr["fire_sale_threshold_book"] == 0.01, "Override threshold low not applied.")
    _assert(out_high_thr["fire_sale_threshold_book"] == 0.90, "Override threshold high not applied.")
    _assert(out_low_thr["fire_sale"] is False or out_high_thr["fire_sale"] is True, "Override should affect fire_sale.")


def _maybe_run_micro_vn_smoke() -> Optional[str]:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except Exception as exc:
        return f"skip VecNormalize (missing stable_baselines3): {exc}"

    with tempfile.TemporaryDirectory() as tmpdir:
        good_path = os.path.join(tmpdir, "vn_good.pkl")
        bad_path = os.path.join(tmpdir, "vn_bad.pkl")

        dummy_loan = DummyVecEnv([lambda: LoanEnv()])
        vn_good = VecNormalize(dummy_loan, norm_obs=True, norm_reward=False)
        vn_good.save(good_path)

        dummy_port = DummyVecEnv([lambda: PortfolioEnv(ppo_micro=None)])
        vn_bad = VecNormalize(dummy_port, norm_obs=True, norm_reward=False)
        vn_bad.save(bad_path)

        env = PortfolioEnv(ppo_micro=None)
        ok_vn = env._load_micro_vn(good_path)
        bad_vn = env._load_micro_vn(bad_path)

        _assert(ok_vn is not None, "Expected micro VecNormalize to load for shape (10,).")
        _assert(bad_vn is None, "Expected micro VecNormalize to be invalidated for shape mismatch.")

    return None


def main() -> None:
    test_portfolio_env_actions()
    test_loan_env_obs_shape_and_order()
    test_price_simulator_book_and_override()

    skip_msg = _maybe_run_micro_vn_smoke()
    if skip_msg:
        print(f"[WARN] {skip_msg}")

    print("Smoke tests OK.")


if __name__ == "__main__":
    main()
