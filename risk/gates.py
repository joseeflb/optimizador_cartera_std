# -*- coding: utf-8 -*-
# ============================================================
# risk/gates.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Funciones puras y deterministas de gates: viabilidad de reestructuración (DSCR) y fire-sale de venta.
# ============================================================
"""Centralized gate checks (pure, deterministic)."""
from __future__ import annotations

from typing import Optional, Tuple
import math


def _safe_float(x: Optional[float]) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def check_restruct_viability(
    current_income: Optional[float],
    new_payment: Optional[float],
    dscr_min: float,
) -> Tuple[bool, float, str]:
    """Check DSCR viability for restructure.

    Returns: (allowed, dscr_post, reason)
    """
    income = _safe_float(current_income)
    payment = _safe_float(new_payment)
    dscr_thr = _safe_float(dscr_min)

    if not math.isfinite(income) or not math.isfinite(payment) or payment <= 0:
        return False, float("nan"), "DSCR_INPUT_MISSING"

    dscr_post = float(income / max(payment, 1e-12))
    if not math.isfinite(dscr_thr):
        return False, dscr_post, "DSCR_THRESHOLD_MISSING"

    if dscr_post < dscr_thr:
        return False, dscr_post, "DSCR_BELOW_MIN"

    return True, dscr_post, "DSCR_OK"


def check_sell_fire_sale(
    price_neto: Optional[float],
    book_value: Optional[float],
    allow_fire_sale: bool,
    thr_book: float,
) -> Tuple[bool, float, bool, str]:
    """Check fire-sale guardrail vs book.

    Returns: (allowed, price_book_ratio, fire_sale, reason)
    """
    price = _safe_float(price_neto)
    book = _safe_float(book_value)
    thr = _safe_float(thr_book)

    if not math.isfinite(price) or not math.isfinite(book) or book <= 0 or not math.isfinite(thr):
        return False, float("nan"), False, "FIRE_SALE_INPUT_MISSING"

    ratio = float(price / max(book, 1e-12))
    fire_sale = bool(ratio < thr)
    if fire_sale and not allow_fire_sale:
        return False, ratio, True, "FIRE_SALE_BLOCKED"

    if fire_sale and allow_fire_sale:
        return True, ratio, True, "FIRE_SALE_ALLOWED"

    return True, ratio, False, "NO_FIRE_SALE"


def check_sell_capital(
    pl_impact: Optional[float],
    rwa_before: Optional[float],
    rwa_after: Optional[float],
    min_capital_release: float = 0.0,
) -> Tuple[bool, float, str]:
    """Check capital release gate (placeholder for phase 2)."""
    rwa_b = _safe_float(rwa_before)
    rwa_a = _safe_float(rwa_after)
    min_rel = _safe_float(min_capital_release)

    if not math.isfinite(rwa_b) or not math.isfinite(rwa_a):
        return False, 0.0, "CAPITAL_INPUT_MISSING"

    capital_release = float(max(0.0, rwa_b - rwa_a))
    if math.isfinite(min_rel) and capital_release < min_rel:
        return False, capital_release, "CAPITAL_RELEASE_BELOW_MIN"

    return True, capital_release, "CAPITAL_RELEASE_OK"
