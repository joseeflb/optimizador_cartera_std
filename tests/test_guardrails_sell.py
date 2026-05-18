# -*- coding: utf-8 -*-
# ============================================================
# tests/test_guardrails_sell.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Tests unitarios del guardrail de venta (fire-sale threshold, capital, PnL).
# ============================================================
import pytest
from optimizer.guardrails import check_sell_constraints

class MockConfig:
    def __init__(self, thr=0.20, allow=False):
        self.fire_sale = type("FS", (), {"threshold_book": thr, "allow_fire_sale": allow})
        self.capital_ratio = 0.08
    def __repr__(self): return "MockConfig"

def test_sell_success():
    state = {"book_value": 100, "ead": 100, "rw": 1.5}
    # Mocking price_simulator.py output structure
    pricing = {
        "precio_optimo": 70,  # 70% recovery -> 30% loss (OK < 40%)
        "pnl": -30, 
        "capital_liberado": 12.0, # 100*1.5*0.08
        "coste_tx": 0, 
        "rw": 1.5,
        "book_value": 100
    } 
    
    cfg = MockConfig(thr=0.20, allow=False)
    ok, reasons, metrics = check_sell_constraints(state, pricing, cfg)
    
    # Debug if fails
    if not ok:
        print(f"DEBUG FAIL REASONS: {reasons}")
        
    assert ok is True
    # metrics now have different keys based on guardrails.py
    assert metrics["sell_ok"] is True
    assert metrics["bid_price"] == 70

def test_sell_fail_fire_sale_blocked():
    state = {"book_value": 100, "ead": 100}
    # Mock pricing that triggers fire sale
    pricing = {
        "precio_optimo": 10, 
        "pnl": -90, 
        "capital_liberado": 0,
        "coste_tx": 0,
        "fire_sale": True,
        "fire_sale_reason": "Low Price vs Book"
    } 
    
    cfg = MockConfig(thr=0.20, allow=False) # prudential
    ok, reasons, metrics = check_sell_constraints(state, pricing, cfg)
    assert ok is False
    assert any("FIRE_SALE_BLOCK" in r for r in reasons)

def test_sell_fail_price_negative():
    state = {"book_value": 100, "ead": 100}
    pricing = {
        "precio_optimo": -5, 
        "pnl": -105, 
        "capital_liberado": 0,
        "coste_tx": 0
    }
    
    cfg = MockConfig()
    ok, reasons, metrics = check_sell_constraints(state, pricing, cfg)
    assert ok is False
    # Guardrail 4: BID_TOO_LOW usually catches negative or very low prices
    # Guardrail 2: MISSING_PRICING_OUTPUTS is not triggered as we provide keys over
    assert any("BID_TOO_LOW" in r for r in reasons)

if __name__ == '__main__':
    try:
        test_sell_success()
        test_sell_fail_fire_sale_blocked()
        test_sell_fail_price_negative()
        print(' test_guardrails_sell.py PASSED')
    except AssertionError as e:
        print(f' FAIL: {e}')
        import traceback
        traceback.print_exc()

