import pytest
from optimizer.guardrails import check_sell_constraints

class MockConfig:
    def __init__(self, thr=0.20, allow=False):
        self.fire_sale = type("FS", (), {"threshold_book": thr, "allow_fire_sale": allow})
        self.capital_ratio = 0.08
    def __repr__(self): return "MockConfig"

def test_sell_success():
    state = {"book_value": 100, "ead": 100, "rw": 1.5}
    pricing = {"price": 50, "net_price": 50} # 50% ratio > 20%
    
    cfg = MockConfig(thr=0.20, allow=False)
    ok, reasons, metrics = check_sell_constraints(state, pricing, cfg)
    assert ok is True
    assert metrics["price_book_ratio"] == 0.50
    assert metrics["capital_release"] == 100 * 1.5 * 0.08

def test_sell_fail_fire_sale_blocked():
    state = {"book_value": 100}
    pricing = {"net_price": 10} # 10% < 20%
    
    cfg = MockConfig(thr=0.20, allow=False) # prudential
    ok, reasons, metrics = check_sell_constraints(state, pricing, cfg)
    assert ok is False
    assert any("FIRE_SALE_BLOCK" in r for r in reasons)

def test_sell_fail_price_negative():
    state = {"book_value": 100}
    pricing = {"net_price": -5}
    
    cfg = MockConfig()
    ok, reasons, metrics = check_sell_constraints(state, pricing, cfg)
    assert ok is False
    assert any("PRICE_NON_POSITIVE" in r for r in reasons)

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

