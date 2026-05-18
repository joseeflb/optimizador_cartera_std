# -*- coding: utf-8 -*-
# ============================================================
# tests/test_guardrails_restructure.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Tests unitarios del guardrail de reestructuración (PTI, DSCR, mejora EVA).
# ============================================================

import pytest
from optimizer.guardrails import check_restructure_constraints

class MockConfig:
    def __init__(self, pti=0.40, dscr=1.10):
        self.risk_params = type("RP", (), {"pti_limit": pti, "dscr_min": dscr})

def test_restructure_success():
    # PTI ok, DSCR ok
    state = {
        "pti_actual": 0.30,
        "dscr_actual": 1.20,
        "eva_pre": 100,
        "eva_post": 150
    }
    cfg = MockConfig()
    ok, reasons, metrics = check_restructure_constraints(state, cfg)
    assert ok is True
    assert len(reasons) == 0

def test_restructure_fail_pti():
    # PTI high -> Fail
    state = {
        "pti_actual": 0.50, # > 0.40
        "dscr_actual": 1.20
    }
    cfg = MockConfig(pti=0.40)
    ok, reasons, metrics = check_restructure_constraints(state, cfg)
    assert ok is False
    assert any("PTI_HIGH" in r for r in reasons)

def test_restructure_fail_dscr():
    # DSCR low -> Fail
    state = {
        "pti_actual": 0.30,
        "dscr_actual": 0.90 # < 1.10
    }
    cfg = MockConfig(dscr=1.10)
    ok, reasons, metrics = check_restructure_constraints(state, cfg)
    assert ok is False
    assert any("DSCR_LOW" in r for r in reasons)

if __name__ == "__main__":
    try:
        test_restructure_success()
        test_restructure_fail_pti()
        test_restructure_fail_dscr()
        print("[OK] test_guardrails_restructure.py PASSED")
    except AssertionError as e:
        print(f"[ERR] FAIL: {e}")
        import traceback
        traceback.print_exc()

