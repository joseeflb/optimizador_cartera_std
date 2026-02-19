# tests/test_guardrails_extended.py
import pytest
from unittest.mock import MagicMock
from optimizer.guardrails import check_restructure_constraints, check_sell_constraints
import config as global_cfg

def test_restructure_config_overrides():
    """Verifica que el guardrail respeta los umbrales del objeto config pasado."""
    loan_state = {"pti_actual": 0.50, "dscr_actual": 1.05}
    
    # 1. Config Standard (deberia fallar, PTI 0.50 > 0.45 default)
    ok, reasons, metrics = check_restructure_constraints(loan_state, config_obj=None)
    assert not ok
    assert "PTI_HIGH" in str(reasons)
    assert metrics["pti_max"] == 0.45

    # 2. Config Permisiva (mock)
    mock_cfg = MagicMock()
    mock_cfg.risk_params.pti_limit = 0.60
    mock_cfg.risk_params.dscr_min = 1.00
    
    ok, reasons, metrics = check_restructure_constraints(loan_state, config_obj=mock_cfg)
    assert ok
    assert len(reasons) == 0
    assert metrics["pti_max"] == 0.60

def test_sell_constraints_missing_keys():
    """Verifica que bloquea si el pricing no esta completo."""
    loan_state = {"ead": 1000, "book_value": 800}
    pricing_out = {"price": 500} # Faltan pnl, capital_liberado
    
    ok, reasons, metrics = check_sell_constraints(loan_state, pricing_out, config_obj=None)
    assert not ok
    assert "MISSING_PRICING_OUTPUTS" in str(reasons)
    assert not metrics["sell_ok"]

def test_sell_constraints_pnl_too_negative():
    """Verifica bloqueo por perdida excesiva (Prudencial)."""
    # EAD 1000, Perdida Max 40% (400)
    # Venta con perdida de 500
    loan_state = {"ead": 1000, "book_value": 1000}
    pricing_out = {
        "precio_optimo": 500,
        "pnl": -500,
        "capital_liberado": 50,
        "coste_tx": 10,
        "fire_sale": False
    }
    
    # Config default (usa global GR_SELL_MAX_FIRE_SALE_LOSS_PCT_EAD_PRUDENCIAL = 0.40)
    ok, reasons, metrics = check_sell_constraints(loan_state, pricing_out, config_obj=None)
    assert not ok
    assert "PNL_TOO_NEGATIVE_PRUDENCIAL" in str(reasons)
    
def test_sell_constraints_bid_too_low():
    """Verifica bloqueo por oferta ridicula."""
    loan_state = {"ead": 1000, "book_value": 1000}
    pricing_out = {
        "precio_optimo": 10, # 1% EAD
        "pnl": -990,
        "capital_liberado": 0,
        "coste_tx": 0,
        "fire_sale": False
    }
    
    ok, reasons, metrics = check_sell_constraints(loan_state, pricing_out, config_obj=None)
    assert not ok
    assert "BID_TOO_LOW" in str(reasons)

def test_sell_constraints_fire_sale_logic():
    """Verifica integracion con flag de Fire Sale."""
    loan_state = {"ead": 1000, "book_value": 1000}
    pricing_out = {
        "precio_optimo": 800,
        "pnl": -200,
        "capital_liberado": 80,
        "fire_sale": True,
        "fire_sale_reason": "Low Book Ratio"
    }
    
    # Default config (no allow fire sale explícito) -> Bloquea
    ok, reasons, metrics = check_sell_constraints(loan_state, pricing_out, config_obj=None)
    assert not ok
    assert "FIRE_SALE_BLOCK" in str(reasons)

    # Config que permite fire sale
    mock_cfg = MagicMock()
    mock_cfg.fire_sale.allow_fire_sale = True
    mock_cfg.fire_sale.threshold_book = 0.20 # low bar
    
    ok, reasons, metrics = check_sell_constraints(loan_state, pricing_out, config_obj=mock_cfg)
    assert ok # Deberia pasar si se permite explicitamente
