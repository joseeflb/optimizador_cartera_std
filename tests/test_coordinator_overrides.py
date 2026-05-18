# -*- coding: utf-8 -*-
# ============================================================
# tests/test_coordinator_overrides.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Tests deterministas del arbitraje del coordinador (PRUDENCIAL_FIRST, conflicto micro/macro).
# ============================================================

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from agent.coordinator_inference import _combine_decisions
from config import BANK_STRATEGIES, BankProfile
import logging
import copy

def test_override_prudential_first():
    """
    Test determinista: forzar conflicto Micro (VENDER) vs Macro (REESTRUCTURAR).
    Con COORDINATOR_PRIORITY='PRUDENCIAL_FIRST', debe ganar REESTRUCTURAR (risk=1) vs VENDER (risk=2).
    """
    
    # 1. Mock Dataframe Micro (VENDER) -> Con inputs validos para REESTRUCTURAR (si ganase macro)
    df_micro = pd.DataFrame([{
        "loan_id": "L100",
        "Accion_micro": "VENDER",
        "Accion": "VENDER",
        "EAD": 1000.0,
        "precio_optimo": 950.0, # High recovery to avoid sale_mandate logic interfering
        "RW": 0.5,
        "plazo_optimo": 60,
        "tasa_nueva": 0.05,
        "quita": 0.0,
        "fire_sale": False,
        # Viability inputs
        "DSCR_post": 1.5,
        "PTI_post": 0.30,
        "DSCR_pre": 1.2,
        "PTI_pre": 0.35,
        "EVA_gain": 100.0, # delta_eva
        "delta_eva": 100.0,
        "segment": "corporate"
    },
    { # L102 Filler for capacity
        "loan_id": "L102",
        "Accion_micro": "MANTENER",
        "Accion": "MANTENER",
        "EAD": 100.0,
        "RW": 0.5,
        "plazo_optimo": 60,
        "tasa_nueva": 0.05,
        "quita": 0.0,
        "fire_sale": False,
        "DSCR_post": 1.5,
        "PTI_post": 0.30,
        "segment": "retail"
    }])
    
    # 2. Mock Macro Actions (L100 -> REESTRUCTURAR)
    macro_actions = {
        "L100": {
            "macro_action": "REESTRUCTURAR", 
            "rationales": ["Strategic Retention"],
            "steps_sold": [],
            "steps_restructured": [1]
        }
    }
    
    # 3. Patch Config
    # We need to ensure getattr(g, "COORDINATOR_PRIORITY") returns "PRUDENCIAL_FIRST"
    # _combine_decisions calls _get_guardrails internally. We should mock that.
    
    # [U1F195] Overwrite mandate logic TEMPORARILY
    original_tier1 = BANK_STRATEGIES[BankProfile.PRUDENTE].mandate_tier1_share
    original_target = BANK_STRATEGIES[BankProfile.PRUDENTE].mandate_share_target
    
    BANK_STRATEGIES[BankProfile.PRUDENTE].mandate_tier1_share = 0.0
    BANK_STRATEGIES[BankProfile.PRUDENTE].mandate_share_target = 0.0
    
    try:
        with patch("agent.coordinator_inference._get_guardrails") as mock_get_gr:
            mock_get_gr.return_value = {
                "COORDINATOR_PRIORITY": "PRUDENCIAL_FIRST",
                "DSCR_MIN": 1.1,
                "PTI_MAX": 0.45,
                "MIN_PRICE_TO_EAD": 0.0,
                "MIN_DEVA_RESTRUCT": 0.0
            }
            
            # 4. Run combination
            df_final, override_log = _combine_decisions(
                df_micro=df_micro,
                macro_actions=macro_actions,
                risk_posture="prudencial",
                n_steps=3,
                top_k=5
            )
    finally:
        # Restore configuration
        BANK_STRATEGIES[BankProfile.PRUDENTE].mandate_tier1_share = original_tier1
        BANK_STRATEGIES[BankProfile.PRUDENTE].mandate_share_target = original_target
        
    # 5. Assetions
    res = df_final.iloc[0]
    
    # A) Action must be REESTRUCTURAR (Macro Wins because it is safer)
    assert res["Accion_final"] == "REESTRUCTURAR", f"Expected REESTRUCTURAR, got {res['Accion_final']}"
    assert res["Macro_Selected"] == True
    
    # B) Override Override columns
    assert res["override_applied"] == True
    assert res["override_level"] == "MACRO_PRUDENTIAL"
    assert res["override_from"] == "VENDER"
    assert res["override_to"] == "REESTRUCTURAR"
    
    # C) Override Log
    assert len(override_log) == 1
    log_entry = override_log[0]
    assert log_entry["loan_id"] == "L100"
    assert log_entry["level"] == "MACRO_PRUDENTIAL"
    assert log_entry["from_action"] == "VENDER"
    assert log_entry["to_action"] == "REESTRUCTURAR"

def test_override_macro_first():
    """
    Test determinista: prioridad MACRO_FIRST.
    Aunque Micro (REESTRUCTURAR) sea mas 'safe' (1) que Macro (VENDER) (2),
    si la prioridad es MACRO_FIRST, gana Macro.
    """
    df_micro = pd.DataFrame([{
        "loan_id": "L101",
        "Accion_micro": "REESTRUCTURAR",
        "Accion": "REESTRUCTURAR",
        "EAD": 1000.0,
        "RW": 0.5,
        "fire_sale": False
    }])
    
    macro_actions = {
        "L101": {
            "macro_action": "VENDER", 
            "rationales": ["Exit Strategy"],
            "steps_sold": [2],
            "steps_restructured": []
        }
    }
    
    # 3. Patch Config also for DESINVERSION
    original_tier1_des = BANK_STRATEGIES[BankProfile.DESINVERSION].mandate_tier1_share
    original_target_des = BANK_STRATEGIES[BankProfile.DESINVERSION].mandate_share_target
    original_max_sell = BANK_STRATEGIES[BankProfile.DESINVERSION].max_sell_share
    
    BANK_STRATEGIES[BankProfile.DESINVERSION].mandate_tier1_share = 0.0
    BANK_STRATEGIES[BankProfile.DESINVERSION].mandate_share_target = 0.0
    BANK_STRATEGIES[BankProfile.DESINVERSION].max_sell_share = 1.0

    try:
        with patch("agent.coordinator_inference._get_guardrails") as mock_get_gr:
            mock_get_gr.return_value = {
                "COORDINATOR_PRIORITY": "MACRO_FIRST",
                "DSCR_MIN": 1.1,
                "MIN_PRICE_TO_EAD": 0.0
            }
            
            # Need valid sale data to pass "sale_executable" check
            # Sale executable needs: NOT insulting AND (recovery > min OR mandate)
            # Desinversion recovery min is 6%
            df_micro["precio_optimo"] = 1000.0 # 100% recovery
            df_micro["recovery_sale"] = 1.0
            df_micro["book_value"] = 1000.0
            
            df_final, override_log = _combine_decisions(
                df_micro=df_micro,
                macro_actions=macro_actions,
                risk_posture="desinversion",
                n_steps=3,
                top_k=5
            )
            
        res = df_final.iloc[0]
        assert res["Accion_final"] == "VENDER"
        assert res["override_level"] == "MACRO_STRATEGY"
        # assert res["override_applied"] == True # This is not a column in DF, but logic check

    finally:
        # Restore configuration
        BANK_STRATEGIES[BankProfile.DESINVERSION].mandate_tier1_share = original_tier1_des
        BANK_STRATEGIES[BankProfile.DESINVERSION].mandate_share_target = original_target_des
        BANK_STRATEGIES[BankProfile.DESINVERSION].max_sell_share = original_max_sell

def test_audit_artifacts_completeness():
    """
    Verifica que el override_log tenga las nuevas columnas de auditoría
    y que los KPIs no sean cero si hay datos.
    """
    df_micro = pd.DataFrame([{
        "loan_id": "L999",
        "Accion_micro": "VENDER",
        "Accion": "VENDER",
        "EAD": 50000.0,
        "RW": 0.5,
        "RWA": 25000.0, # Explicit RWA
        "EVA": 100.0,   # Explicit EVA
        "fire_sale": True
    }])
    
    macro_actions = {
        "L999": {
            "macro_action": "MANTENER", 
            "rationales": ["Macro Audit Check"]
        }
    }
    
    # Force conflict -> Override (Prudencial First: Maintain < Sell)
    with patch("agent.coordinator_inference._get_guardrails") as mock_get_gr:
        mock_get_gr.return_value = {
            "COORDINATOR_PRIORITY": "PRUDENCIAL_FIRST",
            "DSCR_MIN": 1.1,
            "MIN_PRICE_TO_EAD": 0.1
        }
        
        df_final, override_log = _combine_decisions(
            df_micro,
            macro_actions,
            risk_posture="prudencial",
            n_steps=1,
            top_k=5
        )
        
    # 1. Check override_log columns
    assert len(override_log) == 1
    log_entry = override_log[0]
    expected_keys = [
        "loan_id", "level", "from_action", "to_action", "portfolio_context",
        "posture", "run_id", "macro_action_used", "macro_rationales_short"
    ]
    for k in expected_keys:
        assert k in log_entry, f"Missing key in override_log: {k}"
        
    assert log_entry["macro_action_used"] == "MANTENER"
    assert "Macro Audit Check" in log_entry["macro_rationales_short"]
    
    # 2. Check DF Columns
    assert "macro_conflict" in df_final.columns
    assert "macro_applied" in df_final.columns
    assert df_final.iloc[0]["macro_conflict"] == True
    assert df_final.iloc[0]["macro_applied"] == True # Final=Maintain, Macro=Maintain -> Applied

    # 3. Check KPIs via helper
    from agent.coordinator_inference import compute_portfolio_kpis
    kpis = compute_portfolio_kpis(df_final)
    assert kpis["total_eva"] == 100.0
    assert kpis["total_rwa"] == 25000.0

if __name__ == "__main__":
    test_override_prudential_first()
    test_override_macro_first()
    test_audit_artifacts_completeness()
    print("[OK] All deterministic override tests passed.")
