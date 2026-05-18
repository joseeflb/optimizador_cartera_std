# -*- coding: utf-8 -*-
# ============================================================
# tests/test_postures_differentiation.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Valida la diferenciación cuantitativa entre posturas (monotonías esperadas).
# ============================================================
import pandas as pd
import pytest
import os
import glob

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

def test_postures_differentiation():
    """
    Validates that the 3 postures (Prudencial, Balanceado, Desinversion) 
    produce differentiated results following logical monotonies.
    """
    # 1. Locate the comparison CSV
    # We look for compare_postures_pc5_postures_validation.csv specifically
    target_file = os.path.join(REPORTS_DIR, "compare_postures_pc5_postures_validation.csv")
    
    if not os.path.exists(target_file):
        # Fallback: try to find any compare_postures_*.csv sorted by time
        candidates = glob.glob(os.path.join(REPORTS_DIR, "compare_postures_*.csv"))
        if not candidates:
            pytest.skip("No compare_postures_*.csv found in reports. Run comparison script first.")
        # Sort by modification time
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        target_file = candidates[0]
        print(f"Using most recent comparison file: {target_file}")

    print(f"Loading comparison data from: {target_file}")
    df = pd.read_csv(target_file)
    
    # 2. Extract values per posture
    def get_val(p, col):
        row = df[df['posture'] == p]
        if row.empty:
            return None
        return row.iloc[0][col]

    s_pru = get_val('prudencial', 'final_action_sell')
    s_bal = get_val('balanceado', 'final_action_sell')
    s_des = get_val('desinversion', 'final_action_sell')
    
    c_pru = get_val('prudencial', 'final_capital_release')
    c_bal = get_val('balanceado', 'final_capital_release')
    c_des = get_val('desinversion', 'final_capital_release')

    r_pru = get_val('prudencial', 'final_total_rwa')
    r_bal = get_val('balanceado', 'final_total_rwa')
    r_des = get_val('desinversion', 'final_total_rwa')

    # Ensure we have data
    assert s_pru is not None, "Missing data for prudencial"
    assert s_bal is not None, "Missing data for balanceado"
    assert s_des is not None, "Missing data for desinversion"

    # 3. Apply Monotonic Rules
    
    # Rule 1: Sell Count (P <= B <= D)
    # Explanation: Divestment posture should sell more or equal than balanced, which is >= prudential
    assert s_pru <= s_bal, f"Sell Count P->B violation: {s_pru} > {s_bal}"
    assert s_bal <= s_des, f"Sell Count B->D violation: {s_bal} > {s_des}"
    
    # Rule 2: Capital Release (P <= B <= D)
    # Explanation: Divestment releases more capital
    assert c_pru <= c_bal, f"Cap Release P->B violation: {c_pru} > {c_bal}"
    assert c_bal <= c_des, f"Cap Release B->D violation: {c_bal} > {c_des}"
    
    # Rule 3: RWA Final (D <= B <= P)
    # Explanation: Divestment reduces RWA the most (lowest RWA final)
    assert r_des <= r_bal, f"RWA Final D->B violation: {r_des} > {r_bal}"
    assert r_bal <= r_pru, f"RWA Final B->P violation: {r_bal} > {r_pru}"

    print("\nAll monotonic differentiation checks PASSED.")

if __name__ == "__main__":
    test_postures_differentiation()
