# -*- coding: utf-8 -*-
# ============================================================
# tests/conftest.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Fixtures pytest comunes (DataFrame mínimo) compartidas por el banco de pruebas.
# ============================================================
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def df():
    """Returns a minimal dataframe for testing guardrails and policies."""
    # Mimic the structure expected by policy_inference and coordinator
    data = {
        "id_prestamo": ["L001", "L002", "L003"],
        "EAD": [10000.0, 50000.0, 75000.0],
        "PD": [0.02, 0.05, 0.12],
        "LGD": [0.45, 0.45, 0.60],
        "RW": [0.50, 0.75, 1.10],
        "EVA": [500.0, -2000.0, 100.0],
        "segmento_id": ["Retail", "Retail", "Corporate"],
        "rating_num": [3, 6, 8],
        "DPD": [0, 15, 60],
        "secured": [1, 1, 0],
        "Maturity": [2.0, 5.0, 1.0],
        "Sector": ["Services", "Real Estate", "Manufacturing"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def posture():
    """Returns a default risk posture."""
    return "balanceado"

@pytest.fixture
def postura(posture):
    """Alias for posture (for Spanish tests)."""
    return posture

@pytest.fixture
def summaries(df, posture):
    """
    Returns a dict of dataframes keyed by posture name, mimicking multi-posture run.
    """
    # Create fake variations
    df_pru = df.copy()
    df_pru["run_id"] = f"run_pru_{posture}"
    df_bal = df.copy()
    df_bal["run_id"] = f"run_bal_{posture}"
    df_des = df.copy()
    df_des["run_id"] = f"run_des_{posture}"
    
    return {
        "prudencial": df_pru,
        "balanceado": df_bal,
        "desinversion": df_des
    }
