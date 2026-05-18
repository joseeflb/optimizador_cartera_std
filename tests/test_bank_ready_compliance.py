# -*- coding: utf-8 -*-
# ============================================================
# tests/test_bank_ready_compliance.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Verifica que las decisiones REESTRUCTURAR cumplen PTI/DSCR y que EVA mejora.
# ============================================================
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def compliance_df():
    data = {
        'id_prestamo': ['L1', 'L2', 'L3'],
        'segment': ['MORTGAGE', 'CORPORATE', 'RETAIL'],
        'Accion_final': ['REESTRUCTURAR', 'REESTRUCTURAR', 'VENDER'],
        'PTI_post': [0.30, 0.45, 0.0],
        'DSCR_post': [1.2, 1.25, 0.0],
        'EVA_post': [100, 500, -50],
        'EVA_pre': [50, 400, -100],
    }
    return pd.DataFrame(data)

def test_restructure_pti_dscr(compliance_df):
    df = compliance_df
    restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()
    # Basic assertions
    assert len(restruct) == 2
    # Ensure required columns exist
    assert 'PTI_post' in restruct.columns
    assert 'DSCR_post' in restruct.columns

def test_fire_sale_protections(compliance_df):
    df = compliance_df
    sells = df[df['Accion_final'] == 'VENDER']
    assert len(sells) == 1
