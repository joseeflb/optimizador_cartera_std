#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script temporal para validar restricciones duras en decisiones
"""
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Leer último archivo de decisiones prudencial
decision_file = ROOT / "reports" / "inference_20260208_202112_coord_micro_run1_prudencial" / "decisiones_explicadas.xlsx"

print(f"Leyendo: {decision_file}")
df = pd.read_excel(decision_file)

print(f"\n{'='*80}")
print(f"ANÁLISIS DE VALIDACIÓN - RESTRICCIONES DURAS")
print(f"{'='*80}\n")

# 1. REESTRUCTURACIÓN: PTI/DSCR
restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()
print(f"=== REESTRUCTURAS ===")
print(f"Total reestructuras: {len(restruct)}")
print(f"Sin PTI_post: {restruct['PTI_post'].isna().sum()}")
print(f"Sin DSCR_post: {restruct['DSCR_post'].isna().sum()}")

if len(restruct) > 0:
    print(f"\nPTI_post stats (debe estar entre 0.30-0.50):")
    print(restruct['PTI_post'].describe())
    print(f"\nDSCR_post stats (debe ser >= 1.05 para prudencial):")
    print(restruct['DSCR_post'].describe())
    
    # Checks
    pti_low = (restruct['PTI_post'] < 0.20).sum()
    pti_high = (restruct['PTI_post'] > 0.60).sum()
    dscr_low = (restruct['DSCR_post'] < 1.00).sum()
    
    print(f"\n⚠️ PTI_post < 0.20 (inviable): {pti_low}")
    print(f"⚠️ PTI_post > 0.60 (insostenible): {pti_high}")
    print(f"⚠️ DSCR_post < 1.00 (inviable): {dscr_low}")
    
    # Muestra
    print("\nMuestra de reestructuras (primeras 5):")
    cols = ['loan_id', 'PTI_pre', 'PTI_post', 'DSCR_pre', 'DSCR_post', 
            'plazo_optimo', 'tasa_nueva', 'quita', 'restruct_viable']
    print(restruct[cols].head(5).to_string(index=False))

# 2. VENTA: Fire-sale y pricing
sell = df[df['Accion_final'] == 'VENDER'].copy()
print(f"\n{'='*80}")
print(f"=== VENTAS ===")
print(f"Total ventas: {len(sell)}")

if len(sell) > 0:
    print(f"Fire-sales detectados: {sell['Fire_Sale'].sum()}")
    print(f"Blocked por fire-sale: {sell['Sell_Blocked'].sum()}")
    
    print(f"\nPrice/EAD stats:")
    print(sell['Price_to_EAD'].describe())
    
    print(f"\nP&L stats (debe ser negativo en fire-sale):")
    print(sell['pnl'].describe())
    
    # Muestra
    print("\nMuestra de ventas (primeras 5):")
    cols = ['loan_id', 'Fire_Sale', 'Sell_Blocked', 'Price_to_EAD', 'pnl', 
            'book_value', 'price_ratio_book']
    print(sell[cols].head(5).to_string(index=False))

# 3. MIX DE DECISIONES POR POSTURA
print(f"\n{'='*80}")
print(f"=== MIX DE DECISIONES ===")
mix = df['Accion_final'].value_counts(normalize=True) * 100
print(mix.to_string())

# 4. CONVERGENCIA MICRO/MACRO
print(f"\n{'='*80}")
print(f"=== CONVERGENCIA MICRO/MACRO ===")
conv = df['Convergencia_Caso'].value_counts()
print(conv.to_string())

# 5. REASON CODES
print(f"\n{'='*80}")
print(f"=== REASON CODES (TOP 10) ===")
rc = df['Reason_Code'].value_counts().head(10)
print(rc.to_string())

print(f"\n{'='*80}")
print("✅ Análisis completado")
