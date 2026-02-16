"""
Analizar recovery_sale (columna portfolio) vs precio_optimo/EAD 
Para entender por qué mandate recovery_low NO activó
"""

import pandas as pd
import numpy as np

# Cargar uno de los Excels finales (tiene recovery_sale y precio_optimo)
excel = r"C:\Users\EV543NB\OneDrive - EY\Desktop\optimizador_cartera_std\reports\runs\20260216_001627_run1_DELIVERABLE\decisiones_finales_desinversion.xlsx"

df = pd.read_excel(excel)

print("="*80)
print("ANÁLISIS: recovery_sale vs precio_optimo/EAD")
print("="*80)

# 1. recovery_sale (columna original del portfolio)
if "recovery_sale" in df.columns:
    rec_sale = df["recovery_sale"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n[1] RECOVERY_SALE (columna portfolio original):")
    print(f"  N válidos: {len(rec_sale)}/500")
    print(f"  Mean: {rec_sale.mean():.2%}")
    print(f"  Median: {rec_sale.median():.2%}")
    print(f"\n  Percentiles:")
    for p in [1, 5, 10, 20, 30, 50]:
        val = rec_sale.quantile(p/100)
        count_below = (rec_sale < val).sum()
        pct_below = (count_below / len(rec_sale)) * 100
        print(f"    p{p:2d}: {val:.4f} ({val*100:5.2f}%) | {count_below:3d} loans < ({pct_below:5.1f}%)")
    
    # Thresholds mandato
    print(f"\n  Thresholds mandato (recovery_sale < X):")
    print(f"    PRUDENCIAL:   < 9.5%   → {(rec_sale < 0.095).sum()} loans ({(rec_sale < 0.095).sum()/len(rec_sale)*100:.1f}%)")
    print(f"    BALANCEADO:   < 11.3%  → {(rec_sale < 0.113).sum()} loans ({(rec_sale < 0.113).sum()/len(rec_sale)*100:.1f}%)")
    print(f"    DESINVERSION: < 12.2%  → {(rec_sale < 0.122).sum()} loans ({(rec_sale < 0.122).sum()/len(rec_sale)*100:.1f}%)")
else:
    print("\n[1] RECOVERY_SALE: Columna NO encontrada")

# 2. precio_optimo / EAD (recovery rate calculado)
if "precio_optimo" in df.columns and "EAD" in df.columns:
    rec_calc = (df["precio_optimo"] / df["EAD"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\n[2] RECOVERY_CALC (precio_optimo/EAD - calculado en inferencia):")
    print(f"  N válidos: {len(rec_calc)}/500")
    print(f"  Mean: {rec_calc.mean():.2%}")
    print(f"  Median: {rec_calc.median():.2%}")
    print(f"\n  Percentiles:") 
    for p in [1, 5, 10, 20, 30, 50]:
        val = rec_calc.quantile(p/100)
        count_below = (rec_calc < val).sum()
        pct_below = (count_below / len(rec_calc)) * 100
        print(f"    p{p:2d}: {val:.4f} ({val*100:5.2f}%) | {count_below:3d} loans < ({pct_below:5.1f}%)")
    
    # Thresholds mandato
    print(f"\n  Thresholds mandato (recovery_calc < X):")
    print(f"    PRUDENCIAL:   < 9.5%   → {(rec_calc < 0.095).sum()} loans ({(rec_calc < 0.095).sum()/len(rec_calc)*100:.1f}%)")
    print(f"    BALANCEADO:   < 11.3%  → {(rec_calc < 0.113).sum()} loans ({(rec_calc < 0.113).sum()/len(rec_calc)*100:.1f}%)")
    print(f"    DESINVERSION: < 12.2%  → {(rec_calc < 0.122).sum()} loans ({(rec_calc < 0.122).sum()/len(rec_calc)*100:.1f}%)")
else:
    print("\n[2] RECOVERY_CALC: Columnas NO encontradas")

# 3. Comparación
if "recovery_sale" in df.columns and "precio_optimo" in df.columns and "EAD" in df.columns:
    rec_sale_full = df["recovery_sale"].replace([np.inf, -np.inf], np.nan).fillna(0)
    rec_calc_full = (df["precio_optimo"] / df["EAD"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    diff = rec_calc_full - rec_sale_full
    print(f"\n[3] COMPARACIÓN (recovery_calc - recovery_sale):")
    print(f"  Mean diff: {diff.mean():.4f} ({diff.mean()*100:+.2f}%)")
    print(f"  Median diff: {diff.median():.4f} ({diff.median()*100:+.2f}%)")
    print(f"  Loans where calc > sale: {(rec_calc_full > rec_sale_full).sum()}")
    print(f"  Loans where calc < sale: {(rec_calc_full < rec_sale_full).sum()}")
    print(f"  Loans where calc ~ sale: {(abs(diff) < 0.01).sum()}")

print("\n" + "="*80)
print("CONCLUSIÓN:")
print("  Si recovery_sale (pre-existente en portfolio) es significativamente")
print("  diferente de precio_optimo/EAD (calculado en inferencia), entonces")
print("  los thresholds basados en percentiles de precio_optimo/EAD NO aplican")
print("  correctamente a recovery_sale.")
print("="*80)
