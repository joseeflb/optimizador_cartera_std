"""
ANÁLISIS DE PERCENTILES - Calibración de thresholds de mandato
================================================================
Analiza distribución de RWA, recovery, age_npl para definir
thresholds que capturen ~20-30% casos más críticos (mandato real)
"""
import pandas as pd
import numpy as np

# Cargar portfolio
df = pd.read_excel("data/portfolio_synth.xlsx")

print("="*80)
print("ANÁLISIS DE PERCENTILES - CALIBRACIÓN MANDATOS BANK-GRADE")
print("="*80)

# =========================================================================
# 1. RWA DISTRIBUTION
# =========================================================================
print("\n[1] RWA (Risk-Weighted Assets) DISTRIBUTION")
print("-"*80)

rw = df["RW"].replace([np.inf, -np.inf], np.nan).dropna()
print(f"  N válidos: {len(rw)}/500")
print(f"\n  Percentiles:")
for p in [10, 20, 25, 30, 50, 70, 75, 80, 90, 95, 99]:
    val = rw.quantile(p/100)
    count_above = (rw >= val).sum()
    pct_above = (count_above / len(rw)) * 100
    print(f"    p{p:2d}: {val:6.2f} | {count_above:3d} loans >= ({pct_above:5.1f}%)")

print(f"\n  Estadísticas:")
print(f"    Media:  {rw.mean():.2f}")
print(f"    Median: {rw.median():.2f}")
print(f"    Std:    {rw.std():.2f}")

# Thresholds actuales
print(f"\n  Thresholds actuales:")
print(f"    PRUDENCIAL:   RWA >= 2.50 -> {(rw >= 2.50).sum()} loans ({(rw >= 2.50).sum()/len(rw)*100:.1f}%)")
print(f"    BALANCEADO:   RWA >= 2.00 -> {(rw >= 2.00).sum()} loans ({(rw >= 2.00).sum()/len(rw)*100:.1f}%)")
print(f"    DESINVERSION: RWA >= 1.50 -> {(rw >= 1.50).sum()} loans ({(rw >= 1.50).sum()/len(rw)*100:.1f}%)")

print(f"\n  ✅ PROPUESTA BANK-GRADE (target 20-30% más críticos):")
print(f"    PRUDENCIAL:   RWA >= {rw.quantile(0.90):.2f} (p90) -> ~10% mandatos")
print(f"    BALANCEADO:   RWA >= {rw.quantile(0.80):.2f} (p80) -> ~20% mandatos")
print(f"    DESINVERSION: RWA >= {rw.quantile(0.70):.2f} (p70) -> ~30% mandatos")

# =========================================================================
# 2. RECOVERY DISTRIBUTION (sale)
# =========================================================================
print("\n[2] RECOVERY_SALE DISTRIBUTION")
print("-"*80)

# Load inference to get recovery_sale
try:
    inf_file = "reports/coordinated_inference_NPL_BANKGRADE_des_20260216_000319_desinversion/decisiones_finales_desinversion.xlsx"
    df_inf = pd.read_excel(inf_file)
    
    if "recovery_sale" in df_inf.columns:
        rec = df_inf["recovery_sale"].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        # Calcular recovery_rate = sale_price / EAD
        rec = (df_inf["precio_optimo"] / df_inf["EAD"]).replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"  N válidos: {len(rec)}/500")
    print(f"\n  Percentiles:")
    for p in [1, 5, 10, 20, 25, 30, 50, 70, 75, 80]:
        val = rec.quantile(p/100)
        count_below = (rec < val).sum()
        pct_below = (count_below / len(rec)) * 100
        print(f"    p{p:2d}: {val:.4f} ({val*100:5.2f}%) | {count_below:3d} loans < ({pct_below:5.1f}%)")
    
    print(f"\n  Estadísticas:")
    print(f"    Media:  {rec.mean():.4f} ({rec.mean()*100:.2f}%)")
    print(f"    Median: {rec.median():.4f} ({rec.median()*100:.2f}%)")
    
    # Thresholds actuales
    print(f"\n  Thresholds actuales:")
    print(f"    PRUDENCIAL:   recovery < 20% -> {(rec < 0.20).sum()} loans ({(rec < 0.20).sum()/len(rec)*100:.1f}%)")
    print(f"    BALANCEADO:   recovery < 25% -> {(rec < 0.25).sum()} loans ({(rec < 0.25).sum()/len(rec)*100:.1f}%)")
    print(f"    DESINVERSION: recovery < 30% -> {(rec < 0.30).sum()} loans ({(rec < 0.30).sum()/len(rec)*100:.1f}%)")
    
    print(f"\n  ✅ PROPUESTA BANK-GRADE (target 20-30% peores recoveries):")
    print(f"    PRUDENCIAL:   recovery < {rec.quantile(0.10):.2%} (p10) -> ~10% mandatos")
    print(f"    BALANCEADO:   recovery < {rec.quantile(0.20):.2%} (p20) -> ~20% mandatos")
    print(f"    DESINVERSION: recovery < {rec.quantile(0.30):.2%} (p30) -> ~30% mandatos")
    
except Exception as e:
    print(f"  [ERROR] No se pudo cargar recovery: {e}")

# =========================================================================
# 3. AGE NPL DISTRIBUTION (o DPD como proxy)
# =========================================================================
print("\n[3] AGE_NPL / DPD DISTRIBUTION")
print("-"*80)

age_col = None
if "age_npl_m" in df.columns:
    age_col = "age_npl_m"
    age = df["age_npl_m"].replace([np.inf, -np.inf], np.nan).dropna()
elif "DPD" in df.columns:
    age_col = "DPD"
    age = df["DPD"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  [INFO] Usando DPD (days past due) como proxy de age_npl")
else:
    print(f"  [WARNING] Ni age_npl_m ni DPD encontrados en portfolio")
    age = None

if age is not None:
    print(f"  Columna: {age_col}")
    print(f"  N válidos: {len(age)}/500")
    print(f"\n  Percentiles:")
    for p in [50, 60, 70, 75, 80, 85, 90, 95, 99]:
        val = age.quantile(p/100)
        count_above = (age >= val).sum()
        pct_above = (count_above / len(age)) * 100
        print(f"    p{p:2d}: {val:6.1f} | {count_above:3d} loans >= ({pct_above:5.1f}%)")
    
    print(f"\n  Estadísticas:")
    print(f"    Media:  {age.mean():.1f}")
    print(f"    Median: {age.median():.1f}")
    
    # Thresholds actuales
    units = "m" if age_col == "age_npl_m" else "d"
    print(f"\n  Thresholds actuales:")
    if age_col == "age_npl_m":
        print(f"    PRUDENCIAL:   age >= 48m (4 años) -> {(age >= 48).sum()} loans ({(age >= 48).sum()/len(age)*100:.1f}%)")
        print(f"    BALANCEADO:   age >= 42m (3.5 años) -> {(age >= 42).sum()} loans ({(age >= 42).sum()/len(age)*100:.1f}%)")
        print(f"    DESINVERSION: age >= 36m (3 años) -> {(age >= 36).sum()} loans ({(age >= 36).sum()/len(age)*100:.1f}%)")
    else:  # DPD
        print(f"    PRUDENCIAL:   DPD >= 1460d (~4 años) -> {(age >= 1460).sum()} loans ({(age >= 1460).sum()/len(age)*100:.1f}%)")
        print(f"    BALANCEADO:   DPD >= 1260d (~3.5 años) -> {(age >= 1260).sum()} loans ({(age >= 1260).sum()/len(age)*100:.1f}%)")
        print(f"    DESINVERSION: DPD >= 1080d (~3 años) -> {(age >= 1080).sum()} loans ({(age >= 1080).sum()/len(age)*100:.1f}%)")
    
    print(f"\n  ✅ PROPUESTA BANK-GRADE (target 20-30% más antiguos/morosos):")
    print(f"    PRUDENCIAL:   {age_col} >= {age.quantile(0.90):.0f}{units} (p90) -> ~10% mandatos")
    print(f"    BALANCEADO:   {age_col} >= {age.quantile(0.80):.0f}{units} (p80) -> ~20% mandatos")
    print(f"    DESINVERSION: {age_col} >= {age.quantile(0.70):.0f}{units} (p70) -> ~30% mandatos")

# =========================================================================
# RESUMEN CALIBRACIÓN
# =========================================================================
print("\n" + "="*80)
print("RESUMEN - THRESHOLDS PROPUESTOS BANK-GRADE")
print("="*80)

print("\n✅ CALIBRACIÓN POR PERCENTILES (Defensible en comité):")
print("\nPRUDENCIAL (conservador - mandatos excepcionales ~10%):")
print(f"  - RWA >= {rw.quantile(0.90):.2f} (p90 - solo casos críticos capital)")
print(f"  - recovery < {rec.quantile(0.10):.2%} (p10 - worst recoveries)")
print(f"  - age_npl >= {age.quantile(0.90):.0f}m (p90 - más antiguos)")

print("\nBALANCEADO (equilibrado - mandatos selectivos ~20%):")
print(f"  - RWA >= {rw.quantile(0.80):.2f} (p80 - casos moderadamente críticos)")
print(f"  - recovery < {rec.quantile(0.20):.2%} (p20 - low recoveries)")
print(f"  - age_npl >= {age.quantile(0.80):.0f}m (p80 - veteranos)")

print("\nDESINVERSION (agresivo - mandatos frecuentes pero justificables ~30%):")
print(f"  - RWA >= {rw.quantile(0.70):.2f} (p70 - presión capital moderada)")
print(f"  - recovery < {rec.quantile(0.30):.2%} (p30 - below-median recoveries)")
print(f"  - age_npl >= {age.quantile(0.70):.0f}m (p70 - NPL envejecido)")

print("\n📊 LÓGICA DE MANDATO (OR de 3 criterios):")
print("  Si (RWA ≥ threshold) OR (recovery < threshold) OR (age ≥ threshold)")
print("  → sale_mandate = True (venta obligatoria)")

print("\n💡 DIFERENCIACIÓN:")
print("  PRUD: solo outliers extremos (p90) → ~10% mandatos")
print("  BAL:  presión moderada (p80) → ~20% mandatos")
print("  DESINV: presión relajada (p70) → ~30% mandatos")
print("  → Monotonicidad: mandatos(PRUD) < mandatos(BAL) < mandatos(DESINV)")

print("\n" + "="*80)
