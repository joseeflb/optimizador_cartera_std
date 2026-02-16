"""
Analizar mandatos recalibrados - Opción B bank-grade
Compara mandatos ANTES vs AHORA (post-recalibración percentiles)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\EV543NB\OneDrive - EY\Desktop\optimizador_cartera_std\reports")

# Run actual (latest)
RUN_DIR = BASE / "runs" / "20260216_001627_run1_DELIVERABLE"

posturas = ["prudencial", "balanceado", "desinversion"]

print("="*80)
print("ANÁLISIS MANDATOS RECALIBRADOS - BANK-GRADE PERCENTILES")
print("="*80)

results = {}

for postura in posturas:
    excel_file = RUN_DIR / f"decisiones_finales_{postura}.xlsx"
    
    if not excel_file.exists():
        print(f"\n⚠️  {postura.upper()}: Excel no encontrado")
        continue
    
    df = pd.read_excel(excel_file)  # Cargar primera hoja (default)
    
    # 1. Mandatos
    n_mandatos = df["sale_mandate"].sum() if "sale_mandate" in df.columns else 0
    pct_mandatos = (n_mandatos / len(df)) * 100
    
    # 2. Mix acciones FINALES
    mix = df["Accion_final"].value_counts()
    n_vender = mix.get("VENDER", 0)
    n_mantener = mix.get("MANTENER", 0)
    n_reestr = mix.get("REESTRUCTURAR", 0)
    
    pct_vender = (n_vender / len(df)) * 100
    pct_mantener = (n_mantener / len(df)) * 100
    pct_reestr = (n_reestr / len(df)) * 100
    
    # 3. Execution status (mandatos ejecutados)
    if "Execution_status" in df.columns:
        status_counts = df["Execution_status"].value_counts()
        n_mandate_exec = status_counts.get("MANDATE_EXECUTED", 0)
    else:
        n_mandate_exec = 0
    
    # 4. Recovery rate de mandatos
    mandatos_df = df[df["sale_mandate"] == True]
    
    if len(mandatos_df) > 0:
        # Calcular recovery_rate = precio_optimo / EAD
        if "precio_optimo" in mandatos_df.columns and "EAD" in mandatos_df.columns:
            recovery_mandatos = (mandatos_df["precio_optimo"] / mandatos_df["EAD"]).replace([np.inf, -np.inf], np.nan)
            recovery_mean = recovery_mandatos.mean()
            recovery_median = recovery_mandatos.median()
            recovery_max = recovery_mandatos.max()
        else:
            recovery_mean = recovery_median = recovery_max = np.nan
    else:
        recovery_mean = recovery_median = recovery_max = np.nan
    
    # 5. Breakdown mandatos por razón
    if len(mandatos_df) > 0 and "sale_mandate_reason" in mandatos_df.columns:
        # Contar cuántos mandatos por cada trigger
        reasons = mandatos_df["sale_mandate_reason"].dropna()
        n_rwa = sum("RWA_HIGH" in str(r) for r in reasons)
        n_recovery = sum("RECOVERY_LOW" in str(r) for r in reasons)
        n_age = sum("AGE_NPL" in str(r) for r in reasons)
    else:
        n_rwa = n_recovery = n_age = 0
    
    results[postura] = {
        "n_loans": len(df),
        "n_mandatos": n_mandatos,
        "pct_mandatos": pct_mandatos,
        "n_vender": n_vender,
        "pct_vender": pct_vender,
        "n_mantener": n_mantener,
        "pct_mantener": pct_mantener,
        "n_reestr": n_reestr,
        "pct_reestr": pct_reestr,
        "n_mandate_exec": n_mandate_exec,
        "recovery_mean": recovery_mean,
        "recovery_median": recovery_median,
        "recovery_max": recovery_max,
        "n_rwa": n_rwa,
        "n_recovery": n_recovery,
        "n_age": n_age,
    }
    
    print(f"\n[{postura.upper()}]")
    print("-"*80)
    print(f"  Mandatos:        {n_mandatos}/{len(df)} ({pct_mandatos:.1f}%) ← TARGET: PRUD ~10%, BAL ~20%, DESINV ~30%")
    print(f"  Mandate_exec:    {n_mandate_exec}")
    print(f"\n  Mix FINAL:")
    print(f"    VENDER:        {n_vender} ({pct_vender:.1f}%)")
    print(f"    MANTENER:      {n_mantener} ({pct_mantener:.1f}%)")
    print(f"    REESTRUCTURAR: {n_reestr} ({pct_reestr:.1f}%)")
    
    print(f"\n  Recovery mandatos:")
    print(f"    Mean:   {recovery_mean:.2%}" if not np.isnan(recovery_mean) else "    Mean:   N/A")
    print(f"    Median: {recovery_median:.2%}" if not np.isnan(recovery_median) else "    Median: N/A")
    
    print(f"\n  Razones mandato (triggers):")
    print(f"    RWA_HIGH:      {n_rwa}")
    print(f"    RECOVERY_LOW:  {n_recovery}")
    print(f"    AGE_NPL:       {n_age}")

# =========================================================================
# RESUMEN COMPARATIVO
# =========================================================================
print("\n" + "="*80)
print("RESUMEN COMPARATIVO - ANTE vs HOY")
print("="*80)

print("\n[ANTES - Run 20260208_111332 (thresholds laxos)]:")
print("  PRUDENCIAL:   0 mandatos (0.0%)   | VENDER: 63.4%")
print("  BALANCEADO:   0 mandatos (0.0%)   | VENDER: 78.0%")
print("  DESINVERSION: 455 mandatos (91%)  | VENDER: 98.6%  ← PROBLEMA")

print("\n[HOY - Run 20260216_001627 (thresholds percentiles bank-grade)]:")
for postura in posturas:
    r = results[postura]
    print(f"  {postura.upper():13s}: {r['n_mandatos']:3d} mandatos ({r['pct_mandatos']:4.1f}%)  | VENDER: {r['pct_vender']:.1f}%")

print("\n✅ VALIDACIÓN:")
print("  1. Mandatos selectivos (~10%/20%/30%)?")
prud_ok = 8 <= results["prudencial"]["pct_mandatos"] <= 12
bal_ok = 18 <= results["balanceado"]["pct_mandatos"] <= 22
des_ok = 28 <= results["desinversion"]["pct_mandatos"] <= 32
print(f"     PRUDENCIAL:   {results['prudencial']['pct_mandatos']:.1f}%  {'✅' if prud_ok else '❌'} (target 8-12%)")
print(f"     BALANCEADO:   {results['balanceado']['pct_mandatos']:.1f}%  {'✅' if bal_ok else '❌'} (target 18-22%)")
print(f"     DESINVERSION: {results['desinversion']['pct_mandatos']:.1f}%  {'✅' if des_ok else '❌'} (target 28-32%)")

print("\n  2. Monotonicidad mandatos?")
mono_ok = (results["prudencial"]["n_mandatos"] < results["balanceado"]["n_mandatos"] < 
           results["desinversion"]["n_mandatos"])
print(f"     PRUD < BAL < DESINV: {mono_ok} {'✅' if mono_ok else '❌'}")

print("\n  3. Cap 70% respetado (DESINV)?")
cap_ok = results["desinversion"]["pct_vender"] <= 70
print(f"     DESINV ventas: {results['desinversion']['pct_vender']:.1f}%  {'✅' if cap_ok else '❌'} (max 70%)")

print("\n  4. Diferenciación ventas PRUD < BAL < DESINV?")
diff_ok = (results["prudencial"]["pct_vender"] < results["balanceado"]["pct_vender"] < 
           results["desinversion"]["pct_vender"])
print(f"     {results['prudencial']['pct_vender']:.1f}% < {results['balanceado']['pct_vender']:.1f}% < {results['desinversion']['pct_vender']:.1f}%: {diff_ok} {'✅' if diff_ok else '❌'}")

print("\n" + "="*80)
print("FIN ANÁLISIS")
print("="*80)
