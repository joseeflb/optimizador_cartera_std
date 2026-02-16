"""
Quick check: What % of loans have precio_optimo >= 20% valor_ref?
This verifies if floor=20% is still too high.
"""
import pandas as pd
import numpy as np

# Load PRUDENCIAL results
pru_file = r"reports\coordinated_inference_executability_v2_RECALIBRADO_pru_20260215_233338_prudencial\decisiones_finales_prudencial.xlsx"
df = pd.read_excel(pru_file)

print("\n" + "="*70)
print("PRUDENCIAL FLOOR ANALYSIS (floor=20%)")
print("="*70)

# Get pricing columns
precio = df["precio_optimo"]
valor_ref = df["valor_referencia"]
ratio = precio / valor_ref

print(f"\n[1] DISTRIBUCIÓN RATIO precio/valor_ref:")
print(f"  p5:     {ratio.quantile(0.05):.4f} ({ratio.quantile(0.05)*100:.2f}%)")
print(f"  p10:    {ratio.quantile(0.10):.4f} ({ratio.quantile(0.10)*100:.2f}%)")
print(f"  p25:    {ratio.quantile(0.25):.4f} ({ratio.quantile(0.25)*100:.2f}%)")
print(f"  Median: {ratio.median():.4f} ({ratio.median()*100:.2f}%)")
print(f"  p75:    {ratio.quantile(0.75):.4f} ({ratio.quantile(0.75)*100:.2f}%)")
print(f"  p90:    {ratio.quantile(0.90):.4f} ({ratio.quantile(0.90)*100:.2f}%)")
print(f"  p95:    {ratio.quantile(0.95):.4f} ({ratio.quantile(0.95)*100:.2f}%)")

# Test different floors
print(f"\n[2] % EJECUTABLE CON DIFERENTES FLOORS:")
for floor_pct in [5, 10, 15, 20, 25, 30]:
    floor = floor_pct / 100.0
    n_ok = (ratio >= floor).sum()
    pct_ok = (n_ok / len(df)) * 100
    print(f"  Floor {floor_pct:2d}%: {n_ok:3d}/500 ejecutables ({pct_ok:5.1f}%)")

# Analyze sale_mandate
n_mandate = df.get("sale_mandate", pd.Series([False]*500)).sum()
n_insulting = df.get("sale_insulting_flag", pd.Series([False]*500)).sum()
n_executable = df.get("sale_executable", pd.Series([False]*500)).sum()

print(f"\n[3] GATES APLICADOS (PRUDENCIAL floor=20%):")
print(f"  sale_mandate:        {n_mandate}/500")
print(f"  sale_insulting_flag: {n_insulting}/500")
print(f"  sale_executable:     {n_executable}/500")

# Check loss_cap impact
EAD = df["EAD"]
loss_abs = EAD - precio
loss_pct = loss_abs / EAD

print(f"\n[4] LOSS CAP ANALYSIS (loss_cap=50%):")
print(f"  Loss% median:   {loss_pct.median():.4f} ({loss_pct.median()*100:.2f}%)")
print(f"  Loss% p75:      {loss_pct.quantile(0.75):.4f} ({loss_pct.quantile(0.75)*100:.2f}%)")
print(f"  Loss% p90:      {loss_pct.quantile(0.90):.4f} ({loss_pct.quantile(0.90)*100:.2f}%)")
print(f"  Loans with loss > 50%: {(loss_pct > 0.50).sum()}/500")

# CONCLUSION
print(f"\n[5] DIAGNOSTICO:")
if ratio.median() < 0.20:
    print(f"  ❌ PROBLEMA: Ratio mediano ({ratio.median()*100:.1f}%) < floor (20%)")
    print(f"  → Floor PRUDENCIAL=20% es DEMASIADO ALTO para esta cartera")
    print(f"  → Sugerencia: bajar a {int(ratio.quantile(0.30)*100)}% (p30) para ~70% ejecutable")
else:
    print(f"  ✅ OK: Ratio mediano ({ratio.median()*100:.1f}%) >= floor (20%)")

print("="*70)
