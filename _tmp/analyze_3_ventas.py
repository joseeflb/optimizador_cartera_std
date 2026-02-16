"""
Analyze why only 3/500 sales ejecutable with floor=20% + loss_cap=80%
"""
import pandas as pd
import numpy as np

pru_file = r"reports\coordinated_inference_v4_FINAL_FIX_pru_20260215_235320_prudencial\decisiones_finales_prudencial.xlsx"
df = pd.read_excel(pru_file)

print("\n" + "="*70)
print("ANALYSIS: Why only 3/500 ventas ejecutables?")
print("="*70)

# Check gates
sale_price = df["precio_optimo"]
valor_ref = df["valor_referencia"]
EAD = df["EAD"]
ratio = sale_price / valor_ref
loss = (EAD - sale_price) / EAD

n_insulting = df.get("sale_insulting_flag", pd.Series([False]*500)).sum()
n_within_cap = df.get("sale_within_loss_cap", pd.Series([False]*500)).sum()
n_executable = df.get("sale_executable", pd.Series([False]*500)).sum()

print(f"\n[1] GATE RESULTS:")
print(f"  sale_insulting_flag (price < 20% valor_ref): {n_insulting}/500")
print(f"  sale_within_loss_cap (loss <= 80%):          {n_within_cap}/500")
print(f"  sale_executable (NOT insulting AND within cap): {n_executable}/500")

# Ratios
print(f"\n[2] RATIO precio/valor_ref:")
print(f"  p10:    {ratio.quantile(0.10):.4f} ({ratio.quantile(0.10)*100:.2f}%)")
print(f"  p25:    {ratio.quantile(0.25):.4f} ({ratio.quantile(0.25)*100:.2f}%)")
print(f"  Median: {ratio.median():.4f} ({ratio.median()*100:.2f}%)")
print(f"  Loans >= floor(20%): {(ratio >= 0.20).sum()}/500")
print(f"  Loans <  floor(20%): {(ratio < 0.20).sum()}/500")

# Loss distribution
print(f"\n[3] LOSS (EAD - price) / EAD:")
print(f"  p10:    {loss.quantile(0.10):.4f} ({loss.quantile(0.10)*100:.2f}%)")
print(f"  p25:    {loss.quantile(0.25):.4f} ({loss.quantile(0.25)*100:.2f}%)")
print(f"  Median: {loss.median():.4f} ({loss.median()*100:.2f}%)")
print(f"  p75:    {loss.quantile(0.75):.4f} ({loss.quantile(0.75)*100:.2f}%)")
print(f"  p90:    {loss.quantile(0.90):.4f} ({loss.quantile(0.90)*100:.2f}%)")
print(f"  Loans <= loss_cap(80%): {(loss <= 0.80).sum()}/500")
print(f"  Loans >  loss_cap(80%): {(loss > 0.80).sum()}/500")

# BOTH gates
both_pass = (ratio >= 0.20) & (loss <= 0.80)
print(f"\n[4] BOTH GATES:")
print(f"  Loans passing BOTH (floor + loss_cap): {both_pass.sum()}/500")

# Analyze the 3 that executed
ventas = df[df["Accion_final"] == "VENDER"]
print(f"\n[5] THE 3 VENTAS:")
for idx, row in ventas.iterrows():
    loan_id = row.get("loan_id", idx)
    precio = row["precio_optimo"]
    vref = row["valor_referencia"]
    ead = row["EAD"]
    r = precio / vref
    l = (ead - precio) / ead
    exec_status = row.get("execution_status", "?")
    print(f"  {loan_id}: precio={precio:,.0f}, vref={vref:,.0f}, EAD={ead:,.0f}")
    print(f"    ratio={r:.2%}, loss={l:.2%}, status={exec_status}")

# Sample failures
fail_ratio = df[(ratio < 0.20)].head(5)
fail_loss = df[(ratio >= 0.20) & (loss > 0.80)].head(5)

print(f"\n[6] SAMPLE FAILURES - RATIO < 20%:")
for idx, row in fail_ratio.iterrows():
    loan_id = row.get("loan_id", idx)
    precio = row["precio_optimo"]
    vref = row["valor_referencia"]
    r = precio / vref
    print(f"  {loan_id}: ratio={r:.2%} (precio={precio:,.0f}, vref={vref:,.0f})")

print(f"\n[7] SAMPLE FAILURES - LOSS > 80%:")
for idx, row in fail_loss.iterrows():
    loan_id = row.get("loan_id", idx)
    ead = row["EAD"]
    precio = row["precio_optimo"]
    l = (ead - precio) / ead
    print(f"  {loan_id}: loss={l:.2%} (EAD={ead:,.0f}, precio={precio:,.0f})")

print("="*70)
