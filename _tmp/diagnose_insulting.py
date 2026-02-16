"""
Diagnose: Why 100% insulting when precio_optimo/valor_ref ratio = 28% > floor 20%?
"""
import pandas as pd
import numpy as np

pru_file = r"reports\coordinated_inference_v3_NPL_LOSS_CAP_pru_20260215_234655_prudencial\decisiones_finales_prudencial.xlsx"
df = pd.read_excel(pru_file)

print("\n" + "="*70)
print("DIAGNOSIS: Why 100% insult when ratio = 28% > floor = 20%?")
print("="*70)

# Check available price columns
price_cols = [col for col in df.columns if 'prec' in col.lower() or 'price' in col.lower()]
print(f"\n[1] AVAILABLE PRICE COLUMNS:")
for col in price_cols:
    print(f"  - {col}")

# Key columns
precio_optimo = df.get("precio_optimo", pd.Series([np.nan]*len(df)))
precio_NPL = df.get("precio_NPL", pd.Series([np.nan]*len(df)))
valor_ref = df["valor_referencia"]
EAD = df["EAD"]

print(f"\n[2] MEDIANS:")
print(f"  precio_optimo median:  {precio_optimo.median():,.0f}")
print(f"  precio_NPL median:     {precio_NPL.median():,.0f}")
print(f"  valor_referencia median: {valor_ref.median():,.0f}")
print(f"  EAD median:            {EAD.median():,.0f}")

# Ratios
if not precio_NPL.isna().all():
    ratio_NPL = precio_NPL / valor_ref
    print(f"\n[3] RATIO precio_NPL / valor_ref:")
    print(f"  p25:    {ratio_NPL.quantile(0.25):.4f} ({ratio_NPL.quantile(0.25)*100:.2f}%)")
    print(f"  Median: {ratio_NPL.median():.4f} ({ratio_NPL.median()*100:.2f}%)")
    print(f"  p75:    {ratio_NPL.quantile(0.75):.4f} ({ratio_NPL.quantile(0.75)*100:.2f}%)")
    
    # Check floor=20%
    n_pass = (ratio_NPL >= 0.20).sum()
    n_fail = (ratio_NPL < 0.20).sum()
    print(f"\n[4] FLOOR TEST (floor=20%):")
    print(f"  PASS (>= 20%): {n_pass}/500")
    print(f"  FAIL (<  20%): {n_fail}/500")
    
    # Sample failures
    if n_fail > 0:
        fails = df[ratio_NPL < 0.20].head(10)
        print(f"\n[5] SAMPLE FAILURES (ratio < 20%):")
        for idx, row in fails.iterrows():
            loan_id = row.get("loan_id", idx)
            pnpl = row.get("precio_NPL", np.nan)
            vref = row.get("valor_referencia", np.nan)
            ratio_val = pnpl / vref if vref > 0 else 0
            print(f"  {loan_id}: precio_NPL={pnpl:,.0f}, valor_ref={vref:,.0f}, ratio={ratio_val:.2%}")
else:
    print(f"\n[ERROR] precio_NPL is ALL NaN!")
    print(f"This means price_simulator didn't populate precio_NPL")
    
ratio_optimo = precio_optimo / valor_ref
print(f"\n[6] RATIO precio_optimo / valor_ref (for comparison):")
print(f"  Median: {ratio_optimo.median():.4f} ({ratio_optimo.median()*100:.2f}%)")
print(f"  Loans >= 20%: {(ratio_optimo >= 0.20).sum()}/500")

print("="*70)
