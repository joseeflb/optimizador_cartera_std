"""
Check if new loss_cap (80%) is actually being applied
"""
import pandas as pd

pru_file = r"reports\coordinated_inference_v3_NPL_LOSS_CAP_pru_20260215_234655_prudencial\decisiones_finales_prudencial.xlsx"
df = pd.read_excel(pru_file)

print("\n" + "="*70)
print("VERIFICATION: PRUDENCIAL loss_cap=80% application")
print("="*70)

# Check sale gates
n_insulting =df.get("sale_insulting_flag", pd.Series([False]*500)).sum()
n_within_cap = df.get("sale_within_loss_cap", pd.Series([False]*500)).sum()
n_executable = df.get("sale_executable", pd.Series([False]*500)).sum()

print(f"\n[1] SALE GATES:")
print(f"  sale_insulting_flag:     {n_insulting}/500")
print(f"  sale_within_loss_cap:    {n_within_cap}/500")
print(f"  sale_executable:         {n_executable}/500")

# Check actual loss distribution
if "sale_loss_pct" in df.columns:
    loss = df["sale_loss_pct"]
    print(f"\n[2] LOSS DISTRIBUTION:")
    print(f"  Median loss:  {loss.median():.2%}")
    print(f"  p75 loss:     {loss.quantile(0.75):.2%}")
    print(f"  p90 loss:     {loss.quantile(0.90):.2%}")
    print(f"  p95 loss:     {loss.quantile(0.95):.2%}")
    
    # Test different caps
    print(f"\n[3] LOANS PASSING LOSS_CAP thresholds:")
    for cap in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
        n_ok = (loss <= cap).sum()
        print(f"  loss_cap={cap:.0%}: {n_ok:3d}/500 pass ({n_ok/5:.1f}%)")
else:
    print("\n[ERROR] sale_loss_pct column not found!")

# Check reason codes
if "sale_reason_code" in df.columns:
    print(f"\n[4] SALE_REASON_CODE distribution:")
    reason_counts = df["sale_reason_code"].value_counts()
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}")

print("="*70)
