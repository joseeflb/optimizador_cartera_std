"""Check if old Excel has precio_NPL"""
import pandas as pd

old_file = r"reports\inference_20260208_120558_coord_micro_run1_prudencial\decisiones_explicadas.xlsx"
df = pd.read_excel(old_file)

print("[OLD INFERENCE] Available columns with 'precio':")
price_cols = [col for col in df.columns if 'precio' in col.lower()]
for col in price_cols:
    n_notnull = df[col].notnull().sum()
    median = df[col].median()
    print(f"  {col}: {n_notnull}/500 not null, median={median:,.0f}")
