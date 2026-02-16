"""Ver valores de monthly_income en portfolio de entrada"""
import pandas as pd

df = pd.read_excel(r'data\portfolio_synth_smoke.xlsx')

print("="*80)
print("ANÁLISIS DE monthly_income EN PORTFOLIO")
print("="*80)

print(f"\n1. monthly_income stats:")
print(df["monthly_income"].describe())

print(f"\n2. NaN count: {df['monthly_income'].isna().sum()}/{len(df)}")
print(f"   None count: {df['monthly_income'].isnull().sum()}/{len(df)}")

print(f"\n3. Sample (first 10):")
print(df[["loan_id", "segment", "monthly_income", "monthly_cfo"]].head(10).to_string())

print(f"\n4. Por segmento:")
for seg in df['segment'].unique():
    seg_data = df[df['segment'] == seg]
    missing = seg_data['monthly_income'].isna().sum()
    total = len(seg_data)
    print(f"   {seg:12} | Total: {total:3} | monthly_income NaN: {missing:3} ({missing/total*100 if total>0 else 0:.1f}%)")

print("\n✅ Análisis completado")
