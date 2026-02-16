"""Script rápido para verificar si PTI_post está calculado después del fix"""
import pandas as pd

file_path = r'reports\coordinated_inference_test_fix_pti_20260215_190625_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(file_path)

restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()

print("="*80)
print("VERIFICACIÓN PTI_post FIX")
print("="*80)
print(f"\nTotal reestructuras: {len(restruct)}")
print(f"PTI_post NaN: {restruct['PTI_post'].isna().sum()}")
print(f"\nPTI_post stats:")
print(restruct['PTI_post'].describe())

print(f"\n\n=== Por segmento ===")
for seg in ['CONSUMER', 'MORTGAGE', 'SME', 'CORPORATE', 'OTHER']:
    seg_data = restruct[restruct['segmento'] == seg]
    if len(seg_data) > 0:
        pti_nan = seg_data['PTI_post'].isna().sum()
        dscr_nan = seg_data['DSCR_post'].isna().sum()
        print(f"{seg:12} | Total: {len(seg_data):3} | PTI_post NaN: {pti_nan:3} | DSCR_post NaN: {dscr_nan:3}")

print(f"\n\n=== Muestra: Reestructuras retail (CONSUMER/MORTGAGE) ===")
retail = restruct[restruct['segmento'].isin(['CONSUMER', 'MORTGAGE'])]
print(retail[['loan_id', 'segmento', 'PTI_pre', 'PTI_post', 'plazo_optimo', 'tasa_nueva']].head(10))

print(f"\n\n=== Muestra: Reestructuras corporate ===")
corp = restruct[restruct['segmento'].isin(['CORPORATE', 'SME'])]
print(corp[['loan_id', 'segmento', 'DSCR_pre', 'DSCR_post', 'plazo_optimo', 'tasa_nueva']].head(10))

print("\n\n✅ Análisis completado")
