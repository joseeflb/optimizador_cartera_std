"""Verificar qué segmentos fueron reestructurados"""
import pandas as pd

coord_file = r'reports\coordinated_inference_debug_pti_20260215_191405_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(coord_file)

restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()

print("="*80)
print("ANÁLISIS REESTRUCTURAS POR SEGMENTO")
print("="*80)

print(f"\nTotal reestructuras: {len(restruct)}")

print(f"\n1. Mix por segmento:")
print(restruct['segment'].value_counts())

print(f"\n2. Retail (Mortgage) reestructuras:")
retail = restruct[restruct['segment'].isin(['Mortgage', 'Consumer'])]
print(f"   Total: {len(retail)}")
if len(retail) > 0:
    print(f"   PTI_post NaN: {retail['PTI_post'].isna().sum()}")
    print(f"\n   Muestra:")
    print(retail[['loan_id', 'segment', 'PTI_pre', 'PTI_post', 'DSCR_post']].head().to_string())

print(f"\n3. Corporate reestructuras:")
corp = restruct[~restruct['segment'].isin(['Mortgage', 'Consumer'])]
print(f"   Total: {len(corp)}")
if len(corp) > 0:
    print(f"   DSCR_post NaN: {corp['DSCR_post'].isna().sum()}")
    print(f"   PTI_post NaN: {corp['PTI_post'].isna().sum()} (expected for corporate)")
    print(f"\n   Muestra:")
    print(corp[['loan_id', 'segment', 'DSCR_pre', 'DSCR_post', 'PTI_post']].head().to_string())

print("\n\n✅ Análisis completado")
