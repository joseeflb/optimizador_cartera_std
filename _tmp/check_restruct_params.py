"""Verificar si optimize_restructure se ejecutó (plazo/tasa/quita)"""
import pandas as pd

coord_file = r'reports\coordinated_inference_debug_pti_20260215_191405_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(coord_file)

restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()

print("="*80)
print("VERIFICACIÓN DE PARÁMETROS DE REESTRUCTURA")
print("="*80)

print(f"\nTotal reestructuras: {len(restruct)}")

# Columns related to restructuring
key_cols = ['loan_id', 'segment', 'DSCR_pre', 'DSCR_post', 'plazo_optimo', 'tasa_nueva', 'quita', 'restruct_viable']
key_cols = [c for c in key_cols if c in restruct.columns]

print(f"\n1. Stats de parámetros:")
if 'plazo_optimo' in restruct.columns:
    print(f"   plazo_optimo: {restruct['plazo_optimo'].describe()}")
if 'tasa_nueva' in restruct.columns:
    print(f"   tasa_nueva: {restruct['tasa_nueva'].describe()}")
if 'quita' in restruct.columns:
    print(f"   quita: {restruct['quita'].describe()}")

print(f"\n2. Muestra de reestructuras (primeras 10):")
print(restruct[key_cols].head(10).to_string())

print(f"\n3. Verificación restruct_viable:")
if 'restruct_viable' in restruct.columns:
    print(f"   True: {(restruct['restruct_viable'] == True).sum()}")
    print(f"   False: {(restruct['restruct_viable'] == False).sum()}")
    print(f"   NaN/None: {restruct['restruct_viable'].isna().sum()}")

print("\n\n✅ Análisis completado")
