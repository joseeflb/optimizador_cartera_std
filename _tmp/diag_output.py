"""Diagnóstico completo del archivo de salida"""
import pandas as pd

file_path = r'reports\coordinated_inference_test_fix_pti_20260215_190625_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(file_path)

print("="*80)
print("DIAGNÓSTICO ARCHIVO DE SALIDA")
print("="*80)

print(f"\n1. Shape: {df.shape}")
print(f"\n2. Columnas disponibles ({len(df.columns)}):")
cols = sorted(df.columns.tolist())
for i, col in enumerate(cols, 1):
    print(f"   {i:2}. {col}")

restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()
print(f"\n3. Total reestructuras: {len(restruct)}")

# Buscar columnas relacionadas con PTI
pti_cols = [c for c in df.columns if 'PTI' in c.upper() or 'pti' in c]
print(f"\n4. Columnas PTI: {pti_cols}")

# Buscar columnas relacionadas con income
income_cols = [c for c in df.columns if 'income' in c.lower() or 'ingreso' in c.lower()]
print(f"\n5. Columnas income: {income_cols}")

# Ver si hay columnas de viabilidad
viab_cols = [c for c in df.columns if 'viab' in c.lower() or 'restruct' in c.lower()]
print(f"\n6. Columnas viabilidad/restruct: {viab_cols}")

# Segmento
seg_cols = [c for c in df.columns if 'seg' in c.lower()]
print(f"\n7. Columnas segmento: {seg_cols}")

if len(restruct) > 0:
    print(f"\n8. Primeras 3 filas de reestructuras (columnas claves):")
    key_cols = ['loan_id', 'Accion_final'] + pti_cols + income_cols + seg_cols[:2]
    key_cols = [c for c in key_cols if c in restruct.columns]
    print(restruct[key_cols].head(3).to_string())

print("\n\n✅ Diagnóstico completado")
