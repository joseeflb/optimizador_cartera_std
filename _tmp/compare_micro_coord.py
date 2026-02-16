"""Verificar si el problema está en el paso micro o en el coordinador"""
import pandas as pd

# Archivo generado por el paso MICRO (antes del coordinador)
micro_file = r'reports\inference_20260215_191358_coord_micro_debug_pti\decisiones_explicadas.xlsx'
df_micro = pd.read_excel(micro_file)

# Archivo final del COORDINADOR
coordinated_file = r'reports\coordinated_inference_debug_pti_20260215_191405_prudencial\decisiones_finales_prudencial.xlsx'
df_coord = pd.read_excel(coordinated_file)

print("="*80)
print("COMPARACIÓN MICRO vs COORDINADOR")
print("="*80)

# Columnas relacionadas con ingresos
income_cols_micro = [c for c in df_micro.columns if 'income' in c.lower() or 'ingreso' in c.lower()]
print(f"\n1. Columnas income en MICRO: {income_cols_micro}")

income_cols_coord = [c for c in df_coord.columns if 'income' in c.lower() or 'ingreso' in c.lower()]
print(f"2. Columnas income en COORDINADOR: {income_cols_coord}")

# PTI en ambos
restruct_micro = df_micro[df_micro['Accion'] == 'REESTRUCTURAR'].copy()
print(f"\n3. MICRO - Reestructuras: {len(restruct_micro)}")
print(f"   PTI_post NaN: {restruct_micro['PTI_post'].isna().sum() if 'PTI_post' in restruct_micro.columns else 'columna no existe'}")

restruct_coord = df_coord[df_coord['Accion_final'] == 'REESTRUCTURAR'].copy()
print(f"\n4. COORDINADOR - Reestructuras: {len(restruct_coord)}")
print(f"   PTI_post NaN: {restruct_coord['PTI_post'].isna().sum() if 'PTI_post' in restruct_coord.columns else 'columna no existe'}")

# Muestra de una reestructura de cada archivo
if len(restruct_micro) > 0:
    print(f"\n5. Muestra MICRO (primera reestructura):")
    cols = ['loan_id', 'Accion', 'PTI_pre', 'PTI_post', 'DSCR_pre', 'DSCR_post', 'restruct_viable']
    cols = [c for c in cols if c in restruct_micro.columns]
    print(restruct_micro[cols].iloc[0])

if len(restruct_coord) > 0:
    print(f"\n6. Muestra COORDINADOR (primera reestructura):")
    cols = ['loan_id', 'Accion_final', 'PTI_pre', 'PTI_post', 'DSCR_pre', 'DSCR_post', 'restruct_viable']
    cols = [c for c in cols if c in restruct_coord.columns]
    print(restruct_coord[cols].iloc[0])

print("\n\n✅ Comparación completada")
