"""Ver mix de decisiones micro vs coordinador"""
import pandas as pd

micro_file = r'reports\inference_20260215_191358_coord_micro_debug_pti\decisiones_explicadas.xlsx'
df_micro = pd.read_excel(micro_file)

coordinated_file = r'reports\coordinated_inference_debug_pti_20260215_191405_prudencial\decisiones_finales_prudencial.xlsx'
df_coord = pd.read_excel(coordinated_file)

print("="*80)
print("ANÁLISIS MIX DE DECISIONES")
print("="*80)

print(f"\n=== MICRO (policy_inference.py) ===")
print(df_micro['Accion'].value_counts())
print(f"\nTotal: {len(df_micro)}")

print(f"\n=== MICRO column names ===")
print([c for c in df_micro.columns if 'Accion' in c or 'score' in c or 'Score' in c][:10])

print(f"\n=== COORDINADOR (decisión final) ===")
print(df_coord['Accion_final'].value_counts())
print(f"\nTotal: {len(df_coord)}")

print(f"\n=== Convergencia ===")
if 'Accion_micro' in df_coord.columns and 'Accion_macro' in df_coord.columns:
    convergence = df_coord.groupby(['Accion_micro', 'Accion_macro', 'Accion_final']).size()
    print(convergence.head(20))
else:
    print("Columnas de convergencia no disponibles")

# Ver una muestra de reestructuras finales
restruct = df_coord[df_coord['Accion_final'] == 'REESTRUCTURAR']
if len(restruct) > 0:
    print(f"\n=== Muestra reestructuras (origen de decisión) ===")
    cols = ['loan_id', 'Accion_micro', 'Accion_macro', 'Accion_final', 'PTI_post', 'DSCR_post']
    cols = [c for c in cols if c in restruct.columns]
    print(restruct[cols].head(10).to_string())

print("\n✅ Análisis completado")
