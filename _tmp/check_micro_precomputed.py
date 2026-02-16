"""Ver si el archivo MICRO tiene parámetros de reestructura precomputados"""
import pandas as pd

# Archivo generado por policy_inference (MICRO) - ANTES del coordinador
micro_file = r'reports\inference_20260215_191358_coord_micro_debug_pti\decisiones_explicadas.xlsx'
df_micro = pd.read_excel(micro_file)

print("="*80)
print("ANÁLISIS OUTPUT MICRO (policy_inference)")
print("="*80)

print(f"\n1. Total préstamos: {len(df_micro)}")

# Buscar columnas de reestructura
restruct_cols = [c for c in df_micro.columns if any(x in c.lower() for x in ['plazo', 'tasa', 'quita', 'pti_post', 'dscr_post'])]
print(f"\n2. Columnas relacionadas con reestructura: {restruct_cols}")

if len(restruct_cols) > 0:
    print(f"\n3. Stats de columnas clave:")
    for col in ['PTI_post', 'DSCR_post', 'plazo_optimo', 'tasa_nueva', 'quita']:
        if col in df_micro.columns:
            non_nan = df_micro[col].notna().sum()
            print(f"   {col:15}: {non_nan}/{len(df_micro)} valores no-NaN ({non_nan/len(df_micro)*100:.1f}%)")

# Ver muestra de préstamos que luego fueron forzados a REESTRUCTURAR
# (préstamos con Accion=VENDER en micro pero que el coordinador cambió a REESTRUCTURAR)
vender_micro = df_micro[df_micro['Accion'] == 'VENDER'].copy()
print(f"\n4. Préstamos con Accion=VENDER en MICRO: {len(vender_micro)}")

if len(vender_micro) > 0 and 'DSCR_post' in vender_micro.columns:
    dscr_populated = vender_micro['DSCR_post'].notna().sum()
    print(f"   Con DSCR_post: {dscr_populated}/{len(vender_micro)}")
    
    print(f"\n5. Muestra de VENDER con datos de viabilidad:")
    sample_cols = ['loan_id', 'Accion', 'segment', 'DSCR_pre', 'DSCR_post', 'restruct_viable']
    sample_cols = [c for c in sample_cols if c in vender_micro.columns]
    print(vender_micro[sample_cols].head(10).to_string())

print("\n\n✅ Análisis completado")
