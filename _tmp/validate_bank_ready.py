"""Validar implementación bank-ready con escalación y metadata"""
import pandas as pd
import numpy as np

output_file = r'reports\coordinated_inference_bank_ready_v1_20260215_192825_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(output_file)

print("="*80)
print("VALIDACIÓN IMPLEMENTACIÓN BANK-READY")
print("="*80)

print(f"\n1. Total préstamos: {len(df)}")

# Verificar que los nuevos campos existen
new_fields = ['case_status', 'next_step', 'next_step_reason', 'review_due_days', 
              'required_data_flags', 'override_reason', 
              'plazo_optimo', 'tasa_nueva', 'quita']

missing_fields = [f for f in new_fields if f not in df.columns]
if missing_fields:
    print(f"\n⚠️ CAMPOS FALTANTES: {missing_fields}")
else:
    print(f"\n✅ Todos los campos bank-ready presentes")

# Analizar casos de escalación
print(f"\n2. CASE_STATUS:")
if 'case_status' in df.columns:
    print(df['case_status'].value_counts())

# Casos con escalación (HOLD_NO_EXECUTABLE_ACTION)
if 'case_status' in df.columns:
    escalated = df[df['case_status'] == 'HOLD_NO_EXECUTABLE_ACTION']
    print(f"\n3. Casos escalados (HOLD_NO_EXECUTABLE_ACTION): {len(escalated)}")
    
    if len(escalated) > 0:
        print(f"\n4. Análisis de casos escalados:")
        print(f"   - next_step:")
        print(escalated['next_step'].value_counts())
        
        print(f"\n   - next_step_reason:")
        print(escalated['next_step_reason'].value_counts())
        
        print(f"\n   - review_due_days (mean): {escalated['review_due_days'].mean():.0f} días")
        
        print(f"\n   - Accion_final de casos escalados:")
        print(escalated['Accion_final'].value_counts())
        
        print(f"\n   - Fire_Sale detectado:")
        print(escalated['Fire_Sale'].value_counts() if 'Fire_Sale' in escalated.columns else "N/A")
        
        print(f"\n   - restruct_viable:")
        print(escalated['restruct_viable'].value_counts() if 'restruct_viable' in escalated.columns else "N/A")
        
        print(f"\n5. Muestra de casos escalados (primeros 5):")
        cols = ['loan_id', 'segment', 'Accion_final', 'case_status', 'next_step', 
                'next_step_reason', 'review_due_days', 'required_data_flags', 
                'override_reason', 'Fire_Sale', 'restruct_viable']
        cols = [c for c in cols if c in escalated.columns]
        print(escalated[cols].head().to_string())

# Verificar propagación de parámetros de reestructura
restruct = df[df['Accion_final'] == 'REESTRUCTURAR']
print(f"\n6. REESTRUCTURAS: {len(restruct)}")

if len(restruct) > 0 and 'plazo_optimo' in df.columns:
    print(f"   - plazo_optimo poblado: {restruct['plazo_optimo'].notna().sum()}/{len(restruct)}")
    print(f"   - tasa_nueva poblado: {restruct['tasa_nueva'].notna().sum()}/{len(restruct)}")
    print(f"   - quita poblado: {restruct['quita'].notna().sum()}/{len(restruct)}")
    
    if restruct['plazo_optimo'].notna().sum() > 0:
        print(f"\n   Muestra de reestructuras con parámetros (primeras 5):")
        cols = ['loan_id', 'segment', 'plazo_optimo', 'tasa_nueva', 'quita', 
                'PTI_post', 'DSCR_post', 'restruct_viable']
        cols = [c for c in cols if c in restruct.columns]
        print(restruct[restruct['plazo_optimo'].notna()][cols].head().to_string())

# Verificar override_reason
if 'override_reason' in df.columns:
    overrides = df[df['override_reason'] != '']
    print(f"\n7. OVERRIDES: {len(overrides)}")
    if len(overrides) > 0:
        print(df['override_reason'].value_counts())

print("\n\n✅ Validación completada")
