"""Validar el archivo más reciente"""
import pandas as pd
import glob
import os

# Encontrar el archivo más reciente
pattern = r'reports\coordinated_inference_bank_ready_v*\decisiones_finales_prudencial.xlsx'
files = glob.glob(pattern)
if not files:
    print("No se encontraron archivos")
    exit(1)

latest_file = max(files, key=os.path.getctime)
print(f"Analizando: {latest_file}\n")

df = pd.read_excel(latest_file)

print("="*80)
print("VALIDACIÓN IMPLEMENTACIÓN BANK-READY (LATEST)")
print("="*80)

print(f"\n1. Total préstamos: {len(df)}")

# Case status
print(f"\n2. CASE_STATUS:")
if 'case_status' in df.columns:
    print(df['case_status'].value_counts())
else:
    print("   Columna no existe")

# Casos escalados
escalated = df[df['case_status'] == 'HOLD_NO_EXECUTABLE_ACTION'] if 'case_status' in df.columns else pd.DataFrame()
print(f"\n3. Casos escalados: {len(escalated)}")

if len(escalated) > 0:
    print(f"\n4. Detalle de casos escalados:")
    cols = ['loan_id', 'segment', 'Accion_final', 'Fire_Sale', 'Sell_Blocked', 
            'restruct_viable', 'case_status', 'next_step', 'next_step_reason',
            'review_due_days', 'required_data_flags', 'override_reason']
    cols = [c for c in cols if c in escalated.columns]
    print(escalated[cols].to_string())

# Análisis de Sell_Blocked
if 'Sell_Blocked' in df.columns:
    sell_blocked = df[df['Sell_Blocked'] == True]
    print(f"\n5. Sell_Blocked=True: {len(sell_blocked)}")
    
    if len(sell_blocked) > 0:
        print(f"   - Con restruct_viable=False: {(sell_blocked['restruct_viable'] == False).sum()}")
        print(f"   - Con Accion_final=MANTENER: {(sell_blocked['Accion_final'] == 'MANTENER').sum()}")
        
        # Critical cases
        critical = sell_blocked[(sell_blocked['restruct_viable'] == False) & 
                               (sell_blocked['Accion_final'] == 'MANTENER')]
        print(f"   - CASOS CRÍTICOS (debería tener metadata): {len(critical)}")
        
        if len(critical) > 0:
            print(f"\n   Muestra de casos críticos (primeros 3):")
            cols = ['loan_id', 'segment', 'case_status', 'next_step', 'override_reason']
            cols = [c for c in cols if c in critical.columns]
            print(critical[cols].head(3).to_string())

print("\n\n✅ Análisis completado")
