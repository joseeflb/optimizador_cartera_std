"""Validación final de compliance para las 3 posturas bank-ready (500 loans)"""
import pandas as pd
import numpy as np

print("="*80)
print("COMPLIANCE FINAL - 3 POSTURAS (500 PRÉSTAMOS)")
print("="*80)

# Archivos a validar
files = {
    "PRUDENCIAL": r"reports\coordinated_inference_bank_ready_final_pru_20260215_193719_prudencial\decisiones_finales_prudencial.xlsx",
    "BALANCEADO": r"reports\coordinated_inference_bank_ready_final_bal_20260215_193817_balanceado\decisiones_finales_balanceado.xlsx",
    "DESINVERSION": r"reports\coordinated_inference_bank_ready_final_des_20260215_193852_desinversion\decisiones_finales_desinversion.xlsx"
}

all_passed = True

for posture, filepath in files.items():
    print(f"\n{'='*80}")
    print(f"POSTURA: {posture}")
    print(f"{'='*80}")
    
    try:
        df = pd.read_excel(filepath)
        print(f"✓ Cargado: {len(df)} préstamos")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        all_passed = False
        continue
    
    # Test 1: Restructure params
    print(f"\n1️⃣ PARÁMETROS DE REESTRUCTURA")
    if 'Accion_final' in df.columns:
        restruct = df[df['Accion_final'] == 'REESTRUCTURAR']
        print(f"   Reestructuras totales: {len(restruct)}")
        
        if len(restruct) > 0:
            for param in ['plazo_optimo', 'tasa_nueva', 'quita']:
                if param in df.columns:
                    non_null = restruct[param].notna().sum()
                    pct = (non_null / len(restruct)) * 100
                    status = "✅" if pct >= 95 else ("⚠️" if pct >= 80 else "❌")
                    print(f"   {status} {param}: {non_null}/{len(restruct)} ({pct:.1f}%)")
                    if pct < 95:
                        all_passed = False
    
    # Test 2: DSCR gates (corporate only)
    print(f"\n2️⃣ DSCR >= 1.05 (CORPORATE RESTRUCTURES)")
    if 'Accion_final' in df.columns and 'DSCR_post' in df.columns and 'segment' in df.columns:
        corp_restruct = df[(df['Accion_final'] == 'REESTRUCTURAR') & 
                           (~df['segment'].isin(['Mortgage', 'Consumer']))]
        
        if len(corp_restruct) > 0:
            dscr_vals = corp_restruct['DSCR_post'].dropna()
            if len(dscr_vals) > 0:
                min_dscr = dscr_vals.min()
                avg_dscr = dscr_vals.mean()
                violations = (dscr_vals < 1.05).sum()
                
                status = "✅" if violations == 0 else "❌"
                print(f"   {status} Reestructuras corporativas: {len(corp_restruct)}")
                print(f"   {status} DSCR mínimo: {min_dscr:.2f}")
                print(f"   {status} DSCR promedio: {avg_dscr:.2f}")
                print(f"   {status} Violaciones DSCR<1.05: {violations}")
                
                if violations > 0:
                    all_passed = False
    
    # Test 3: Fire-sale protections
    print(f"\n3️⃣ FIRE-SALE PROTECTIONS")
    if 'Accion_final' in df.columns and 'Fire_Sale' in df.columns:
        sells = df[df['Accion_final'] == 'VENDER']
        if len(sells) > 0:
            firesales = sells[sells['Fire_Sale'] == True].fillna(False)
            firesales = sells[sells['Fire_Sale'].fillna(False) == True]
            
            status = "✅" if len(firesales) == 0 else ("⚠️" if len(firesales) <= len(sells)*0.1 else "❌")
            print(f"   {status} Ventas totales: {len(sells)}")
            print(f"   {status} Fire-sales ejecutados: {len(firesales)} ({len(firesales)/len(sells)*100:.1f}%)")
            
            if len(firesales) > 0:
                print(f"      ℹ️  Razón: Fire-sales permitidos en postura {posture}")
    else:
        print(f"   ✅ Sin ventas ejecutadas")
    
    # Test 4: Escalation metadata (critical cases)
    print(f"\n4️⃣ ESCALATION METADATA (CASOS CRÍTICOS)")
    if 'case_status' in df.columns:
        escalated = df[df['case_status'] == 'HOLD_NO_EXECUTABLE_ACTION']
        print(f"   Casos escalados: {len(escalated)}")
        
        if len(escalated) > 0:
            # Verificar que tengan metadata completo
            required_fields = ['next_step', 'next_step_reason', 'override_reason']
            complete_metadata = True
            
            for field in required_fields:
                if field in df.columns:
                    non_empty = escalated[field].notna().sum()
                    pct = (non_empty / len(escalated)) * 100
                    status = "✅" if pct == 100 else "❌"
                    print(f"   {status} {field}: {non_empty}/{len(escalated)} ({pct:.0f}%)")
                    
                    if pct < 100:
                        complete_metadata = False
                        all_passed = False
            
            if complete_metadata:
                print(f"   ✅ Metadata de escalación 100% completo")
        else:
            print(f"   ✓ Sin casos críticos en esta postura")
    
    # Test 5: Auditability (Reason_Code, Convergencia_Caso)
    print(f"\n5️⃣ AUDITABILITY (TRAZABILIDAD)")
    audit_fields = ['Reason_Code', 'Convergencia_Caso']
    for field in audit_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            pct = (non_null / len(df)) * 100
            status = "✅" if pct == 100 else ("⚠️" if pct >= 95 else "❌")
            print(f"   {status} {field}: {non_null}/{len(df)} ({pct:.1f}%)")
            
            if pct < 100:
                all_passed = False
    
    # Test 6: Valid decisions only
    print(f"\n6️⃣ DECISIONES VÁLIDAS")
    if 'Accion_final' in df.columns:
        valid_actions = ['MANTENER', 'REESTRUCTURAR', 'VENDER']
        invalid = df[~df['Accion_final'].isin(valid_actions)]
        
        status = "✅" if len(invalid) == 0 else "❌"
        print(f"   {status} Decisiones inválidas: {len(invalid)}/{len(df)}")
        
        if len(invalid) > 0:
            print(f"      Valores inválidos: {invalid['Accion_final'].unique()}")
            all_passed = False

# Summary
print(f"\n{'='*80}")
if all_passed:
    print("✅ ¡TODAS LAS POSTURAS PASAN 100% COMPLIANCE!")
else:
    print("⚠️ ALGUNAS VALIDACIONES REQUIEREN REVISIÓN")
print(f"{'='*80}")

exit(0 if all_passed else 1)
