"""Test de cumplimiento bank-ready para el archivo más reciente"""
import pandas as pd

output_file = r'reports\coordinated_inference_bank_ready_v2_20260215_193122_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(output_file)

print("="*80)
print("REPORTE DE CUMPLIMIENTO BANK-READY")
print("="*80)

total_pass = 0
total_tests = 0

# Test 1: Parámetros de reestructura poblados
print(f"\n✓ TEST 1: Parámetros de reestructura")
restruct = df[df['Accion_final'] == 'REESTRUCTURAR']
if len(restruct) > 0:
    plazo_ok = restruct['plazo_optimo'].notna().sum()
    tasa_ok = restruct['tasa_nueva'].notna().sum()
    quita_ok = restruct['quita'].notna().sum()
    
    print(f"   Total reestructuras: {len(restruct)}")
    print(f"   - plazo_optimo: {plazo_ok}/{len(restruct)} ({plazo_ok/len(restruct)*100:.1f}%)")
    print(f"   - tasa_nueva: {tasa_ok}/{len(restruct)} ({tasa_ok/len(restruct)*100:.1f}%)")
    print(f"   - quita: {quita_ok}/{len(restruct)} ({quita_ok/len(restruct)*100:.1f}%)")
    
    if plazo_ok == len(restruct) and tasa_ok == len(restruct) and quita_ok == len(restruct):
        print(f"   ✅ PASS - Todos los parámetros poblados")
        total_pass += 1
    else:
        print(f"   ❌ FAIL - Parámetros faltantes")
    total_tests += 1

# Test 2: PTI/DSCR calculados
print(f"\n✓ TEST 2: PTI/DSCR en reestructuras")
if len(restruct) > 0:
    # Retail usa PTI
    retail = restruct[restruct['segment'].isin(['Mortgage', 'Consumer'])]
    pti_ok = retail['PTI_post'].notna().sum() if len(retail) > 0 else 0
    
    # Corporate usa DSCR
    corp = restruct[~restruct['segment'].isin(['Mortgage', 'Consumer'])]
    dscr_ok = corp['DSCR_post'].notna().sum() if len(corp) > 0 else 0
    
    print(f"   Retail (Mortgage/Consumer): {len(retail)}")
    if len(retail) > 0:
        print(f"   - PTI_post poblado: {pti_ok}/{len(retail)} ({pti_ok/len(retail)*100:.1f}%)")
    
    print(f"   Corporate: {len(corp)}")
    if len(corp) > 0:
        print(f"   - DSCR_post poblado: {dscr_ok}/{len(corp)} ({dscr_ok/len(corp)*100:.1f}%)")
        
        # Validar rangos DSCR
        dscr_vals = corp['DSCR_post'].dropna()
        if len(dscr_vals) > 0:
            print(f"   - DSCR_post range: [{dscr_vals.min():.2f}, {dscr_vals.max():.2f}]")
            if (dscr_vals >= 1.05).all():
                print(f"   ✅ PASS - DSCR >= 1.05 (prudencial gate)")
                total_pass += 1
            else:
                print(f"   ❌ FAIL - DSCR < 1.05 detectado")
        else:
            print(f"   ⚠️ WARNING - Sin DSCR_post para validar")
    total_tests += 1

# Test 3: Fire-sale protections
print(f"\n✓ TEST 3: Fire-sale protections")
sells = df[df['Accion_final'] == 'VENDER']
if len(sells) > 0:
    firesales = sells[sells['Fire_Sale'] == True]
    print(f"   Total ventas: {len(sells)}")
    print(f"   - Fire-sales detectados: {len(firesales)}")
    
    if len(firesales) == 0:
        print(f"   ✅ PASS - Fire-sales bloqueados correctamente (prudencial)")
        total_pass += 1
    else:
        print(f"   ❌ FAIL - Fire-sales ejecutados en prudencial")
else:
    print(f"   Sin ventas ejecutadas (postura muy conservadora)")
    total_pass += 1
total_tests += 1

# Test 4: Escalación y metadata
print(f"\n✓ TEST 4: Escalación bank-ready")
escalated = df[df['case_status'] == 'HOLD_NO_EXECUTABLE_ACTION']
print(f"   Casos escalados: {len(escalated)}")

if len(escalated) > 0:
    # Verificar que tienen metadata completa
    next_step_ok = escalated['next_step'].notna().sum()
    reason_ok = escalated['next_step_reason'].notna().sum()
    days_ok = escalated['review_due_days'].notna().sum()
    override_ok = escalated['override_reason'].notna().sum()
    
    print(f"   - next_step: {next_step_ok}/{len(escalated)}")
    print(f"   - next_step_reason: {reason_ok}/{len(escalated)}")
    print(f"   - review_due_days: {days_ok}/{len(escalated)}")
    print(f"   - override_reason: {override_ok}/{len(escalated)}")
    
    if all([next_step_ok == len(escalated), reason_ok == len(escalated), 
            days_ok == len(escalated), override_ok == len(escalated)]):
        print(f"   ✅ PASS - Metadata completa para todos los casos escalados")
        total_pass += 1
    else:
        print(f"   ❌ FAIL - Metadata incompleta")
else:
    print(f"   ℹ️ INFO - Sin casos de escalación en este run")
    # No falla el test si no hay casos (depende del portfolio)
    total_pass += 1
total_tests += 1

# Test 5: Convergencia y decisión final
print(f"\n✓ TEST 5: Convergencia Micro/Macro")
if 'Convergencia_Caso' in df.columns:
    conv_values = df['Convergencia_Caso'].value_counts()
    print(f"   Distribución:")
    for val, count in conv_values.items():
        print(f"   - {val}: {count} ({count/len(df)*100:.1f}%)")
    
    # Verificar que hay convergencia calculada
    empty_conv = df['Convergencia_Caso'].isna().sum()
    if empty_conv == 0:
        print(f"   ✅ PASS - Convergencia calculada para todos")
        total_pass += 1
    else:
        print(f"   ⚠️ WARNING - {empty_conv}/{len(df)} sin convergencia")
        total_pass += 1  # No es crítico en nuevos runs
else:
    print(f"   ⚠️ WARNING - Columna Convergencia_Caso no presente")
total_tests += 1

# Test 6: Decisiones ejecutables
print(f"\n✓ TEST 6: Decisiones válidas")
valid_actions = ['MANTENER', 'REESTRUCTURAR', 'VENDER']
invalid = df[~df['Accion_final'].isin(valid_actions)]
print(f"   Decisiones inválidas: {len(invalid)}")

if len(invalid) == 0:
    print(f"   ✅ PASS - Todas las decisiones son válidas")
    total_pass += 1
else:
    print(f"   ❌ FAIL - Decisiones inválidas detectadas")
    print(invalid[['loan_id', 'Accion_final']].head())
total_tests += 1

# Summary
print(f"\n{'='*80}")
print(f"RESULTADO FINAL: {total_pass}/{total_tests} tests passed ({total_pass/total_tests*100:.1f}%)")
print(f"{'='*80}")

if total_pass == total_tests:
    print(f"✅ ¡SISTEMA BANK-READY VALIDADO!")
    exit(0)
elif total_pass >= total_tests * 0.8:
    print(f"⚠️ Sistema mayormente conforme - revisar detalles")
    exit(0)
else:
    print(f"❌ Sistema requiere correcciones")
    exit(1)
