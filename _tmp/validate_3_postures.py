"""Validación comparativa de las 3 posturas bank-ready"""
import pandas as pd
import numpy as np
import glob
import os

print("="*80)
print("VALIDACIÓN COMPARATIVA - 3 POSTURAS BANK-READY")
print("="*80)

# Buscar las inferencias más recientes para cada postura
def find_latest_inference(posture_tag):
    pattern = f"reports/coordinated_inference_*{posture_tag}*"
    folders = glob.glob(pattern)
    if not folders:
        return None
    latest = max(folders, key=os.path.getctime)
    
    # Buscar excel
    excel_candidates = glob.glob(os.path.join(latest, "decisiones_finales_*.xlsx"))
    if excel_candidates:
        return excel_candidates[0]
    return None

# Cargar las 3 posturas
postures = {
    "PRUDENCIAL": find_latest_inference("pru"),
    "BALANCEADO": find_latest_inference("bal"),
    "DESINVERSION": find_latest_inference("des")
}

print(f"\n📂 Archivos encontrados:")
for posture, path in postures.items():
    if path:
        print(f"   ✅ {posture:15}: {os.path.basename(os.path.dirname(path))}")
    else:
        print(f"   ❌ {posture:15}: NO ENCONTRADO")

# Load data
dfs = {}
for posture, path in postures.items():
    if path:
        try:
            dfs[posture] = pd.read_excel(path)
            print(f"   ✓ {posture}: {len(dfs[posture])} préstamos cargados")
        except Exception as e:
            print(f"   ✗ {posture}: Error cargando - {e}")

if len(dfs) == 0:
    print("\n❌ No se pudieron cargar archivos. Verifica que las inferencias hayan completado.")
    exit(1)

print(f"\n{'='*80}")
print("ANÁLISIS COMPARATIVO")
print(f"{'='*80}")

# 1. Mix de decisiones
print(f"\n1️⃣ MIX DE DECISIONES (DIFERENCIACIÓN DE POSTURAS)")
print(f"\n{'Postura':<15} | {'MANTENER':<12} | {'REESTRUCTURAR':<15} | {'VENDER':<12}")
print("-" * 60)

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    if 'Accion_final' not in df.columns:
        print(f"{posture:<15} | ERROR: Columna Accion_final no encontrada")
        continue
    
    mix = df['Accion_final'].value_counts()
    total = len(df)
    
    mantener = mix.get('MANTENER', 0)
    reestructurar = mix.get('REESTRUCTURAR', 0)
    vender = mix.get('VENDER', 0)
    
    print(f"{posture:<15} | {mantener:4} ({mantener/total*100:5.1f}%) | "
          f"{reestructurar:4} ({reestructurar/total*100:5.1f}%) | "
          f"{vender:4} ({vender/total*100:5.1f}%)")

# 2. Validación de criterios bank-ready
print(f"\n2️⃣ CRITERIOS BANK-READY (POR POSTURA)")

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    print(f"\n--- {posture} ---")
    
    # Reestructuras con parámetros
    if 'Accion_final' in df.columns:
        restruct = df[df['Accion_final'] == 'REESTRUCTURAR']
        print(f"   Reestructuras: {len(restruct)}")
        
        if len(restruct) > 0 and 'plazo_optimo' in df.columns:
            plazo_ok = restruct['plazo_optimo'].notna().sum()
            print(f"   - plazo_optimo: {plazo_ok}/{len(restruct)} ({plazo_ok/len(restruct)*100:.1f}%)")
            
            if 'DSCR_post' in df.columns:
                corp = restruct[~restruct['segment'].isin(['Mortgage', 'Consumer'])]
                if len(corp) > 0:
                    dscr_vals = corp['DSCR_post'].dropna()
                    if len(dscr_vals) > 0:
                        print(f"   - DSCR_post (corp): min={dscr_vals.min():.2f}, mean={dscr_vals.mean():.2f}")
    
    # Fire-sales
    if 'Accion_final' in df.columns and 'Fire_Sale' in df.columns:
        sells = df[df['Accion_final'] == 'VENDER']
        if len(sells) > 0:
            firesales = sells[sells['Fire_Sale'] == True]
            print(f"   Ventas: {len(sells)} | Fire-sales ejecutados: {len(firesales)}")
    
    # Escalación
    if 'case_status' in df.columns:
        escalated = df[df['case_status'] == 'HOLD_NO_EXECUTABLE_ACTION']
        print(f"   Casos escalados: {len(escalated)} ({len(escalated)/len(df)*100:.1f}%)")

# 3. Test de divergencia (criterio crítico)
print(f"\n3️⃣ TEST DE DIVERGENCIA (CUMPLIMIENTO)")

# Criterios esperados
criteria = {
    "PRUDENCIAL": {"vender_max": 10, "reestructurar_min": 80},
    "DESINVERSION": {"vender_min": 70}
}

passed = 0
total = 0

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    mix = df['Accion_final'].value_counts()
    total_loans = len(df)
    
    vender_pct = (mix.get('VENDER', 0) / total_loans) * 100
    reestructurar_pct = (mix.get('REESTRUCTURAR', 0) / total_loans) * 100
    
    if posture in criteria:
        crit = criteria[posture]
        
        # Prudencial
        if "vender_max" in crit:
            total += 1
            if vender_pct <= crit["vender_max"]:
                print(f"   ✅ {posture}: VENDER {vender_pct:.1f}% <= {crit['vender_max']}%")
                passed += 1
            else:
                print(f"   ❌ {posture}: VENDER {vender_pct:.1f}% > {crit['vender_max']}% (FALLA)")
        
        if "reestructurar_min" in crit:
            total += 1
            if reestructurar_pct >= crit["reestructurar_min"]:
                print(f"   ✅ {posture}: REESTRUCTURAR {reestructurar_pct:.1f}% >= {crit['reestructurar_min']}%")
                passed += 1
            else:
                print(f"   ❌ {posture}: REESTRUCTURAR {reestructurar_pct:.1f}% < {crit['reestructurar_min']}% (FALLA)")
        
        # Desinversion
        if "vender_min" in crit:
            total += 1
            if vender_pct >= crit["vender_min"]:
                print(f"   ✅ {posture}: VENDER {vender_pct:.1f}% >= {crit['vender_min']}%")
                passed += 1
            else:
                print(f"   ❌ {posture}: VENDER {vender_pct:.1f}% < {crit['vender_min']}% (FALLA)")

# 4. KPIs agregados (si disponibles)
print(f"\n4️⃣ KPIs AGREGADOS (CARTERA)")

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    print(f"\n--- {posture} ---")
    
    if 'EVA_post' in df.columns:
        eva_total = df['EVA_post'].sum()
        print(f"   EVA_post total: {eva_total:,.0f} EUR")
    
    if 'RWA_post' in df.columns:
        rwa_total = df['RWA_post'].sum()
        print(f"   RWA_post total: {rwa_total:,.0f} EUR")
    
    if 'capital_release_realized' in df.columns:
        cap_rel = df['capital_release_realized'].sum()
        print(f"   Capital liberado: {cap_rel:,.0f} EUR")

# Summary
print(f"\n{'='*80}")
print(f"RESULTADO FINAL: {passed}/{total} criterios de divergencia PASS ({passed/total*100 if total > 0 else 0:.1f}%)")
print(f"{'='*80}")

if passed == total and total > 0:
    print(f"\n✅ ¡POSTURAS DIFERENCIADAS Y VALIDADAS!")
    exit(0)
elif passed >= total * 0.67:
    print(f"\n⚠️ Posturas mayormente diferenciadas - revisar detalles")
    exit(0)
else:
    print(f"\n❌ Posturas requieren ajustes para diferenciación")
    exit(1)
