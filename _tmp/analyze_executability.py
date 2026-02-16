"""Análisis de ejecutabilidad - Evidencia de diferenciación de posturas"""
import pandas as pd
import numpy as np
import glob

print("="*80)
print("ANÁLISIS DE EJECUTABILIDAD - RECOMENDACIÓN vs EJECUCIÓN")
print("="*80)

# Buscar últimas inferencias para cada postura (específicamente las de executability_v1)
def find_latest_inference(posture_tag):
    pattern = f"reports/coordinated_inference_executability_v1_{posture_tag}*"
    folders = glob.glob(pattern)
    if not folders:
        return None
    latest = max(folders, key=lambda x: x)
    excel_files = glob.glob(f"{latest}/decisiones_finales_*.xlsx")
    return excel_files[0] if excel_files else None

# Cargar las 3 posturas
posturas = {
    "PRUDENCIAL": find_latest_inference("pru"),
    "BALANCEADO": find_latest_inference("bal"),
    "DESINVERSION": find_latest_inference("des")
}

dfs = {}
for posture, path in posturas.items():
    if path:
        try:
            dfs[posture] = pd.read_excel(path)
            print(f"[OK] {posture}: {len(dfs[posture])} prestamos cargados desde {path}")
        except Exception as e:
            print(f"[ERROR] {posture}: Error - {e}")

if len(dfs) == 0:
    print("\n[ERROR] No se encontraron archivos. Ejecuta las inferencias primero.")
    exit(1)

print(f"\n{'='*80}")
print("[SECTION 1] MIX DE ACCIONES (DIFERENCIACION DE POSTURAS)")
print(f"{'='*80}\n")

# Tabla comparativa
print(f"{'Postura':<15} | {'MANTENER':<12} | {'REESTRUCTURAR':<15} | {'VENDER':<12}")
print("-" * 60)

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    mix = df["Accion_final"].value_counts()
    total = len(df)
    
    mantener = mix.get("MANTENER", 0)
    reestructurar = mix.get("REESTRUCTURAR", 0)
    vender = mix.get("VENDER", 0)
    
    print(f"{posture:<15} | {mantener:4} ({mantener/total*100:5.1f}%) | "
          f"{reestructurar:4} ({reestructurar/total*100:5.1f}%) | "
          f"{vender:4} ({vender/total*100:5.1f}%)")

print(f"\n{'='*80}")
print("[SECTION 2] MANDATOS Y EJECUTABILIDAD")
print(f"{'='*80}\n")

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    print(f"--- {posture} ---")
    
    if "sale_mandate" in df.columns:
        n_mandate = df["sale_mandate"].sum()
        print(f"   Mandatos de venta: {n_mandate} ({n_mandate/len(df)*100:.1f}%)")
    
    if "sale_executable" in df.columns:
        n_sale_exec = df["sale_executable"].sum()
        n_insulting = df.get("sale_insulting_flag", pd.Series(False)).sum()
        print(f"   Ventas ejecutables: {n_sale_exec} ({n_sale_exec/len(df)*100:.1f}%)")
        print(f"   Precios insultantes: {n_insulting} ({n_insulting/len(df)*100:.1f}%)")
    
    if "restruct_executable" in df.columns:
        n_rest_exec = df["restruct_executable"].sum()
        print(f"   Reestructuras ejecutables: {n_rest_exec} ({n_rest_exec/len(df)*100:.1f}%)")
    
    if "acceptance_score" in df.columns:
        acc_mean = df["acceptance_score"].mean()
        acc_min = df["acceptance_score"].min()
        acc_max = df["acceptance_score"].max()
        print(f"   Acceptance score: mean={acc_mean:.1f}, range=[{acc_min:.0f}, {acc_max:.0f}]")
    
    if "execution_status" in df.columns:
        status_dist = df["execution_status"].value_counts()
        print(f"   Status de ejecución:")
        for status, count in status_dist.items():
            print(f"      {status}: {count} ({count/len(df)*100:.1f}%)")
    
    print()

print(f"{'='*80}")
print("[SECTION 3] CASOS FRONTERA (RECOMENDACION != EJECUCION)")
print(f"{'='*80}\n")

# Para cada postura, mostrar 10 casos donde recommended_action != final_action
for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    print(f"--- {posture} ---\n")
    
    if "recommended_action" not in df.columns or "Accion_final" not in df.columns:
        print("   [WARNING] Columnas de recomendacion/ejecucion no disponibles\n")
        continue
    
    # Filtrar casos donde hay cambio
    changed = df[df["recommended_action"] != df["Accion_final"]].copy()
    
    if len(changed) == 0:
        print("   ℹ️ No hay casos donde recomendación difiera de ejecución\n")
        continue
    
    print(f"   Total casos con cambio: {len(changed)}/{len(df)} ({len(changed)/len(df)*100:.1f}%)\n")
    
    # Mostrar primeros 10
    sample = changed.head(10)
    
    cols_to_show = [
        "loan_id", "segment", "EAD", 
        "recommended_action", "Accion_final",
        "sale_mandate", "sale_executable", "sale_insulting_flag",
        "restruct_executable", "acceptance_score",
        "execution_status", "case_status",
        "next_step", "override_reason"
    ]
    
    # Filtrar solo las que existen
    cols_available = [c for c in cols_to_show if c in sample.columns]
    
    for idx, row in sample.iterrows():
        print(f"   Préstamo: {row.get('loan_id', idx)}")
        print(f"      Segmento: {row.get('segment', 'N/A')}, EAD: {row.get('EAD', 0):,.0f}")
        print(f"      RECOMENDACIÓN: {row.get('recommended_action', 'N/A')}")
        print(f"      EJECUCIÓN: {row.get('Accion_final', 'N/A')}")
        
        if row.get("sale_mandate", False):
            print(f"      [MANDATO DE VENTA]: {row.get('sale_mandate_reason', 'N/A')[:60]}")
        
        if "sale_executable" in row:
            print(f"      Venta ejecutable: {row.get('sale_executable', False)}")
            if row.get("sale_insulting_flag", False):
                print(f"         └─ Precio insultante (< {row.get('valor_referencia', 0)*0.4:,.0f})")
        
        if "restruct_executable" in row:
            print(f"      Reestructura ejecutable: {row.get('restruct_executable', False)}")
            if "acceptance_score" in row:
                print(f"         └─ Acceptance score: {row.get('acceptance_score', 0):.0f}")
        
        if "execution_status" in row:
            print(f"      Status: {row.get('execution_status', 'N/A')}")
        
        if "override_reason" in row and pd.notna(row.get("override_reason")) and row.get("override_reason") != "":
            print(f"      Razón: {row.get('override_reason', 'N/A')[:80]}")
        
        print()
    
    if len(changed) > 10:
        print(f"   ... y {len(changed)-10} casos más con cambio\n")
    print()

print(f"{'='*80}")
print("[SECTION 4] KPIS AGREGADOS (IMPACTO DE EJECUTABILIDAD)")
print(f"{'='*80}\n")

for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
    if posture not in dfs:
        continue
    
    df = dfs[posture]
    print(f"--- {posture} ---")
    
    if "EVA_post" in df.columns:
        eva_total = df["EVA_post"].sum()
        print(f"   EVA_post total: {eva_total:,.0f} EUR")
    
    if "capital_release_realized" in df.columns:
        cap_rel = df["capital_release_realized"].sum()
        print(f"   Capital liberado: {cap_rel:,.0f} EUR")
    
    if "RWA_post" in df.columns:
        rwa_total = df["RWA_post"].sum()
        print(f"   RWA_post total: {rwa_total:,.0f} EUR")
    
    # Ventas con mandato vs sin mandato
    if "sale_mandate" in df.columns and "Accion_final" in df.columns:
        ventas = df[df["Accion_final"] == "VENDER"]
        if len(ventas) > 0:
            ventas_mandato = ventas[ventas["sale_mandate"] == True]
            ventas_voluntarias = ventas[ventas["sale_mandate"] == False]
            print(f"   Ventas ejecutadas: {len(ventas)}")
            print(f"      - Con mandato: {len(ventas_mandato)}")
            print(f"      - Voluntarias: {len(ventas_voluntarias)}")
    
    print()

print(f"{'='*80}")
print("[SECTION 5] TESTS DE MONOTONICIDAD (PRUDENCIAL < BALANCEADO < DESINVERSION)")
print(f"{'='*80}\n")

if len(dfs) == 3:
    # Test 1: Ventas
    sells = {}
    for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
        mix = dfs[posture]["Accion_final"].value_counts()
        sells[posture] = mix.get("VENDER", 0) / len(dfs[posture]) * 100
    
    print(f"% VENTAS:")
    print(f"   PRUDENCIAL:   {sells['PRUDENCIAL']:.1f}%")
    print(f"   BALANCEADO:   {sells['BALANCEADO']:.1f}%")
    print(f"   DESINVERSION: {sells['DESINVERSION']:.1f}%")
    
    if sells["PRUDENCIAL"] <= sells["BALANCEADO"] <= sells["DESINVERSION"]:
        print("   [PASS] Monotonicidad en ventas (PRUD <= BAL <= DESINV)")
    else:
        print("   [WARNING] Monotonicidad en ventas NO cumplida")
    
    # Test 2: Mantener (inversa)
    holds = {}
    for posture in ["PRUDENCIAL", "BALANCEADO", "DESINVERSION"]:
        mix = dfs[posture]["Accion_final"].value_counts()
        holds[posture] = mix.get("MANTENER", 0) / len(dfs[posture]) * 100
    
    print(f"\n% MANTENER (inversa esperada):")
    print(f"   PRUDENCIAL:   {holds['PRUDENCIAL']:.1f}%")
    print(f"   BALANCEADO:   {holds['BALANCEADO']:.1f}%")
    print(f"   DESINVERSION: {holds['DESINVERSION']:.1f}%")
    
    if holds["PRUDENCIAL"] >= holds["BALANCEADO"] >= holds["DESINVERSION"]:
        print("   [PASS] Monotonicidad inversa en MANTENER (PRUD >= BAL >= DESINV)")
    else:
        print("   [WARNING] Monotonicidad inversa en MANTENER NO cumplida")
    
    # Test 3: Diferenciación mínima
    diff_pru_bal = abs(sells["BALANCEADO"] - sells["PRUDENCIAL"])
    diff_bal_des = abs(sells["DESINVERSION"] - sells["BALANCEADO"])
    
    print(f"\nDIFERENCIACIÓN (en ventas):")
    print(f"   BAL vs PRUD: {diff_pru_bal:.1f}pp")
    print(f"   DESINV vs BAL: {diff_bal_des:.1f}pp")
    
    if diff_pru_bal >= 5 and diff_bal_des >= 10:
        print("   [PASS] Diferenciacion significativa (BAL-PRUD>=5pp, DESINV-BAL>=10pp)")
    else:
        print("   [WARNING] Diferenciacion insuficiente")

print(f"\n{'='*80}")
print("CONCLUSIÓN:")
print("="*80)
print("Este análisis demuestra la separación entre RECOMENDACIÓN y EJECUCIÓN.")
print("Las decisiones finales reflejan:")
print("  - Mandatos de capital (ventas obligatorias con salvaguardas)")
print("  - Gates de ejecutabilidad (precios no insultantes, loss caps, acceptance scores)")
print("  - Fallbacks coherentes (si recomendacion no ejecutable -> alternativa viable)")
print("  - Diferenciación real entre posturas (knobs parametrizados audiblemente)")
print("="*80)
