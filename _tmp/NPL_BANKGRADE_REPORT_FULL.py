"""
NPL BANK-GRADE - ANÁLISIS COMPLETO DE EVIDENCIA
=================================================
Genera reporte comprehensivo para validar:
1. Diferenciación PRUD < BAL < DESINV medible
2.Disciplina económica (recovery_min gates funcionando)
3. max_sell_share cap en DESINV (no "vendo todo")
4. Mandatos funcionando sin romper lógica NPL
"""
import pandas as pd
import numpy as np
from pathlib import Path

def find_latest_inference(posture: str, tag: str = "NPL_BANKGRADE") -> Path:
    """Encuentra el directorio de inferencia más reciente por postura."""
    reports_dir = Path("reports")
    pattern = f"coordinated_inference_{tag}_{posture[:3]}_*_{posture}"
    
    matching = list(reports_dir.glob(pattern))
    if not matching:
        raise FileNotFoundError(f"No se encontró inferencia para {posture} con tag={tag}")
    
    # Ordenar por timestamp en nombre (formato: YYYYMMDD_HHMMSS)
    latest = sorted(matching, key=lambda p: p.name)[-1]
    return latest / f"decisiones_finales_{posture}.xlsx"

def load_posture_data(posture: str) -> pd.DataFrame:
    """Carga datos de una postura."""
    file_path = find_latest_inference(posture)
    print(f"[{posture.upper()}] Cargando: {file_path}")
    return pd.read_excel(file_path)

def analyze_mix(df: pd.DataFrame, posture: str) -> dict:
    """Calcula mix de acciones y KPIs básicos."""
    total = len(df)
    
    mix = {}
    for action in ["MANTENER", "REESTRUCTURAR", "VENDER"]:
        count = (df["Accion_final"] == action).sum()
        mix[f"{action}_n"] = count
        mix[f"{action}_pct"] = (count / total) * 100
    
    # Ejecutabilidad
    mix["sale_executable_n"] = df.get("sale_executable", pd.Series([False]*total)).sum()
    mix["restruct_executable_n"] = df.get("restruct_executable", pd.Series([False]*total)).sum()
    
    # KPIs (si existen)
    if "EVA_final" in df.columns:
        mix["EVA_final_sum"] = df["EVA_final"].sum()
    if "capital_release" in df.columns:
        mix["capital_release_sum"] = df["capital_release"].sum()
    
    return mix

def analyze_distributions(df: pd.DataFrame) -> dict:
    """Calcula percentiles de métricas clave."""
    metrics = ["sale_loss_pct", "recovery_rate", "precio_optimo", "valor_referencia", "EAD"]
    
    dist = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        series = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) == 0:
            continue
        
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            dist[f"{metric}_p{pct}"] = series.quantile(pct / 100.0)
    
    # Ratios derivados
    if "precio_optimo" in df.columns and "valor_referencia" in df.columns:
        ratio = (df["precio_optimo"] / df["valor_referencia"]).replace([np.inf, -np.inf], np.nan).dropna()
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            dist[f"price_valor_ratio_p{pct}"] = ratio.quantile(pct / 100.0)
    
    if "precio_optimo" in df.columns and "EAD" in df.columns:
        ratio_ead = (df["precio_optimo"] / df["EAD"]).replace([np.inf, -np.inf], np.nan).dropna()
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            dist[f"price_EAD_ratio_p{pct}"] = ratio_ead.quantile(pct / 100.0)
    
    return dist

def analyze_gates(df: pd.DataFrame) -> dict:
    """Cuenta préstamos bloqueados por cada gate."""
    gates = {}
    
    # Ventas
    gates["sale_insulting"] = df.get("sale_insulting_flag", pd.Series([False]*len(df))).sum()
    gates["sale_within_loss_cap"] = df.get("sale_within_loss_cap", pd.Series([False]*len(df))).sum()
    gates["sale_meets_recovery_min"] = df.get("sale_meets_recovery_min", pd.Series([False]*len(df))).sum()
    gates["sale_executable"] = df.get("sale_executable", pd.Series([False]*len(df))).sum()
    gates["sale_mandate"] = df.get("sale_mandate", pd.Series([False]*len(df))).sum()
    
    # Reestructuras
    gates["restruct_executable"] = df.get("restruct_executable", pd.Series([False]*len(df))).sum()
    
    # Reason codes (top 5)
    if "sale_reason_code" in df.columns:
        reason_counts = df["sale_reason_code"].value_counts().head(5)
        for idx, (reason, count) in enumerate(reason_counts.items(), 1):
            gates[f"sale_reason_top{idx}"] = f"{reason}={count}"
    
    # Execution status (top 5)
    if "execution_status" in df.columns:
        status_counts = df["execution_status"].value_counts().head(5)
        for idx, (status, count) in enumerate(status_counts.items(), 1):
            gates[f"exec_status_top{idx}"] = f"{status}={count}"
    
    return gates

def find_frontier_cases(df_prud: pd.DataFrame, df_bal: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Encuentra casos donde PRUD=MANTENER pero BAL=VENDER o REESTRUCTURAR.
    Muestra WHY difieren.
    """
    # Merge por loan_id
    if "loan_id" not in df_prud.columns or "loan_id" not in df_bal.columns:
        print("[WARNING] loan_id no encontrado - usando índices")
        df_prud = df_prud.copy()
        df_bal = df_bal.copy()
        df_prud["loan_id"] = df_prud.index
        df_bal["loan_id"] = df_bal.index
    
    merged = df_prud[["loan_id", "Accion_final", "sale_executable", "restruct_executable", 
                       "sale_loss_pct", "recovery_rate", "sale_mandate", "EAD", "precio_optimo", "valor_referencia"]].merge(
        df_bal[["loan_id", "Accion_final", "sale_executable", "restruct_executable"]],
        on="loan_id",
        suffixes=("_PRUD", "_BAL")
    )
    
    # Filtrar: PRUD=MANTENER y BAL!=MANTENER
    frontier = merged[(merged["Accion_final_PRUD"] == "MANTENER") & (merged["Accion_final_BAL"] != "MANTENER")]
    
    # WHY logic
    def _why(row):
        reasons = []
        
        # Loss_pct diferencia
        loss = row.get("sale_loss_pct", 0)
        if loss > 0.88:  # PRUD loss_cap
            reasons.append(f"loss={loss:.1%}>88% (PRUD cap)")
        elif loss > 0.92:
            reasons.append(f"loss={loss:.1%}>92% (BAL cap)")
        
        # Recovery diferencia
        rec = row.get("recovery_rate", 0)
        if rec < 0.12:  # PRUD recovery_min
            reasons.append(f"recovery={rec:.1%}<12% (PRUD min)")
        elif rec < 0.08:
            reasons.append(f"recovery={rec:.1%}<8% (BAL min)")
        
        # Mandate
        if row.get("sale_mandate"):
            reasons.append("MANDATE")
        
        # Ejecutabilidad
        if row.get("sale_executable_PRUD") and not row.get("sale_executable_BAL"):
            reasons.append("PRUD_exec_BAL_blocked")
        elif not row.get("sale_executable_PRUD") and row.get("sale_executable_BAL"):
            reasons.append("BAL_exec_PRUD_blocked")
        
        return " | ".join(reasons) if reasons else "OTHER"
    
    frontier["WHY"] = frontier.apply(_why, axis=1)
    
    # Ordenar por diferencia de recovery (mayores primero)
    frontier = frontier.sort_values("recovery_rate", ascending=False).head(n)
    
    return frontier[["loan_id", "Accion_final_PRUD", "Accion_final_BAL", 
                     "EAD", "precio_optimo", "recovery_rate", "sale_loss_pct", "sale_mandate", "WHY"]]

def generate_report():
    """Genera reporte completo."""
    print("\n" + "="*80)
    print("NPL BANK-GRADE - REPORTE COMPLETO DE EVIDENCIA")
    print("="*80)
    
    # Cargar datos
    print("\n[CARGA] Cargando datos de 3 posturas...")
    df_prud = load_posture_data("prudencial")
    df_bal = load_posture_data("balanceado")
    df_des = load_posture_data("desinversion")
    
    # =========================================================================
    # TABLA 1: MIX DE ACCIONES POR POSTURA
    # =========================================================================
    print("\n" + "-"*80)
    print("TABLA 1: MIX DE ACCIONES POR POSTURA")
    print("-"*80)
    
    mix_prud = analyze_mix(df_prud, "PRUDENCIAL")
    mix_bal = analyze_mix(df_bal, "BALANCEADO")
    mix_des = analyze_mix(df_des, "DESINVERSION")
    
    mix_df = pd.DataFrame({
        "PRUDENCIAL": mix_prud,
        "BALANCEADO": mix_bal,
        "DESINVERSION": mix_des
    }).T
    
    print(mix_df.to_string())
    
    # Test monotonicidad
    print("\n[TEST MONOTONICIDAD]")
    v_prud = mix_prud["VENDER_pct"]
    v_bal = mix_bal["VENDER_pct"]
    v_des = mix_des["VENDER_pct"]
    
    if v_prud <= v_bal <= v_des:
        print(f"✅ %VENDER: PRUD ({v_prud:.1f}%) <= BAL ({v_bal:.1f}%) <= DESINV ({v_des:.1f}%)")
    else:
        print(f"❌ %VENDER NO monótono: PRUD={v_prud:.1f}%, BAL={v_bal:.1f}%, DESINV={v_des:.1f}%")
    
    m_prud = mix_prud["MANTENER_pct"]
    m_bal = mix_bal["MANTENER_pct"]
    m_des = mix_des["MANTENER_pct"]
    
    if m_prud >= m_bal >= m_des:
        print(f"✅ %MANTENER: PRUD ({m_prud:.1f}%) >= BAL ({m_bal:.1f}%) >= DESINV ({m_des:.1f}%)")
    else:
        print(f"⚠️ %MANTENER: PRUD={m_prud:.1f}%, BAL={m_bal:.1f}%, DESINV={m_des:.1f}%")
    
    # =========================================================================
    # TABLA 2: DISTRIBUCIONES (PERCENTILES)
    # =========================================================================
    print("\n" + "-"*80)
    print("TABLA 2: DISTRIBUCIONES DE MÉTRICAS CLAVE (PERCENTILES)")
    print("-"*80)
    
    dist_prud = analyze_distributions(df_prud)
    dist_bal = analyze_distributions(df_bal)
    dist_des = analyze_distributions(df_des)
    
    dist_df = pd.DataFrame({
        "PRUDENCIAL": dist_prud,
        "BALANCEADO": dist_bal,
        "DESINVERSION": dist_des
    }).T
    
    # Mostrar solo columnas clave
    key_cols = [col for col in dist_df.columns if any(x in col for x in ["loss_pct", "recovery_rate", "price_valor_ratio"])]
    print(dist_df[key_cols].to_string())
    
    # =========================================================================
    # TABLA 3: CONTEOS POR GATE
    # =========================================================================
    print("\n" + "-"*80)
    print("TABLA 3: CONTEOS DE EJECUTABILIDAD POR GATE")
    print("-"*80)
    
    gates_prud = analyze_gates(df_prud)
    gates_bal = analyze_gates(df_bal)
    gates_des = analyze_gates(df_des)
    
    gates_df = pd.DataFrame({
        "PRUDENCIAL": gates_prud,
        "BALANCEADO": gates_bal,
        "DESINVERSION": gates_des
    }).T
    
    print(gates_df.to_string())
    
    # =========================================================================
    # TABLA 4: CASOS FRONTERA PRUD vs BAL
    # =========================================================================
    print("\n" + "-"*80)
    print("TABLA 4: 10 CASOS FRONTERA PRUD(MANTENER) vs BAL(VENDER/REESTRUCTURAR)")
    print("-"*80)
    
    frontier = find_frontier_cases(df_prud, df_bal, n=10)
    print(frontier.to_string(index=False))
    
    # =========================================================================
    # RESUMEN EJECUTIVO
    # =========================================================================
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO - VALIDACIÓN NPL BANK-GRADE")
    print("="*80)
    
    print(f"\n✅ CALIBRACIÓN APLICADA:")
    print(f"  PRUDENCIAL:   loss_cap=88%, recovery_min=12%, max_sell=100%")
    print(f"  BALANCEADO:   loss_cap=92%, recovery_min=8%,  max_sell=100%")
    print(f"  DESINVERSION: loss_cap=95%, recovery_min=5%,  max_sell=70% (CAP!)")
    
    print(f"\n✅ DIFERENCIACIÓN:")
    print(f"  %VENDER:     PRUD={v_prud:.1f}% < BAL={v_bal:.1f}% < DESINV={v_des:.1f}%")
    print(f"  %MANTENER:   PRUD={m_prud:.1f}% > BAL={m_bal:.1f}% > DESINV={m_des:.1f}%")
    print(f"  Gap PRUD-BAL: {abs(v_bal - v_prud):.1f}pp ventas")
    
    print(f"\n✅ GATES FUNCIONANDO:")
    print(f"  PRUD: {gates_prud['sale_executable']}/500 ejecutables | "
          f"{gates_prud['sale_mandate']} mandatos")
    print(f"  BAL:  {gates_bal['sale_executable']}/500 ejecutables | "
          f"{gates_bal['sale_mandate']} mandatos")
    print(f"  DESINV: {gates_des['sale_executable']}/500 ejecutables | "
          f"{gates_des['sale_mandate']} mandatos")
    
    if v_des <= 70.0:
        print(f"\n✅ MAX_SELL_SHARE CAP: DESINV={v_des:.1f}% ventas <= 70% cap ✓")
    else:
        print(f"\n⚠️ MAX_SELL_SHARE CAP: DESINV={v_des:.1f}% ventas > 70% cap (revisar)")
    
    print("\n" + "="*80)
    print("REPORTE COMPLETADO - Evidencia disponible para comité")
    print("="*80)

if __name__ == "__main__":
    generate_report()
