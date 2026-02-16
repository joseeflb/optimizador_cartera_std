#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_calibration_evidence.py

Analiza decisiones de 3 posturas recalibradas y genera evidencia para comité:
- Tabla de mixes (% MANTENER / REESTRUCTURAR / VENDER)
- KPIs agregados (EVA, capital liberado, RWA)
- Conteos de mandatos vs ventas voluntarias
- 10 casos frontera PRUD ≠ BAL con WHY completo

USO:
    python _tmp/analyze_calibration_evidence.py --pru reports/decisiones_*_pru*.xlsx --bal reports/decisiones_*_bal*.xlsx --des reports/decisiones_*_des*.xlsx
"""
import os
import sys
import argparse
from glob import glob
from datetime import datetime

import pandas as pd
import numpy as np


def load_latest_file(pattern: str) -> str:
    """Encuentra el archivo más reciente que coincida con el patrón."""
    files = glob(pattern)
    if not files:
        raise FileNotFoundError(f"No se encontró archivo: {pattern}")
    # Ordenar por fecha de modificación (más reciente primero)
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def analyze_mix(df: pd.DataFrame, posture: str) -> dict:
    """Analiza mix de decisiones por postura."""
    total = len(df)
    
    accion = df["Accion_final"].str.upper().str.strip()
    count_mantener = (accion == "MANTENER").sum()
    count_restruct = (accion == "REESTRUCTURAR").sum()
    count_vender = (accion == "VENDER").sum()
    
    return {
        "postura": posture,
        "total": total,
        "MANTENER": count_mantener,
        "REESTRUCTURAR": count_restruct,
        "VENDER": count_vender,
        "pct_MANTENER": count_mantener / total * 100 if total > 0 else 0,
        "pct_REESTRUCTURAR": count_restruct / total * 100 if total > 0 else 0,
        "pct_VENDER": count_vender / total * 100 if total > 0 else 0,
    }


def analyze_kpis(df: pd.DataFrame, posture: str) -> dict:
    """Analiza KPIs agregados por postura."""
    eva_post = pd.to_numeric(df.get("EVA_post") if "EVA_post" in df.columns else pd.Series([0]*len(df)), errors="coerce").sum()
    eva_delta = pd.to_numeric(df.get("EVA_delta") if "EVA_delta" in df.columns else pd.Series([0]*len(df)), errors="coerce").sum()
    rwa_post = pd.to_numeric(df.get("RWA_post") if "RWA_post" in df.columns else pd.Series([0]*len(df)), errors="coerce").sum()
    capital_release = pd.to_numeric(df.get("capital_release_realized") if "capital_release_realized" in df.columns else pd.Series([0]*len(df)), errors="coerce").sum()
    
    return {
        "postura": posture,
        "EVA_post_total": eva_post,
        "EVA_delta_total": eva_delta,
        "RWA_post_total": rwa_post,
        "capital_release_total": capital_release,
    }


def analyze_mandates(df: pd.DataFrame, posture: str) -> dict:
    """Analiza mandatos de venta vs ventas voluntarias."""
    accion = df["Accion_final"].str.upper().str.strip()
    ventas_mask = (accion == "VENDER")
    
    mandate_col = "sale_mandate" if "sale_mandate" in df.columns else None
    
    if mandate_col:
        n_ventas_mandate = (ventas_mask & df[mandate_col]).sum()
        n_ventas_voluntary = (ventas_mask & ~df[mandate_col]).sum()
        n_mandate_total = df[mandate_col].sum()
        n_mandate_blocked = (df[mandate_col] & ~ventas_mask).sum()
    else:
        n_ventas_mandate = 0
        n_ventas_voluntary = ventas_mask.sum()
        n_mandate_total = 0
        n_mandate_blocked = 0
    
    return {
        "postura": posture,
        "ventas_por_mandato": n_ventas_mandate,
        "ventas_voluntarias": n_ventas_voluntary,
        "mandatos_totales": n_mandate_total,
        "mandatos_bloqueados": n_mandate_blocked,
        "pct_mandato_sobre_cartera": n_mandate_total / len(df) * 100 if len(df) > 0 else 0,
    }


def find_frontier_cases(df_pru: pd.DataFrame, df_bal: pd.DataFrame, n_cases: int = 10) -> pd.DataFrame:
    """
    Encuentra casos frontera donde PRUD ≠ BAL (decisiones divergentes).
    Prioriza casos con EVA alto o capital alto (casos importantes).
    """
    # Merge por loan_id
    if "loan_id" not in df_pru.columns or "loan_id" not in df_bal.columns:
        print("⚠️ WARNING: loan_id no encontrado, no se pueden comparar casos frontera")
        return pd.DataFrame()
    
    df_merge = df_pru.merge(
        df_bal,
        on="loan_id",
        suffixes=("_pru", "_bal"),
        how="inner"
    )
    
    # Filtrar casos divergentes (PRUD ≠ BAL)
    accion_pru = df_merge["Accion_final_pru"].str.upper().str.strip()
    accion_bal = df_merge["Accion_final_bal"].str.upper().str.strip()
    
    divergent = df_merge[accion_pru != accion_bal].copy()
    
    if len(divergent) == 0:
        print("⚠️ WARNING: No hay casos divergentes PRUD ≠ BAL")
        return pd.DataFrame()
    
    # Ordenar por importancia (EVA_post alto o capital_release alto)
    eva_pru = pd.to_numeric(divergent.get("EVA_post_pru", 0), errors="coerce").fillna(0)
    eva_bal = pd.to_numeric(divergent.get("EVA_post_bal", 0), errors="coerce").fillna(0)
    cap_pru = pd.to_numeric(divergent.get("capital_release_realized_pru", 0), errors="coerce").fillna(0)
    cap_bal = pd.to_numeric(divergent.get("capital_release_realized_bal", 0), errors="coerce").fillna(0)
    
    divergent["importance_score"] = abs(eva_pru) + abs(eva_bal) + abs(cap_pru) + abs(cap_bal)
    divergent = divergent.sort_values("importance_score", ascending=False)
    
    # Top N casos
    frontier = divergent.head(n_cases).copy()
    
    # Columnas relevantes para análisis
    cols_keep = [
        "loan_id", "segmento_pru", "EAD_pru", "PD_pru", "LGD_pru",
        "Accion_final_pru", "Reason_Code_pru", "EVA_post_pru", "capital_release_realized_pru",
        "Accion_final_bal", "Reason_Code_bal", "EVA_post_bal", "capital_release_realized_bal",
    ]
    cols_available = [c for c in cols_keep if c in frontier.columns]
    
    return frontier[cols_available]


def main():
    parser = argparse.ArgumentParser(
        description="Analizar calibración de posturas (evidencia para comité)"
    )
    parser.add_argument("--pru", required=True, help="Patrón para XLSX PRUDENCIAL (glob pattern)")
    parser.add_argument("--bal", required=True, help="Patrón para XLSX BALANCEADO (glob pattern)")
    parser.add_argument("--des", required=True, help="Patrón para XLSX DESINVERSION (glob pattern)")
    parser.add_argument("--output-dir", default="reports/calibration", help="Directorio de salida")
    parser.add_argument("--sheet", default="decisiones", help="Nombre de la hoja Excel")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📊 ANÁLISIS DE CALIBRACIÓN (EVIDENCIA PARA COMITÉ)")
    print("=" * 70)
    
    # Cargar archivos más recientes
    print("\n[1/5] Cargando decisiones...")
    file_pru = load_latest_file(args.pru)
    file_bal = load_latest_file(args.bal)
    file_des = load_latest_file(args.des)
    
    print(f"   PRUDENCIAL:  {os.path.basename(file_pru)}")
    print(f"   BALANCEADO:  {os.path.basename(file_bal)}")
    print(f"   DESINVERSION: {os.path.basename(file_des)}")
    
    df_pru = pd.read_excel(file_pru, sheet_name=args.sheet)
    df_bal = pd.read_excel(file_bal, sheet_name=args.sheet)
    df_des = pd.read_excel(file_des, sheet_name=args.sheet)
    
    # Análisis de mixes
    print("\n[2/5] Analizando mixes de decisiones...")
    mix_pru = analyze_mix(df_pru, "PRUDENCIAL")
    mix_bal = analyze_mix(df_bal, "BALANCEADO")
    mix_des = analyze_mix(df_des, "DESINVERSION")
    
    df_mixes = pd.DataFrame([mix_pru, mix_bal, mix_des])
    
    # Análisis de KPIs
    print("[3/5] Analizando KPIs agregados...")
    kpi_pru = analyze_kpis(df_pru, "PRUDENCIAL")
    kpi_bal = analyze_kpis(df_bal, "BALANCEADO")
    kpi_des = analyze_kpis(df_des, "DESINVERSION")
    
    df_kpis = pd.DataFrame([kpi_pru, kpi_bal, kpi_des])
    
    # Análisis de mandatos
    print("[4/5] Analizando mandatos de venta...")
    mandate_pru = analyze_mandates(df_pru, "PRUDENCIAL")
    mandate_bal = analyze_mandates(df_bal, "BALANCEADO")
    mandate_des = analyze_mandates(df_des, "DESINVERSION")
    
    df_mandates = pd.DataFrame([mandate_pru, mandate_bal, mandate_des])
    
    # Casos frontera PRUD ≠ BAL
    print("[5/5] Identificando casos frontera PRUD ≠ BAL...")
    df_frontier = find_frontier_cases(df_pru, df_bal, n_cases=10)
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Exportar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_mixes = os.path.join(args.output_dir, f"calibration_mixes_{timestamp}.csv")
    output_kpis = os.path.join(args.output_dir, f"calibration_kpis_{timestamp}.csv")
    output_mandates = os.path.join(args.output_dir, f"calibration_mandates_{timestamp}.csv")
    output_frontier = os.path.join(args.output_dir, f"calibration_frontier_cases_{timestamp}.csv")
    
    df_mixes.to_csv(output_mixes, index=False, encoding="utf-8-sig")
    df_kpis.to_csv(output_kpis, index=False, encoding="utf-8-sig")
    df_mandates.to_csv(output_mandates, index=False, encoding="utf-8-sig")
    
    if not df_frontier.empty:
        df_frontier.to_csv(output_frontier, index=False, encoding="utf-8-sig")
    
    # Imprimir resultados
    print("\n" + "=" * 70)
    print("📋 TABLA 1: MIX DE DECISIONES")
    print("=" * 70)
    print(df_mixes.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("📋 TABLA 2: KPIs AGREGADOS (EUR)")
    print("=" * 70)
    print(df_kpis.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    
    print("\n" + "=" * 70)
    print("📋 TABLA 3: MANDATOS DE VENTA")
    print("=" * 70)
    print(df_mandates.to_string(index=False))
    
    if not df_frontier.empty:
        print("\n" + "=" * 70)
        print("📋 TABLA 4: CASOS FRONTERA (PRUD ≠ BAL, Top 10)")
        print("=" * 70)
        print(df_frontier.to_string(index=False, max_colwidth=30))
    
    # Validación de calibración
    print("\n" + "=" * 70)
    print("✅ VALIDACIÓN DE CALIBRACIÓN")
    print("=" * 70)
    
    # Check 1: DESINV mandatos ~20-30%
    mandate_desinv_pct = mandate_des["pct_mandato_sobre_cartera"]
    if 15 <= mandate_desinv_pct <= 35:
        print(f"✅ DESINV mandatos: {mandate_desinv_pct:.1f}% (target 20-30%)")
    else:
        print(f"⚠️ DESINV mandatos: {mandate_desinv_pct:.1f}% (fuera de rango 20-30%)")
    
    # Check 2: PRUD más conservador que BAL (%mantener)
    if mix_pru["pct_MANTENER"] > mix_bal["pct_MANTENER"]:
        print(f"✅ PRUD más conservador: {mix_pru['pct_MANTENER']:.1f}% mantener vs {mix_bal['pct_MANTENER']:.1f}% BAL")
    else:
        print(f"⚠️ PRUD NO más conservador: {mix_pru['pct_MANTENER']:.1f}% mantener vs {mix_bal['pct_MANTENER']:.1f}% BAL")
    
    # Check 3: BAL más ventas que PRUD (cuando hay ventas)
    if mix_bal["pct_VENDER"] >= mix_pru["pct_VENDER"]:
        print(f"✅ BAL más ejecutivo: {mix_bal['pct_VENDER']:.1f}% ventas vs {mix_pru['pct_VENDER']:.1f}% PRUD")
    else:
        print(f"⚠️ BAL NO más ejecutivo: {mix_bal['pct_VENDER']:.1f}% ventas vs {mix_pru['pct_VENDER']:.1f}% PRUD")
    
    # Check 4: Casos divergentes PRUD ≠ BAL
    if not df_frontier.empty:
        print(f"✅ Encontrados {len(df_frontier)} casos frontera PRUD ≠ BAL (diferenciación clara)")
    else:
        print(f"⚠️ No hay casos frontera PRUD ≠ BAL (posturas idénticas)")
    
    print("\n" + "=" * 70)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"📁 Output dir: {args.output_dir}")
    print(f"   - {os.path.basename(output_mixes)}")
    print(f"   - {os.path.basename(output_kpis)}")
    print(f"   - {os.path.basename(output_mandates)}")
    if not df_frontier.empty:
        print(f"   - {os.path.basename(output_frontier)}")


if __name__ == "__main__":
    main()
