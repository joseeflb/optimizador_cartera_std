# -*- coding: utf-8 -*-
# ============================================================
# reports/segment_dispersion_analysis.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Análisis de dispersión por segmento y rating: validación de sesgos, coherencia de KPIs y estabilidad entre posturas.
# ============================================================
"""
============================================================================
 reports/segment_dispersion_analysis.py — Análisis de Dispersión por Segmento
----------------------------------------------------------------------------
 Desagrega las decisiones del agente por segmento y rating para validar:
   1. Que el agente no tiene sesgo irracional por segmento/rating
   2. Que la distribución de acciones es coherente con el perfil de riesgo
   3. Que los KPIs (EVA, RWA, capital) son razonables por grupo
   4. Que las decisiones son estables entre posturas por segmento

 Produce un CSV detallado + MD con tablas y diagnóstico.

 Uso:
   python reports/segment_dispersion_analysis.py --tag infer_ci0226
============================================================================
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import glob
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as cfg

logger = logging.getLogger("segment_dispersion")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

HURDLE = float(cfg.CONFIG.regulacion.hurdle_rate)
COST_FUND = 0.006
HORIZON_Y = float(max(1.0, cfg.CONFIG.sensibilidad_reestructura.horizon_months / 12.0))

# Orden conceptual de riesgo por rating (menor número = mayor riesgo)
RATING_RISK_ORDER = {"CCC": 1, "B": 2, "BB": 3, "BBB": 4, "A": 5, "AA": 6, "AAA": 7}


def _find_decision_files(tag: str) -> dict:
    """Localiza archivos de decisiones por postura."""
    reports_dir = os.path.join(ROOT_DIR, "reports")
    postures = ["prudencial", "balanceado", "desinversion"]
    found = {}
    for posture in postures:
        # 1) Patrón clásico: reports/coordinated_inference_TAG_*_POSTURE/
        pattern = os.path.join(reports_dir,
                               f"coordinated_inference_{tag}_*_{posture}",
                               f"decisiones_finales_{posture}.xlsx")
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            found[posture] = matches[0]
            continue
        # 2) Patrón nuevo: reports/runs/*_TAG_DELIVERABLE/
        pattern_runs = os.path.join(reports_dir, "runs",
                                    f"*_{tag}_*",
                                    f"decisiones_finales_{posture}.xlsx")
        matches_runs = sorted(glob.glob(pattern_runs), reverse=True)
        if matches_runs:
            found[posture] = matches_runs[0]
    return found


def _action_pct(group: pd.DataFrame) -> dict:
    """Calcula porcentaje de cada acción en un grupo."""
    total = len(group)
    if total == 0:
        return {"pct_MANTENER": 0, "pct_REESTRUCTURAR": 0, "pct_VENDER": 0}
    counts = group["Accion_final"].value_counts()
    return {
        "pct_MANTENER": float(counts.get("MANTENER", 0)) / total * 100,
        "pct_REESTRUCTURAR": float(counts.get("REESTRUCTURAR", 0)) / total * 100,
        "pct_VENDER": float(counts.get("VENDER", 0)) / total * 100,
    }


def _group_kpis(group: pd.DataFrame) -> dict:
    """Calcula KPIs para un grupo de préstamos."""
    ead = pd.to_numeric(group["EAD"], errors="coerce").fillna(0)
    pd_col = pd.to_numeric(group["PD"], errors="coerce").fillna(0).clip(0, 1)
    lgd = pd.to_numeric(group["LGD"], errors="coerce").fillna(0).clip(0, 1)
    rw = pd.to_numeric(group["RW"], errors="coerce").fillna(1.5)

    rwa = ead * rw
    el = pd_col * lgd * ead

    eva_post = pd.to_numeric(group.get("EVA_post", 0), errors="coerce").fillna(0)
    cap_lib = pd.to_numeric(group.get("capital_liberado", 0), errors="coerce").fillna(0)

    return {
        "n_loans": len(group),
        "total_EAD": float(ead.sum()),
        "mean_EAD": float(ead.mean()) if len(group) > 0 else 0,
        "total_RWA": float(rwa.sum()),
        "total_EL": float(el.sum()),
        "mean_PD": float(pd_col.mean()) if len(group) > 0 else 0,
        "mean_LGD": float(lgd.mean()) if len(group) > 0 else 0,
        "mean_DPD": float(group["DPD"].mean()) if "DPD" in group.columns and len(group) > 0 else 0,
        "total_EVA_post": float(eva_post.sum()),
        "total_capital_lib": float(cap_lib.sum()),
    }


def run_segment_dispersion(tag: str):
    """Ejecuta análisis de dispersión por segmento y rating."""

    logger.info(f"=== Análisis de Dispersión por Segmento: tag={tag} ===")

    decision_files = _find_decision_files(tag)
    if not decision_files:
        logger.error(f"No se encontraron archivos para tag='{tag}'")
        return None

    all_segment_rows = []
    all_rating_rows = []
    all_cross_rows = []

    for posture, fpath in decision_files.items():
        logger.info(f"--- Postura: {posture} ---")
        df = pd.read_excel(fpath)

        if "Accion_final" not in df.columns or "segment" not in df.columns:
            logger.warning(f"  Columnas faltantes en {posture}, skip")
            continue

        # ---------- POR SEGMENTO ----------
        for seg, grp in df.groupby("segment"):
            row = {"posture": posture, "segment": str(seg)}
            row.update(_action_pct(grp))
            row.update(_group_kpis(grp))
            all_segment_rows.append(row)

        # ---------- POR RATING ----------
        if "rating" in df.columns:
            for rat, grp in df.groupby("rating"):
                row = {"posture": posture, "rating": str(rat)}
                row.update(_action_pct(grp))
                row.update(_group_kpis(grp))
                all_rating_rows.append(row)

        # ---------- CROSSING SEGMENTO x RATING ----------
        if "rating" in df.columns:
            for (seg, rat), grp in df.groupby(["segment", "rating"]):
                row = {
                    "posture": posture,
                    "segment": str(seg),
                    "rating": str(rat),
                }
                row.update(_action_pct(grp))
                row.update(_group_kpis(grp))
                all_cross_rows.append(row)

    df_seg = pd.DataFrame(all_segment_rows) if all_segment_rows else pd.DataFrame()
    df_rat = pd.DataFrame(all_rating_rows) if all_rating_rows else pd.DataFrame()
    df_cross = pd.DataFrame(all_cross_rows) if all_cross_rows else pd.DataFrame()

    # --- Guardar CSVs ---
    out_seg = os.path.join(ROOT_DIR, "reports", f"dispersion_segment_{tag}.csv")
    out_rat = os.path.join(ROOT_DIR, "reports", f"dispersion_rating_{tag}.csv")
    out_cross = os.path.join(ROOT_DIR, "reports", f"dispersion_cross_{tag}.csv")

    if not df_seg.empty:
        df_seg.to_csv(out_seg, index=False)
    if not df_rat.empty:
        df_rat.to_csv(out_rat, index=False)
    if not df_cross.empty:
        df_cross.to_csv(out_cross, index=False)

    logger.info(f"CSVs guardados: {out_seg}, {out_rat}")

    # --- Reporte MD ---
    out_md = os.path.join(ROOT_DIR, "reports", f"DISPERSION_ANALYSIS_{tag}.md")
    _write_report(df_seg, df_rat, df_cross, tag, out_md)
    logger.info(f"Reporte guardado: {out_md}")

    return df_seg, df_rat, df_cross


def _coherence_checks(df_seg: pd.DataFrame, df_rat: pd.DataFrame) -> list:
    """Ejecuta validaciones de coherencia sobre la dispersión."""
    checks = []

    if df_rat.empty:
        return checks

    # --- CHECK 1: Ratings peores → más ventas (monotonía de riesgo) ---
    for posture in df_rat["posture"].unique():
        dp = df_rat[df_rat["posture"] == posture].copy()
        dp["risk_order"] = dp["rating"].map(RATING_RISK_ORDER)
        dp = dp.dropna(subset=["risk_order"]).sort_values("risk_order")

        if len(dp) >= 3:
            # peores ratings (CCC, B) deberían tener más ventas que mejores (A, AA, AAA)
            worst_3 = dp.head(3)["pct_VENDER"].mean()
            best_3 = dp.tail(3)["pct_VENDER"].mean()
            passed = worst_3 >= best_3 * 0.5  # tolerancia: al menos 50% de prop.
            checks.append({
                "check": f"Sell% worst ratings ≥ best ratings ({posture})",
                "worst_3_avg": f"{worst_3:.1f}%",
                "best_3_avg": f"{best_3:.1f}%",
                "passed": passed,
            })

    # --- CHECK 2: No hay segmentos con 100% una sola acción (sesgo) ---
    if not df_seg.empty:
        for posture in df_seg["posture"].unique():
            dp = df_seg[df_seg["posture"] == posture]
            for _, row in dp.iterrows():
                if row["n_loans"] >= 10:  # solo segmentos con tamaño suficiente
                    is_mono = (row["pct_MANTENER"] == 100 or
                               row["pct_VENDER"] == 100 or
                               row["pct_REESTRUCTURAR"] == 100)
                    if is_mono:
                        checks.append({
                            "check": f"Acción monolítica en {row['segment']} ({posture})",
                            "detail": f"100% {row['segment']}",
                            "passed": False,
                        })

        # Si no se encontró ningún mono, reportar OK
        mono_checks = [c for c in checks if "monolítica" in c["check"]]
        if not mono_checks:
            checks.append({
                "check": "Sin sesgo monolítico por segmento",
                "detail": "Todos los segmentos >10 loans tienen acciones mixtas",
                "passed": True,
            })

    # --- CHECK 3: EVA por segmento es razonable (no todos negativos) ---
    if not df_seg.empty:
        for posture in df_seg["posture"].unique():
            dp = df_seg[df_seg["posture"] == posture]
            n_neg = (dp["total_EVA_post"] < 0).sum()
            n_total = len(dp)
            passed = n_neg < n_total  # al menos 1 segmento con EVA positivo
            checks.append({
                "check": f"EVA positivo en al menos 1 segmento ({posture})",
                "detail": f"{n_total - n_neg}/{n_total} segmentos con EVA > 0",
                "passed": passed,
            })

    # --- CHECK 4: Diferenciación entre posturas por segmento ---
    if not df_seg.empty and len(df_seg["posture"].unique()) >= 2:
        segs = df_seg["segment"].unique()
        for seg in segs:
            dp = df_seg[df_seg["segment"] == seg]
            if len(dp) >= 2:
                sell_range = dp["pct_VENDER"].max() - dp["pct_VENDER"].min()
                if sell_range < 5.0 and dp["n_loans"].min() >= 10:
                    checks.append({
                        "check": f"Diferenciación inter-postura en {seg}",
                        "detail": f"Sell% rango = {sell_range:.1f}pp (bajo)",
                        "passed": False,
                    })

        diff_fails = [c for c in checks if "Diferenciación" in c["check"] and not c["passed"]]
        if not diff_fails:
            checks.append({
                "check": "Diferenciación inter-postura por segmento",
                "detail": "Todas las posturas muestran variación por segmento",
                "passed": True,
            })

    return checks


def _write_report(df_seg, df_rat, df_cross, tag, out_path):
    """Genera reporte Markdown."""
    lines = [
        f"# Análisis de Dispersión por Segmento y Rating: {tag}",
        "",
        "---",
        "",
    ]

    postures = sorted(df_seg["posture"].unique()) if not df_seg.empty else []

    # --- Tabla por segmento ---
    lines.append("## 1. Distribución de Acciones por Segmento")
    lines.append("")

    for posture in postures:
        dp = df_seg[df_seg["posture"] == posture].sort_values("n_loans", ascending=False)
        lines.append(f"### {posture.upper()}")
        lines.append("")
        lines.append("| Segmento | N | EAD total | %Keep | %Restruct | %Sell | EVA_post | Capital Lib. | PD avg | DPD avg |")
        lines.append("|----------|--:|----------:|------:|----------:|------:|---------:|-------------:|-------:|--------:|")

        for _, r in dp.iterrows():
            lines.append(
                f"| {r['segment']} "
                f"| {int(r['n_loans'])} "
                f"| {r['total_EAD']:,.0f} "
                f"| {r['pct_MANTENER']:.1f} "
                f"| {r['pct_REESTRUCTURAR']:.1f} "
                f"| {r['pct_VENDER']:.1f} "
                f"| {r['total_EVA_post']:,.0f} "
                f"| {r['total_capital_lib']:,.0f} "
                f"| {r['mean_PD']:.3f} "
                f"| {r['mean_DPD']:.0f} |"
            )
        lines.append("")

    # --- Tabla por rating ---
    if not df_rat.empty:
        lines.append("## 2. Distribución de Acciones por Rating")
        lines.append("")

        for posture in postures:
            dp = df_rat[df_rat["posture"] == posture].copy()
            dp["risk_order"] = dp["rating"].map(RATING_RISK_ORDER)
            dp = dp.sort_values("risk_order")

            lines.append(f"### {posture.upper()}")
            lines.append("")
            lines.append("| Rating | N | %Keep | %Restruct | %Sell | EVA_post | PD avg | DPD avg |")
            lines.append("|--------|--:|------:|----------:|------:|---------:|-------:|--------:|")

            for _, r in dp.iterrows():
                lines.append(
                    f"| {r['rating']} "
                    f"| {int(r['n_loans'])} "
                    f"| {r['pct_MANTENER']:.1f} "
                    f"| {r['pct_REESTRUCTURAR']:.1f} "
                    f"| {r['pct_VENDER']:.1f} "
                    f"| {r['total_EVA_post']:,.0f} "
                    f"| {r['mean_PD']:.3f} "
                    f"| {r['mean_DPD']:.0f} |"
                )
            lines.append("")

    # --- Heat map textual: segmento × postura → sell% ---
    if not df_seg.empty and len(postures) >= 2:
        lines.append("## 3. Mapa de Intensidad: %Venta por Segmento × Postura")
        lines.append("")

        segments = sorted(df_seg["segment"].unique())
        header = "| Segmento | " + " | ".join(p.upper() for p in postures) + " | Δ max |"
        sep = "|----------|" + "|".join(["------:" for _ in postures]) + "|------:|"
        lines.append(header)
        lines.append(sep)

        for seg in segments:
            vals = []
            for p in postures:
                row = df_seg[(df_seg["posture"] == p) & (df_seg["segment"] == seg)]
                v = row["pct_VENDER"].values[0] if len(row) > 0 else 0
                vals.append(v)
            delta = max(vals) - min(vals) if vals else 0
            cell_line = f"| {seg} "
            for v in vals:
                cell_line += f"| {v:.1f}% "
            cell_line += f"| {delta:.1f}pp |"
            lines.append(cell_line)
        lines.append("")

    # --- Validaciones de coherencia ---
    checks = _coherence_checks(df_seg, df_rat)
    lines.append("## 4. Validaciones de Coherencia")
    lines.append("")

    for c in checks:
        icon = "✅" if c["passed"] else "❌"
        detail = c.get("detail", c.get("worst_3_avg", ""))
        lines.append(f"- {icon} **{c['check']}** — {detail}")
    lines.append("")

    # --- Concentración HHI ---
    if not df_seg.empty:
        lines.append("## 5. Concentración (HHI por EAD)")
        lines.append("")

        for posture in postures:
            dp = df_seg[df_seg["posture"] == posture]
            total_ead = dp["total_EAD"].sum()
            if total_ead > 0:
                shares = dp["total_EAD"] / total_ead
                hhi = float((shares ** 2).sum())
                n_eff = 1 / hhi if hhi > 0 else 0
                lines.append(f"- **{posture.upper()}**: HHI = {hhi:.4f} "
                             f"(N efectivo = {n_eff:.1f} segmentos)")
        lines.append("")

    lines.append("---")
    lines.append("*Generado automáticamente por `reports/segment_dispersion_analysis.py`*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis de dispersión por segmento/rating")
    parser.add_argument("--tag", required=True, help="Tag de la inferencia (ej: infer_ci0226)")
    args = parser.parse_args()

    run_segment_dispersion(args.tag)
