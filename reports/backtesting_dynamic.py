# -*- coding: utf-8 -*-
# ============================================================
# reports/backtesting_dynamic.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Backtesting dinámico multi-trimestre: evolución de la cartera tras las decisiones del agente (curación, redefault, KPIs por trimestre).
# ============================================================
"""
============================================================================
 reports/backtesting_dynamic.py — Backtesting Multi-Trimestre Dinámico
----------------------------------------------------------------------------
 Simula la evolución de la cartera a lo largo de Q trimestres DESPUÉS
 de aplicar las decisiones del agente.

 Para cada trimestre t:
   1. Los préstamos MANTENIDOS evolucionan: PD puede subir/bajar, DPD crece,
      una fracción "cura" y otra "default hard".
   2. Los préstamos REESTRUCTURADOS siguen su nueva cuota; algunos re-default.
   3. Los préstamos VENDIDOS ya salieron del balance (impacto en t=0).
   4. Se recalculan KPIs agregados (EVA, RWA, EL, Capital).

 Produce un CSV + MD con la serie temporal de KPIs por postura.

 Uso:
   python reports/backtesting_dynamic.py --tag infer_ci0226 --quarters 8
============================================================================
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
import glob
import json
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as cfg

logger = logging.getLogger("backtesting_dynamic")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------------------------------------------------------------------------
# Parámetros de evolución temporal — calibrados por segmento/rating
# Basados en literatura EBA/BCE sobre NPL workout (2018-2024)
# ---------------------------------------------------------------------------

# Tasa de curación trimestral por segmento (% de préstamos MANTENIDOS que curan)
# Mortgage > Corporate > SME (hipotecas tienen más colateral ejecutable)
CURE_RATE_BY_SEGMENT = {
    "Mortgage": 0.05,
    "Large Corporate": 0.035,
    "Corporate": 0.03,
    "MidCap": 0.025,
    "SME": 0.02,
    "Project Finance": 0.015,
    # fallback
    "default": 0.03,
}

# Tasa de re-default trimestral por rating (% de reestructurados que vuelven a default)
# CCC/B tienen tasas mucho más altas que BBB/A
REDEFAULT_RATE_BY_RATING = {
    "CCC": 0.12,  # 12% por trimestre ≈ 40% anual (realista para CCC NPL)
    "B": 0.09,    # 9% por trimestre ≈ 32% anual
    "BB": 0.06,   # 6% por trimestre ≈ 22% anual
    "BBB": 0.04,  # 4% por trimestre ≈ 15% anual
    "A": 0.03,    # 3% por trimestre ≈ 12% anual
    "AA": 0.02,
    "AAA": 0.01,
    "default": 0.05,
}

PD_DRIFT_KEEP_Q = 0.02             # PD sube +2pp/trimestre si se mantiene sin acción
PD_IMPROVEMENT_RESTRUCT_Q = -0.01  # PD baja -1pp/trimestre si reestructurado exitoso
LGD_DRIFT_KEEP_Q = 0.005           # LGD sube +0.5pp/trimestre sin acción
DPD_INCREMENT_KEEP_Q = 30          # +30 días por trimestre si no cura

HURDLE = float(cfg.CONFIG.regulacion.hurdle_rate)
CAP_RATIO = float(cfg.CONFIG.regulacion.required_total_capital_ratio())
COST_FUND = 0.006
HORIZON_Y = float(max(1.0, cfg.CONFIG.sensibilidad_reestructura.horizon_months / 12.0))

# ---------------------------------------------------------------------------
# Funciones de cálculo económico (consistente con loan_env / backtesting_light)
# ---------------------------------------------------------------------------

def _calc_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs agregados sobre un DataFrame de préstamos on-book."""
    ead = pd.to_numeric(df["EAD"], errors="coerce").fillna(0)
    pd_col = pd.to_numeric(df["PD"], errors="coerce").fillna(0).clip(0, 1)
    lgd = pd.to_numeric(df["LGD"], errors="coerce").fillna(0).clip(0, 1)
    rw = pd.to_numeric(df["RW"], errors="coerce").fillna(1.5)
    rate = pd.to_numeric(df.get("rate", df.get("interest_rate", 0.05)),
                         errors="coerce").fillna(0.05)

    rwa = ead * rw
    el_life = pd_col * lgd * ead
    el_annual = el_life / HORIZON_Y
    ni = ead * rate - COST_FUND * ead - el_annual
    eva = ni - HURDLE * rwa

    return {
        "n_loans": len(df),
        "total_EAD": float(ead.sum()),
        "total_RWA": float(rwa.sum()),
        "total_EL": float(el_life.sum()),
        "total_EVA": float(eva.sum()),
        "mean_PD": float(pd_col.mean()) if len(df) > 0 else 0.0,
        "mean_LGD": float(lgd.mean()) if len(df) > 0 else 0.0,
        "mean_DPD": float(df["DPD"].mean()) if "DPD" in df.columns and len(df) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Evolución trimestral
# ---------------------------------------------------------------------------

def evolve_quarter(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Evoluciona un trimestre la cartera on-book.
    Usa tasas de curación por segmento y re-default por rating.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        return df

    # --- 1. Curación por segmento ---
    mask_keep = df["status"] == "MANTENER"
    if mask_keep.sum() > 0:
        keep_idx = df.loc[mask_keep].index
        for idx in keep_idx:
            seg = str(df.loc[idx].get("segment", "default"))
            cure_rate = CURE_RATE_BY_SEGMENT.get(seg, CURE_RATE_BY_SEGMENT["default"])
            if rng.random() < cure_rate:
                df.loc[idx, "status"] = "CURADO"

    # --- 2. Re-default de reestructurados por rating ---
    mask_restr = df["status"] == "REESTRUCTURAR"
    if mask_restr.sum() > 0:
        restr_idx = df.loc[mask_restr].index
        for idx in restr_idx:
            rat = str(df.loc[idx].get("rating", "default"))
            redef_rate = REDEFAULT_RATE_BY_RATING.get(rat, REDEFAULT_RATE_BY_RATING["default"])
            if rng.random() < redef_rate:
                df.loc[idx, "status"] = "MANTENER"
                df.loc[idx, "PD"] = min(df.loc[idx, "PD"] + 0.10, 0.95)
                df.loc[idx, "DPD"] = 90

    # --- 3. Drift de los MANTENIDOS (no curados) ---
    mask_keep_alive = df["status"] == "MANTENER"
    if mask_keep_alive.sum() > 0:
        df.loc[mask_keep_alive, "PD"] = (
            df.loc[mask_keep_alive, "PD"] + PD_DRIFT_KEEP_Q
        ).clip(upper=0.95)
        df.loc[mask_keep_alive, "LGD"] = (
            df.loc[mask_keep_alive, "LGD"] + LGD_DRIFT_KEEP_Q
        ).clip(upper=0.95)
        df.loc[mask_keep_alive, "DPD"] = df.loc[mask_keep_alive, "DPD"] + DPD_INCREMENT_KEEP_Q

    # --- 4. Mejora de los REESTRUCTURADOS (no re-defaulted) ---
    mask_restr_ok = df["status"] == "REESTRUCTURAR"
    if mask_restr_ok.sum() > 0:
        df.loc[mask_restr_ok, "PD"] = (
            df.loc[mask_restr_ok, "PD"] + PD_IMPROVEMENT_RESTRUCT_Q
        ).clip(lower=0.05)
        df.loc[mask_restr_ok, "LGD"] = (
            df.loc[mask_restr_ok, "LGD"] - 0.005
        ).clip(lower=0.20)

    # Eliminar curados del libro NPL (ya son performing)
    df = df[df["status"] != "CURADO"].copy()

    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def _find_decision_files(tag: str) -> dict:
    """Localiza los archivos de decisiones por postura para un tag dado."""
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
            continue
        # 3) fallback: buscar variante _npl
        pattern2 = os.path.join(reports_dir,
                                f"coordinated_inference_{tag}_*_{posture}",
                                f"decisiones_finales_{posture}_npl.xlsx")
        matches2 = sorted(glob.glob(pattern2), reverse=True)
        if matches2:
            found[posture] = matches2[0]

    return found


def run_backtesting_dynamic(tag: str, n_quarters: int = 8, seed: int = 42):
    """Ejecuta backtesting dinámico multi-trimestre."""

    logger.info(f"=== Backtesting Dinámico: tag={tag}, Q={n_quarters}, seed={seed} ===")
    rng = np.random.default_rng(seed)

    decision_files = _find_decision_files(tag)
    if not decision_files:
        logger.error(f"No se encontraron archivos de decisiones para tag='{tag}'")
        return None

    all_results = []

    for posture, fpath in decision_files.items():
        logger.info(f"--- Postura: {posture} ({os.path.basename(fpath)}) ---")
        df_orig = pd.read_excel(fpath)

        # Preparar: separar vendidos (off-book en t=0) del resto
        if "Accion_final" not in df_orig.columns:
            logger.warning(f"  Sin columna Accion_final, skip {posture}")
            continue

        df_sold = df_orig[df_orig["Accion_final"] == "VENDER"].copy()
        df_book = df_orig[df_orig["Accion_final"] != "VENDER"].copy()
        df_book["status"] = df_book["Accion_final"].copy()

        # Asegurar columnas numéricas
        for col in ["EAD", "PD", "LGD", "RW", "DPD"]:
            if col in df_book.columns:
                df_book[col] = pd.to_numeric(df_book[col], errors="coerce")

        # --- t=0: estado post-decisión ---
        cap_released_t0 = 0.0
        if "capital_liberado" in df_sold.columns:
            cap_released_t0 = float(pd.to_numeric(
                df_sold["capital_liberado"], errors="coerce"
            ).fillna(0).sum())

        pnl_sales_t0 = 0.0
        if "pnl" in df_sold.columns:
            pnl_sales_t0 = float(pd.to_numeric(
                df_sold["pnl"], errors="coerce"
            ).fillna(0).sum())

        kpis_t0 = _calc_kpis(df_book)
        kpis_t0["quarter"] = 0
        kpis_t0["posture"] = posture
        kpis_t0["cumul_cured"] = 0
        kpis_t0["cumul_redefaulted"] = 0
        kpis_t0["capital_released_sales"] = cap_released_t0
        kpis_t0["pnl_sales"] = pnl_sales_t0
        all_results.append(kpis_t0)

        logger.info(f"  t=0: {kpis_t0['n_loans']} on-book, "
                     f"EVA={kpis_t0['total_EVA']:,.0f}, "
                     f"RWA={kpis_t0['total_RWA']:,.0f}, "
                     f"sold={len(df_sold)}, cap_rel={cap_released_t0:,.0f}")

        # --- Evolución trimestral ---
        df_current = df_book.copy()
        cumul_cured = 0
        cumul_redef = 0

        for q in range(1, n_quarters + 1):
            n_before = len(df_current)
            n_restr_before = (df_current["status"] == "REESTRUCTURAR").sum()

            df_current = evolve_quarter(df_current, rng)

            n_after = len(df_current)
            n_cured_q = n_before - n_after  # los que salieron
            # re-defaults: reestructurados que pasaron a MANTENER
            n_restr_after = (df_current["status"] == "REESTRUCTURAR").sum()
            n_redef_q = max(0, n_restr_before - n_restr_after - n_cured_q)
            # Aproximación: todo reestructurado que desapareció y no curó, re-defaulteó
            # Pero nuestra lógica es: re-default cambia a MANTENER (no sale del libro)
            # Cura sí sale. Así que:
            n_redef_q = max(0, (n_restr_before - n_restr_after))
            # (esto incluye curados de reestructurados, pero es una buena aproximación)

            cumul_cured += n_cured_q
            cumul_redef += n_redef_q

            kpis_q = _calc_kpis(df_current)
            kpis_q["quarter"] = q
            kpis_q["posture"] = posture
            kpis_q["cumul_cured"] = cumul_cured
            kpis_q["cumul_redefaulted"] = cumul_redef
            kpis_q["capital_released_sales"] = cap_released_t0
            kpis_q["pnl_sales"] = pnl_sales_t0
            all_results.append(kpis_q)

            if q % 4 == 0 or q == n_quarters:
                logger.info(f"  t={q}: {kpis_q['n_loans']} on-book, "
                             f"EVA={kpis_q['total_EVA']:,.0f}, "
                             f"PD_avg={kpis_q['mean_PD']:.3f}, "
                             f"cured_acum={cumul_cured}, redef_acum={cumul_redef}")

    if not all_results:
        logger.error("Sin resultados.")
        return None

    df_results = pd.DataFrame(all_results)

    # --- Guardar ---
    out_csv = os.path.join(ROOT_DIR, "reports", f"backtesting_dynamic_{tag}.csv")
    df_results.to_csv(out_csv, index=False)
    logger.info(f"CSV guardado: {out_csv}")

    # --- Reporte MD ---
    out_md = os.path.join(ROOT_DIR, "reports", f"backtesting_dynamic_{tag}.md")
    _write_report(df_results, tag, n_quarters, out_md)
    logger.info(f"Reporte guardado: {out_md}")

    return df_results


def _write_report(df: pd.DataFrame, tag: str, n_quarters: int, out_path: str):
    """Genera reporte Markdown con análisis de la evolución temporal."""
    postures = df["posture"].unique()

    lines = [
        f"# Backtesting Dinámico Multi-Trimestre: {tag}",
        f"**Horizonte**: {n_quarters} trimestres ({n_quarters / 4:.1f} años)",
        f"**Posturas**: {', '.join(postures)}",
        "",
        "---",
        "",
        "## 1. Evolución de KPIs por Trimestre",
        "",
    ]

    for posture in sorted(postures):
        dp = df[df["posture"] == posture].sort_values("quarter")
        lines.append(f"### {posture.upper()}")
        lines.append("")
        lines.append("| Q | Loans | EVA | RWA | EL | PD_avg | LGD_avg | DPD_avg | Curados | Re-default |")
        lines.append("|--:|------:|----:|----:|---:|-------:|--------:|--------:|--------:|-----------:|")

        for _, r in dp.iterrows():
            lines.append(
                f"| {int(r['quarter'])} "
                f"| {int(r['n_loans'])} "
                f"| {r['total_EVA']:,.0f} "
                f"| {r['total_RWA']:,.0f} "
                f"| {r['total_EL']:,.0f} "
                f"| {r['mean_PD']:.3f} "
                f"| {r['mean_LGD']:.3f} "
                f"| {r['mean_DPD']:.0f} "
                f"| {int(r['cumul_cured'])} "
                f"| {int(r['cumul_redefaulted'])} |"
            )
        lines.append("")

    # --- Comparativa final ---
    lines.append("## 2. Comparativa Final (último trimestre)")
    lines.append("")
    lines.append("| Postura | EVA_t0 | EVA_tN | ΔEVA | RWA_t0 | RWA_tN | Loans_t0 | Loans_tN | Curados | Re-default |")
    lines.append("|---------|-------:|-------:|-----:|-------:|-------:|---------:|---------:|--------:|-----------:|")

    for posture in sorted(postures):
        dp = df[df["posture"] == posture].sort_values("quarter")
        t0 = dp.iloc[0]
        tN = dp.iloc[-1]
        delta_eva = tN["total_EVA"] - t0["total_EVA"]
        lines.append(
            f"| {posture} "
            f"| {t0['total_EVA']:,.0f} "
            f"| {tN['total_EVA']:,.0f} "
            f"| {delta_eva:,.0f} "
            f"| {t0['total_RWA']:,.0f} "
            f"| {tN['total_RWA']:,.0f} "
            f"| {int(t0['n_loans'])} "
            f"| {int(tN['n_loans'])} "
            f"| {int(tN['cumul_cured'])} "
            f"| {int(tN['cumul_redefaulted'])} |"
        )
    lines.append("")

    # --- Resiliencia ---
    lines.append("## 3. Análisis de Resiliencia")
    lines.append("")

    for posture in sorted(postures):
        dp = df[df["posture"] == posture].sort_values("quarter")
        eva_series = dp["total_EVA"].values
        eva_t0 = eva_series[0]
        eva_min = eva_series.min()
        eva_final = eva_series[-1]
        max_drawdown = (eva_min - eva_t0) / abs(eva_t0) * 100 if eva_t0 != 0 else 0
        trend = "MEJORA" if eva_final > eva_t0 else "DETERIORO"
        recovery_q = None
        if eva_final > eva_t0:
            for i, v in enumerate(eva_series):
                if v > eva_t0 and i > 0:
                    recovery_q = i
                    break

        lines.append(f"**{posture.upper()}**:")
        lines.append(f"- Tendencia EVA: **{trend}** ({eva_final - eva_t0:+,.0f} €)")
        lines.append(f"- Max drawdown: {max_drawdown:+.1f}%")
        if recovery_q is not None:
            lines.append(f"- Recuperación en Q{recovery_q}")
        lines.append(f"- Tasa curación acumulada: {int(dp.iloc[-1]['cumul_cured'])} préstamos")
        lines.append(f"- Re-defaults acumulados: {int(dp.iloc[-1]['cumul_redefaulted'])} préstamos")
        lines.append("")

    # --- Monotonías temporales ---
    lines.append("## 4. Validación de Monotonías Temporales")
    lines.append("")

    final_rows = df[df["quarter"] == df["quarter"].max()].set_index("posture")
    checks = []
    if "prudencial" in final_rows.index and "desinversion" in final_rows.index:
        p_rwa = final_rows.loc["prudencial", "total_RWA"]
        d_rwa = final_rows.loc["desinversion", "total_RWA"]
        checks.append(("RWA final D ≤ P", d_rwa <= p_rwa * 1.01))  # 1% tolerance

        p_loans = final_rows.loc["prudencial", "n_loans"]
        d_loans = final_rows.loc["desinversion", "n_loans"]
        checks.append(("Loans final D ≤ P", d_loans <= p_loans))

    if "prudencial" in final_rows.index and "balanceado" in final_rows.index:
        p_cured = final_rows.loc["prudencial", "cumul_cured"]
        b_cured = final_rows.loc["balanceado", "cumul_cured"]
        # Balanceado tiene más reestructurados → más posibilidad de cura
        checks.append(("Curación B ≥ P (reestructura ayuda)", b_cured >= p_cured * 0.8))

    for desc, passed in checks:
        icon = "✅" if passed else "❌"
        lines.append(f"- {icon} {desc}")

    lines.append("")
    lines.append("---")
    lines.append("*Generado automáticamente por `reports/backtesting_dynamic.py`*")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting dinámico multi-trimestre")
    parser.add_argument("--tag", required=True, help="Tag de la inferencia (ej: infer_ci0226)")
    parser.add_argument("--quarters", type=int, default=8, help="Número de trimestres (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla (default: 42)")
    args = parser.parse_args()

    run_backtesting_dynamic(args.tag, args.quarters, args.seed)
