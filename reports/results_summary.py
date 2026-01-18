# -*- coding: utf-8 -*-
# ============================================
# reports/results_summary.py — Consolidación y resumen ejecutivo (v4.3.1 bank-safe)
# ============================================
"""
Consolidado RUN-LEVEL:
- Consolida summary.csv recursivamente
- No depende de loan_id (eso es LOAN-LEVEL)
- Exporta CSV/JSON/Excel (styler summary-level)
- No falla si no existe baseline
- Gráficos opcionales
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
SUMMARY_DIR = os.path.join(REPORTS_DIR, "summary")
CHARTS_DIR = os.path.join(SUMMARY_DIR, "charts")
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(SUMMARY_DIR, "results_summary.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("results_summary")

# -----------------------------------------------------------
# Import RUN-LEVEL styler (no loan_id)
# -----------------------------------------------------------
def _load_summary_styler():
    """
    Carga el exporter estilizado de summary (RUN-level).
    - Caso normal: ejecución desde ROOT (main.py) => import reports.xxx
    - Caso alterno: ejecución desde reports/ => import local
    Devuelve callable o None.
    """
    # 1) Modo paquete
    try:
        from reports.export_styled_excel_summary import export_styled_excel_summary  # type: ignore
        return export_styled_excel_summary
    except Exception:
        pass

    # 2) Modo script dentro de reports/
    try:
        from export_styled_excel_summary import export_styled_excel_summary  # type: ignore
        return export_styled_excel_summary
    except Exception as e:
        logger.warning(f"⚠️ No se pudo importar export_styled_excel_summary (se usará fallback básico): {e}")
        return None


EXPORT_STYLED_SUMMARY = _load_summary_styler()

# -----------------------------------------------------------
# Utilidades
# -----------------------------------------------------------
def ratio_change(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, (a - b) / np.abs(b) * 100.0)


def _iter_candidate_source_dirs(primary: str) -> Iterable[str]:
    cand: List[str] = []

    def add(p: str):
        if p and os.path.isdir(p) and p not in cand:
            cand.append(p)

    # 1) Directorio que pasa el usuario
    add(primary)

    # 2) Estructura estándar
    add(REPORTS_DIR)
    add(SUMMARY_DIR)

    # 3) Candidatos típicos
    for name in ["outputs", "output", "runs", "run", "inference", "inferences", "out", "results"]:
        add(os.path.join(ROOT_DIR, name))
        add(os.path.join(REPORTS_DIR, name))

    return cand


def find_summary_files(source: str) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(source):
        for f in files:
            if f.lower() == "summary.csv":
                matches.append(os.path.join(root, f))
    return matches


def find_summary_files_with_fallback(source_dir: str) -> List[str]:
    searched: List[str] = []
    all_matches: List[str] = []

    for cand in _iter_candidate_source_dirs(source_dir):
        searched.append(cand)
        matches = find_summary_files(cand)
        if matches:
            all_matches.extend(matches)

    # dedup manteniendo orden
    seen = set()
    uniq: List[str] = []
    for f in all_matches:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    if not uniq:
        logger.warning("⚠️ No se encontró ningún summary.csv. Directorios inspeccionados:")
        for d in searched:
            logger.warning(f"   - {d}")

    return uniq


# -----------------------------------------------------------
# CONSOLIDACIÓN
# -----------------------------------------------------------
_STD_RUNLEVEL_COLS = [
    "label",
    "evamean",
    "delta_eva_mean",
    "capital_liberado_mean",
    "pct_keep",
    "pct_restructure",
    "pct_sell",
    "source_file",
    "inference_id",
]


def _empty_runlevel_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_STD_RUNLEVEL_COLS)


def consolidate_summaries(source_dir: str) -> pd.DataFrame:
    files = find_summary_files_with_fallback(source_dir)
    if not files:
        return _empty_runlevel_df()

    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if df is None or df.shape[0] == 0:
                logger.info(f"ℹ️ {f} está vacío. Saltando.")
                continue

            # normaliza columnas por robustez (evita 'label ' vs 'label')
            df.columns = [str(c).strip() for c in df.columns]

            df["source_file"] = f
            df["inference_id"] = os.path.basename(os.path.dirname(f))
            dfs.append(df)
        except Exception as e:
            logger.warning(f"⚠️ Error leyendo {f}: {e}")

    if not dfs:
        return _empty_runlevel_df()

    full = pd.concat(dfs, ignore_index=True)

    # Asegura columnas RUN-LEVEL esperadas
    for col in _STD_RUNLEVEL_COLS:
        if col not in full.columns:
            full[col] = np.nan

    # label fallback (si viene vacío)
    full["label"] = full["label"].fillna(full["inference_id"].astype(str))
    full.loc[full["label"].astype(str).str.strip().eq(""), "label"] = full["inference_id"].astype(str)

    logger.info(f"📥 Consolidado {len(files)} archivos summary.csv. Filas totales: {len(full)}")
    return full


# -----------------------------------------------------------
# GRÁFICOS
# -----------------------------------------------------------
def plot_metric_evolution(df: pd.DataFrame, out_path: str):
    if df.empty:
        logger.info("ℹ️ DF vacío: no se generan gráficos (metric_evolution).")
        return

    metrics = [c for c in ["evamean", "delta_eva_mean", "capital_liberado_mean"] if c in df.columns]
    if not metrics:
        logger.info("ℹ️ No hay métricas para gráfico de evolución.")
        return

    plt.figure(figsize=(9, 5))
    grouped = df.groupby("label")[metrics].mean(numeric_only=True)

    for m in metrics:
        plt.plot(grouped.index, grouped[m], marker="o", label=m)

    plt.title("Evolución media de métricas")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_comparison_bars(df: pd.DataFrame, out_path: str):
    if df.empty:
        logger.info("ℹ️ DF vacío: no se generan gráficos (comparison_bars).")
        return

    cols = [c for c in ["evamean", "delta_eva_mean", "capital_liberado_mean"] if c in df.columns]
    if not cols:
        logger.info("ℹ️ No hay métricas para gráfico de barras.")
        return

    grouped = df.groupby("label")[cols].mean(numeric_only=True)
    ax = grouped.plot(kind="bar", figsize=(8, 4))
    ax.set_title("Comparativa de métricas")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------------------------------------
# RATIOS
# -----------------------------------------------------------
def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "label" not in df.columns:
        logger.info("ℹ️ DF vacío o sin label → skipping ratio calc.")
        return df

    base_mask = df["label"].astype(str).str.contains("baseline", case=False, na=False)
    if not base_mask.any():
        logger.info("ℹ️ No baseline found → skipping ratio calc.")
        return df

    base = df[base_mask].mean(numeric_only=True)

    for col in ["evamean", "delta_eva_mean", "capital_liberado_mean"]:
        if col in df.columns:
            df[f"uplift_{col}_%"] = ratio_change(df[col], base.get(col, np.nan))

    logger.info("📈 Ratios de mejora calculados.")
    return df


# -----------------------------------------------------------
# EXECUTIVE SUMMARY
# -----------------------------------------------------------
def generate_executive_summary(df: pd.DataFrame, out_txt: str):
    lines: List[str] = []
    lines.append("OPTIMIZADOR DE CARTERAS — RESUMEN EJECUTIVO (RUN-LEVEL)")
    lines.append("=" * 80)

    if df.empty:
        lines.append("No hay datos (no se encontraron summary.csv).")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"📝 Executive summary generado (vacío): {out_txt}")
        return

    try:
        base = df[df["label"].astype(str).str.contains("baseline", case=False, na=False)]
        best = df[df["label"].astype(str).str.contains("policy|best|ppo", case=False, na=False)]

        if (not base.empty) and (not best.empty):
            eva_base = float(base["evamean"].mean())
            eva_best = float(best["evamean"].mean())
            cap_base = float(base["capital_liberado_mean"].mean())
            cap_best = float(best["capital_liberado_mean"].mean())

            uplift_eva = (eva_best - eva_base) / abs(eva_base) * 100 if (abs(eva_base) > 1e-12) else np.nan
            uplift_cap = (cap_best - cap_base) / abs(cap_base) * 100 if (abs(cap_base) > 1e-12) else np.nan

            lines.append(f"Mejora EVA media: {uplift_eva:.2f}%" if np.isfinite(uplift_eva) else "Mejora EVA media: n/a")
            lines.append(f"Mejora capital liberado: {uplift_cap:.2f}%" if np.isfinite(uplift_cap) else "Mejora capital liberado: n/a")
        else:
            lines.append("No baseline/best detectados; no se calculan mejoras.")
    except Exception as e:
        lines.append(f"Error computing improvements: {e}")

    lines.append("")
    lines.append("Métricas agregadas (media):")
    for col in ["evamean", "delta_eva_mean", "capital_liberado_mean"]:
        if col in df.columns:
            val = df[col].mean()
            lines.append(f" - {col}: {val:,.2f}" if pd.notna(val) else f" - {col}: n/a")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"📝 Executive summary generado: {out_txt}")


# -----------------------------------------------------------
# EXPORT
# -----------------------------------------------------------
def export_outputs(df: pd.DataFrame, save_excel: bool, save_json: bool):
    # CSV siempre
    csv_path = os.path.join(SUMMARY_DIR, "summary_consolidated.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"💾 CSV consolidado: {csv_path}")

    # Excel (RUN-LEVEL styler)
    if save_excel:
        out_xlsx = os.path.join(SUMMARY_DIR, "summary_consolidated.xlsx")

        # Nunca romper pipeline por formato
        try:
            if df.empty:
                df.to_excel(out_xlsx, index=False)
                logger.info(f"💾 Excel básico (vacío): {out_xlsx}")
            else:
                if EXPORT_STYLED_SUMMARY is not None:
                    EXPORT_STYLED_SUMMARY(df, out_xlsx)
                    logger.info(f"💾 Excel estilizado (RUN-LEVEL): {out_xlsx}")
                else:
                    df.to_excel(out_xlsx, index=False)
                    logger.info(f"💾 Excel básico (fallback): {out_xlsx}")
        except Exception as e:
            logger.warning(f"⚠️ Falló export Excel ({e}). Exportando Excel básico.")
            df.to_excel(out_xlsx, index=False)
            logger.info(f"💾 Excel básico (fallback): {out_xlsx}")

    # JSON
    if save_json:
        out_json = os.path.join(SUMMARY_DIR, "summary_consolidated.json")
        df.to_json(out_json, orient="records", indent=2, force_ascii=False)
        logger.info(f"💾 JSON consolidado: {out_json}")


# -----------------------------------------------------------
# MAIN CLI
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Consolidación RUN-LEVEL (summary.csv) + outputs ejecutivos.")
    p.add_argument("--source", type=str, default=REPORTS_DIR)

    p.add_argument("--charts", dest="charts", action="store_true", default=True)
    p.add_argument("--no-charts", dest="charts", action="store_false")
    p.add_argument("--excel", dest="excel", action="store_true", default=True)
    p.add_argument("--no-excel", dest="excel", action="store_false")
    p.add_argument("--json", dest="json", action="store_true", default=True)
    p.add_argument("--no-json", dest="json", action="store_false")
    p.add_argument("--executive", dest="executive", action="store_true", default=True)
    p.add_argument("--no-executive", dest="executive", action="store_false")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = consolidate_summaries(args.source)
    df = compute_ratios(df)
    export_outputs(df, args.excel, args.json)

    if args.charts:
        plot_metric_evolution(df, os.path.join(CHARTS_DIR, "metric_evolution.png"))
        plot_comparison_bars(df, os.path.join(CHARTS_DIR, "comparison_bars.png"))

    if args.executive:
        generate_executive_summary(df, os.path.join(SUMMARY_DIR, "executive_summary.txt"))

    logger.info("✅ Proceso de consolidación completado.")
