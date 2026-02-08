# -*- coding: utf-8 -*-
# ============================================================
# agent/coordinator_inference.py
# — Inferencia coordinada MICRO+MACRO (decisión única por préstamo)
# (v3.7.0 · macro steering consistente · fire-sale robusto · contrafactuals · schema-stable)
# ============================================================
"""
MEJORAS CLAVE (bank-ready):
- Export estable: enforce_schema() SIEMPRE antes de Excel/CSV.
- Macro asigna SOLO loans “tocados” (sell/restructure). Resto => macro_action="NO_ASIGNADO".
- Arbitraje explícito: si macro_selected y hay conflicto, MACRO gana salvo guardrails DUROS.
- Fire-sale robusto:
    * Enabled por postura (prudencial/balanceado).
    * Triggered si se cumple cualquiera: simulador flag, price/EAD < threshold, pnl/book < umbral, pnl_abs < umbral.
- RC10: Guardrail de viabilidad (PTI/DSCR faltantes) bloquea REESTRUCTURAR.
- Contrafactuals mínimos coherentes:
    * capital_release_cf derivable con RWA_pre o EAD*RW y ratio_total (fallback 10.5%).
    * capital_release_realized / pnl_realized coherentes con Accion_final.
- Merge defensivo con portfolio input (rellena EAD/RW/segment si faltan en micro).
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import shutil

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Rutas básicas
# -----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "coordinator_inference.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("coordinator_inference")

# -----------------------------------------------------------
# Imports del proyecto
# -----------------------------------------------------------
import config as cfg
from agent.policy_inference import InferenceConfig, run_inference
from agent.policy_inference_portfolio import PortfolioInferenceConfig, run_portfolio_inference

# --- Schema enforcement (columnas estables bank-ready) ---
try:
    from reports.schema import enforce_schema  # type: ignore
except Exception:
    def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:  # fallback no-op
        return df


# ===========================================================
# Exporter robusto (evita roturas por path)
# ===========================================================
def _load_exporter():
    try:
        from reports.export_styled_excel import export_styled_excel  # type: ignore
        return export_styled_excel
    except Exception:
        pass
    try:
        from summary.export_styled_excel import export_styled_excel  # type: ignore
        return export_styled_excel
    except Exception:
        pass

    # Fallback mínimo “bank-ready”: OpenPyXL con headers y autofit
    def _fallback_export(df: pd.DataFrame, out_path: str) -> None:
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.styles import Font, PatternFill, Alignment

        wb = Workbook()
        ws = wb.active
        ws.title = "Decisiones"

        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)

        header_fill = PatternFill("solid", fgColor="1F4E79")
        header_font = Font(bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for c in ws[1]:
            c.fill = header_fill
            c.font = header_font
            c.alignment = header_align

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    max_len = max(max_len, len(str(cell.value)))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

        wb.save(out_path)

    logger.warning("⚠️ No se encontró export_styled_excel. Usando fallback OpenPyXL.")
    return _fallback_export


export_styled_excel = _load_exporter()


# ===========================================================
# CONFIG GENERAL DEL COORDINADOR
# ===========================================================
@dataclass
class CoordinatorInferenceConfig:
    model_path_micro: str
    portfolio_path: str

    vecnormalize_path_micro: Optional[str] = None

    model_path_macro: Optional[str] = None
    vecnormalize_path_macro: Optional[str] = None
    vecnormalize_path_loan: Optional[str] = None

    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    tag: str = "coordinated_policy"

    n_steps: int = 1
    top_k: int = 5

    deliverable_only: bool = True
    export_audit_csv: bool = False
    base_output_dir: Optional[str] = None


# ===========================================================
# Guardrails CIB (defaults + override por BankStrategy)
# ===========================================================
DEFAULT_GUARDRAILS = {
    "prudencial": {
        "CAP_RELEASE_CRITICAL": 250_000.0,
        "MIN_PRICE_TO_EAD": 0.12,
        "MAX_FIRE_SALE_PNL_RATIO_BOOK": -0.65,
        "MAX_FIRE_SALE_PNL_ABS": -50_000.0,
        "MIN_DEVA_RESTRUCT": 10_000.0,
        "DSCR_MIN": 1.10,
        "PTI_MAX": 0.35,
    },
    "balanceado": {
        "CAP_RELEASE_CRITICAL": 150_000.0,
        "MIN_PRICE_TO_EAD": 0.10,
        "MAX_FIRE_SALE_PNL_RATIO_BOOK": -0.75,
        "MAX_FIRE_SALE_PNL_ABS": -75_000.0,
        "MIN_DEVA_RESTRUCT": 5_000.0,
        "DSCR_MIN": 1.05,
        "PTI_MAX": 0.45,
    },
    "desinversion": {
        "CAP_RELEASE_CRITICAL": 0.0,
        "MIN_PRICE_TO_EAD": 0.07,
        "MAX_FIRE_SALE_PNL_RATIO_BOOK": -0.90,
        "MAX_FIRE_SALE_PNL_ABS": -250_000.0,
        "MIN_DEVA_RESTRUCT": 15_000.0,
        "DSCR_MIN": 1.00,
        "PTI_MAX": 0.55,
    },
}


def _posture_to_profile(risk_posture: str):
    rp = (risk_posture or "balanceado").lower().strip()
    if rp == "prudencial":
        return cfg.BankProfile.PRUDENTE
    if rp == "desinversion":
        return cfg.BankProfile.DESINVERSION
    return cfg.BankProfile.BALANCEADO


def _get_guardrails(risk_posture: str) -> Dict[str, Any]:
    rp = (risk_posture or "balanceado").lower().strip()
    g = DEFAULT_GUARDRAILS.get(rp, DEFAULT_GUARDRAILS["balanceado"]).copy()

    prof = _posture_to_profile(rp)
    if hasattr(cfg, "set_bank_profile"):
        try:
            cfg.set_bank_profile(prof)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo fijar bank_profile en config: {e}")

    try:
        strat = cfg.BANK_STRATEGIES.get(prof)
        if strat is not None:
            if hasattr(strat, "dscr_min"):
                g["DSCR_MIN"] = float(getattr(strat, "dscr_min"))
            if hasattr(strat, "eva_min_improvement"):
                g["MIN_DEVA_RESTRUCT"] = float(getattr(strat, "eva_min_improvement"))

            pti_candidate = None
            if hasattr(strat, "pti_max") and getattr(strat, "pti_max") is not None:
                pti_candidate = getattr(strat, "pti_max")
            else:
                if rp == "prudencial" and hasattr(strat, "esfuerzo_bajo"):
                    pti_candidate = getattr(strat, "esfuerzo_bajo")
                elif rp == "desinversion" and hasattr(strat, "esfuerzo_alto"):
                    pti_candidate = getattr(strat, "esfuerzo_alto")
                elif hasattr(strat, "esfuerzo_alto") and hasattr(strat, "esfuerzo_bajo"):
                    pti_candidate = 0.5 * (
                        float(getattr(strat, "esfuerzo_alto")) + float(getattr(strat, "esfuerzo_bajo"))
                    )
            if pti_candidate is not None:
                g["PTI_MAX"] = float(pti_candidate)

    except Exception as e:
        logger.warning(f"⚠️ No se pudo derivar guardrails desde BankStrategy: {e}")

    return g


# ===========================================================
# Helpers
# ===========================================================
def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_str(x, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    return str(x)


def _is_missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _fmt_eur(x: float) -> str:
    return f"{x:,.0f}€"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _get_loan_id_from_row(r: pd.Series) -> str:
    for k in ("loan_id", "Loan_ID", "LOAN_ID", "LoanID", "loanId", "id", "ID"):
        if k in r and pd.notna(r.get(k)):
            return str(r.get(k))
    try:
        if r.name is not None and str(r.name).lower() != "nan":
            return str(r.name)
    except Exception:
        pass
    return ""


def _ensure_loan_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        if df is None:
            return df
        if "loan_id" not in df.columns:
            df = df.copy()
            df["loan_id"] = ""
        return df

    df = df.copy()

    if "loan_id" in df.columns and df["loan_id"].notna().any():
        df["loan_id"] = df["loan_id"].astype(str)
        return df

    aliases = ["Loan_ID", "LOAN_ID", "LoanID", "loanId", "id", "ID"]
    for a in aliases:
        if a in df.columns and df[a].notna().any():
            df["loan_id"] = df[a].astype(str)
            return df

    try:
        idx = df.index.astype(str)
        if len(idx) == len(df) and pd.Series(idx).notna().any():
            df["loan_id"] = idx
            return df
    except Exception:
        pass

    df["loan_id"] = df.apply(_get_loan_id_from_row, axis=1).astype(str)
    return df


def _normalize_action(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, (int, np.integer)):
            return {0: "MANTENER", 1: "REESTRUCTURAR", 2: "VENDER"}.get(int(x), "")
        if isinstance(x, float) and not np.isnan(x) and abs(x - int(x)) < 1e-9:
            return {0: "MANTENER", 1: "REESTRUCTURAR", 2: "VENDER"}.get(int(x), "")
    except Exception:
        pass

    s = str(x).strip().upper()
    if not s:
        return ""
    if s in ("KEEP", "MANTENER", "HOLD", "MAINTAIN"):
        return "MANTENER"
    if s in ("RESTRUCT", "RESTRUCTURE", "REESTRUCTURAR", "REESTRUCTURA"):
        return "REESTRUCTURAR"
    if s in ("SELL", "VENDER", "VENTA"):
        return "VENDER"
    if "VEND" in s:
        return "VENDER"
    if "REEST" in s:
        return "REESTRUCTURAR"
    if "MANT" in s:
        return "MANTENER"
    if s in ("NO_ASIGNADO", "NO ASIGNADO", "NOT_ASSIGNED", "NOT ASSIGNED"):
        return "NO_ASIGNADO"
    return s


def _pick_default_macro_model() -> Optional[str]:
    for p in (
        os.path.join(MODELS_DIR, "best_model_portfolio.zip"),
        os.path.join(MODELS_DIR, "best_model_macro.zip"),
    ):
        if os.path.exists(p):
            return p
    return None


def _pick_default_micro_vecnorm() -> Optional[str]:
    for p in (
        os.path.join(MODELS_DIR, "vecnormalize_loan.pkl"),
        os.path.join(MODELS_DIR, "vecnormalize_micro.pkl"),
    ):
        if os.path.exists(p):
            return p

    legacy = os.path.join(MODELS_DIR, "vecnormalize_final.pkl")
    if os.path.exists(legacy):
        logger.warning(
            "⚠️ Usando VN legacy para MICRO: models/vecnormalize_final.pkl. "
            "Recomendado: guardar como vecnormalize_loan.pkl."
        )
        return legacy
    return None


def _pick_default_macro_vecnorm() -> Optional[str]:
    for p in (
        os.path.join(MODELS_DIR, "vecnormalize_portfolio.pkl"),
        os.path.join(MODELS_DIR, "vecnormalize_macro.pkl"),
    ):
        if os.path.exists(p):
            return p
    return None


def _sanitize_vecnorm_paths(
    vn_micro: Optional[str],
    vn_macro: Optional[str],
    vn_loan: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def norm(p: Optional[str]) -> Optional[str]:
        if p and os.path.exists(p):
            return p
        return None

    vn_micro = norm(vn_micro)
    vn_macro = norm(vn_macro)
    vn_loan = norm(vn_loan)

    if vn_macro and vn_micro and os.path.abspath(vn_macro) == os.path.abspath(vn_micro):
        logger.warning("⚠️ VN macro == VN micro. Ignorando vn_macro para evitar mismatch loan vs portfolio.")
        vn_macro = None

    if vn_macro:
        b = os.path.basename(vn_macro).lower()
        if "loan" in b or "micro" in b:
            logger.warning("⚠️ vn_macro parece de LoanEnv por nombre. Ignorando vn_macro para evitar mismatch.")
            vn_macro = None

    if vn_loan is None:
        vn_loan = vn_micro

    return vn_micro, vn_macro, vn_loan


def _get_ratio_total_capital() -> float:
    # Ratio total (Tier1+T2) típico ~10.5% si no existe en config
    cands = [
        ("CONFIG", "regulacion", "capital_total_ratio"),
        ("CONFIG", "regulacion", "total_capital_ratio"),
        ("CONFIG", "regulacion", "ratio_total_capital"),
    ]
    for a, b, c in cands:
        try:
            obj = getattr(cfg, a, None)
            obj = getattr(obj, b, None)
            v = getattr(obj, c, None)
            if v is not None:
                return float(v)
        except Exception:
            pass
    return 0.105


def harmonize_portfolio_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- map monthly_* -> legacy names usados por optimizadores
    if "ingreso_mensual" not in df.columns and "monthly_income" in df.columns:
        df["ingreso_mensual"] = df["monthly_income"]
    if "cashflow_operativo_mensual" not in df.columns and "monthly_cfo" in df.columns:
        df["cashflow_operativo_mensual"] = df["monthly_cfo"]
    if "cuota_mensual" not in df.columns and "monthly_payment" in df.columns:
        df["cuota_mensual"] = df["monthly_payment"]

    # --- PTI / DSCR: preferimos pre si existe; si no, derivamos desde monthly_*
    if "PTI" not in df.columns:
        if "PTI_pre" in df.columns:
            df["PTI"] = df["PTI_pre"]
        elif ("monthly_payment" in df.columns) and ("monthly_income" in df.columns):
            df["PTI"] = df["monthly_payment"] / df["monthly_income"].replace(0, np.nan)

    if "DSCR" not in df.columns:
        if "DSCR_pre" in df.columns:
            df["DSCR"] = df["DSCR_pre"]
        elif ("monthly_cfo" in df.columns) and ("monthly_payment" in df.columns):
            df["DSCR"] = df["monthly_cfo"] / df["monthly_payment"].replace(0, np.nan)

    # --- book_value: si no viene, aproximación consistente
    if "book_value" not in df.columns:
        if "coverage_rate" in df.columns and "EAD" in df.columns:
            cov = pd.to_numeric(df["coverage_rate"], errors="coerce")
            cov = np.where(cov > 1.0, cov / 100.0, cov)
            cov = pd.Series(cov).clip(0, 1)
            df["book_value"] = pd.to_numeric(df["EAD"], errors="coerce") * (1.0 - cov)
        elif "LGD" in df.columns and "EAD" in df.columns:
            lgd = pd.to_numeric(df["LGD"], errors="coerce").clip(0, 1)
            df["book_value"] = pd.to_numeric(df["EAD"], errors="coerce") * (1.0 - lgd)

    # --- normalización mínima de segment (para gates retail vs corp)
    if "segment" in df.columns:
        df["segment"] = df["segment"].astype(str).str.strip()

    return df


def _load_portfolio_df(portfolio_path: str) -> pd.DataFrame:
    if not os.path.exists(portfolio_path):
        return pd.DataFrame()

    ext = os.path.splitext(portfolio_path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            dfp = pd.read_excel(portfolio_path)
        else:
            dfp = pd.read_csv(portfolio_path)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer portfolio para merge defensivo: {e}")
        return pd.DataFrame()

    dfp = harmonize_portfolio_schema(dfp)
    dfp = _ensure_loan_id_column(dfp)
    return dfp


def _fill_from_portfolio(df_final: pd.DataFrame, df_port: pd.DataFrame) -> pd.DataFrame:
    if df_port is None or df_port.empty or df_final is None or df_final.empty:
        return df_final

    df = df_final.copy()
    port = df_port.copy()

    # Normaliza alias comunes en portfolio
    alias_map = {
        "segment": ["segment", "segmento_banco", "segment_raw", "SEGMENT", "Segment"],
        "EAD": ["EAD", "ead", "Ead", "EAD_EUR", "exposure"],
        "RW": ["RW", "rw", "risk_weight", "RW_STD"],
        "PTI_pre": ["PTI_pre", "PTI", "pti"],
        "DSCR_pre": ["DSCR_pre", "DSCR", "dscr"],
    }
    for canon, aliases in alias_map.items():
        if canon in port.columns:
            continue
        for a in aliases:
            if a in port.columns:
                port[canon] = port[a]
                break

    keep_cols = ["loan_id"] + [c for c in ("segment", "EAD", "RW", "PTI_pre", "DSCR_pre") if c in port.columns]
    port = port[keep_cols].drop_duplicates(subset=["loan_id"])

    df = df.merge(port, on="loan_id", how="left", suffixes=("", "_port"))

    def fill_col(c: str) -> None:
        if c not in df.columns:
            if f"{c}_port" in df.columns:
                df[c] = df[f"{c}_port"]
            return
        if f"{c}_port" not in df.columns:
            return
        df[c] = df[c].combine_first(df[f"{c}_port"])

    for c in ("segment", "EAD", "RW", "PTI_pre", "DSCR_pre"):
        fill_col(c)

    drop_ports = [c for c in df.columns if c.endswith("_port")]
    if drop_ports:
        df.drop(columns=drop_ports, inplace=True)

    return df


# ===========================================================
# Helpers CIB (razonamiento y codificación auditable)
# ===========================================================
def _cib_rationale_row(
    r: pd.Series,
    risk_posture: str,
    accion_micro: str,
    macro_action: str,
    accion_final: str,
    macro_rationale: str,
    flags: Dict[str, Any],
) -> Tuple[str, str, str]:
    rp_raw = str(risk_posture or "balanceado").strip().lower()
    rp_raw = (
        rp_raw.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    if rp_raw in {"prudente", "prudencial"}:
        rp_l = "prudencial"
    elif rp_raw in {"desinversion", "desinversionn", "desinvertir"}:
        rp_l = "desinversion"
    elif rp_raw in {"balanceado", "balanced"}:
        rp_l = "balanceado"
    else:
        rp_l = "balanceado"

    rp_u = rp_l.upper()

    def _fmt_metric(x, ndigits=2) -> str:
        try:
            if x is None:
                return "NA"
            xf = float(x)
            if pd.isna(xf):
                return "NA"
            return f"{xf:.{ndigits}f}"
        except Exception:
            return "NA"

    eva_pre = _safe_float(r.get("EVA_pre", r.get("EVA_before", r.get("EVA", 0.0))))
    eva_post = _safe_float(r.get("EVA_post", r.get("EVA_after", 0.0)))
    deva = _safe_float(r.get("ΔEVA", r.get("delta_eva", r.get("EVA_gain", 0.0))))
    rorwa_pre = _safe_float(r.get("RORWA_pre", r.get("RORWA", 0.0)))

    cap_rel = _safe_float(r.get("capital_liberado", r.get("Capital_release", 0.0)))
    price = _safe_float(r.get("precio_optimo", r.get("price", 0.0)))
    pnl = _safe_float(r.get("pnl", r.get("PnL", 0.0)))

    pd_ = _safe_float(r.get("PD", 0.0))
    lgd = _safe_float(r.get("LGD", 0.0))
    dpd = _safe_float(r.get("DPD", 0.0))
    secured = _safe_str(r.get("secured", ""))

    pti = r.get("PTI_post", r.get("PTI", None))
    dscr = r.get("DSCR_post", r.get("DSCR", None))

    hurdle = _safe_float(getattr(cfg.CONFIG.regulacion, "hurdle_rate", 0.0))

    governance = "Micro-led"
    if accion_final != accion_micro:
        if accion_final == macro_action:
            governance = "Macro override (portfolio steering) with guardrails"
        else:
            governance = "Guardrail override (risk policy)"

    fire_sale = bool(flags.get("fire_sale", False))
    restruct_feasible = bool(flags.get("restruct_feasible", True))
    micro_meets_hurdle = bool(flags.get("micro_meets_hurdle", False))

    # RC10
    if bool(flags.get("missing_viability_inputs", False)):
        code = "RC10_MISSING_VIABILITY_INPUTS"
        proposed_action = _safe_str(flags.get("proposed_action", ""), "")
        txt = (
            f"[{rp_u}] RESTRUCT blocked: missing viability inputs (PTI/DSCR). "
            f"Proposed={proposed_action or 'REESTRUCTURAR'} (Micro={accion_micro}, Macro={macro_action}) → Final={accion_final}. "
            f"DSCR={_fmt_metric(dscr)}, PTI={_fmt_metric(pti)}. "
            f"PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}."
        )
        governance = "Guardrail override (missing PTI/DSCR)"
        return code, txt, governance

    # SELL bloqueado por fire-sale (rationale)
    sell_requested = (accion_micro == "VENDER") or (macro_action == "VENDER")
    if sell_requested and accion_final != "VENDER" and fire_sale and rp_l != "desinversion":
        code = "RC02_SELL_BLOCKED_FIRE_SALE"
        txt = (
            f"[{rp_u}] SELL blocked by fire-sale guardrail. "
            f"Requested(Micro={accion_micro}, Macro={macro_action}) → Final={accion_final}. "
            f"CapRel={_fmt_eur(cap_rel)}, Price={_fmt_eur(price)}, P&L={_fmt_eur(pnl)}. "
            f"Risk PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}, Secured={secured}. "
            f"MacroEvidence={'YES' if _safe_str(macro_rationale).strip() else 'NO'}."
        )
        return code, txt, governance

    # Casos por decisión
    if accion_final == "VENDER":
        if macro_action == "VENDER" and accion_micro != "VENDER":
            code = "RC01_MACRO_SELL_STEERING"
            txt = (
                f"[{rp_u}] SELL to reduce NPE and optimise capital. "
                f"Micro={accion_micro}, Macro={macro_action}. "
                f"CapRel={_fmt_eur(cap_rel)}, Price={_fmt_eur(price)}, P&L={_fmt_eur(pnl)}. "
                f"PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}, Secured={secured}."
            )
            if fire_sale and rp_l != "desinversion":
                txt += " (Flag: FireSale risk noted.)"
            return code, txt, governance

        code = "RC06_MICRO_SELL_VALUE_NEGATIVE"
        txt = (
            f"[{rp_u}] SELL on weak risk-adjusted economics. "
            f"EVA_pre={_fmt_eur(eva_pre)}, RORWA_pre={_pct(rorwa_pre)} vs hurdle={_pct(hurdle)}. "
            f"CapRel={_fmt_eur(cap_rel)}, Price={_fmt_eur(price)}, P&L={_fmt_eur(pnl)}. "
            f"PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}."
        )
        return code, txt, governance

    if accion_final == "REESTRUCTURAR":
        if not restruct_feasible:
            code = "RC09_RESTRUCT_REQUIRES_VIABILITY_CHECK"
            txt = (
                f"[{rp_u}] RESTRUCT selected but viability constraints fail or are incomplete. "
                f"ΔEVA={_fmt_eur(deva)}, EVA_post={_fmt_eur(eva_post)}; "
                f"DSCR={_fmt_metric(dscr)}, PTI={_fmt_metric(pti)}."
            )
            return code, txt, governance

        if macro_action == "REESTRUCTURAR" and accion_micro != "REESTRUCTURAR":
            code = "RC03_MACRO_RESTRUCT_STEERING"
            txt = (
                f"[{rp_u}] RESTRUCT to stabilise and recover value. "
                f"Micro={accion_micro}, Macro={macro_action}. "
                f"ΔEVA={_fmt_eur(deva)}, EVA_post={_fmt_eur(eva_post)}. "
                f"Affordability: DSCR={_fmt_metric(dscr)}, PTI={_fmt_metric(pti)}. "
                f"PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}."
            )
            return code, txt, governance

        code = "RC03_MICRO_RESTRUCT_VALUE_UPLIFT"
        txt = (
            f"[{rp_u}] RESTRUCT supported by value uplift. "
            f"ΔEVA={_fmt_eur(deva)}, EVA_post={_fmt_eur(eva_post)}; "
            f"DSCR={_fmt_metric(dscr)}, PTI={_fmt_metric(pti)}. "
            f"PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}."
        )
        return code, txt, governance

    # KEEP
    if micro_meets_hurdle:
        code = "RC05_KEEP_MEETS_HURDLE"
        txt = (
            f"[{rp_u}] KEEP: RORWA_pre={_pct(rorwa_pre)} ≥ hurdle={_pct(hurdle)}; "
            f"EVA_pre={_fmt_eur(eva_pre)}. PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}."
        )
        return code, txt, governance

    code = "RC05_KEEP_ACCEPTABLE_ECONOMICS"
    txt = (
        f"[{rp_u}] KEEP: economics acceptable and alternatives not compelling. "
        f"EVA_pre={_fmt_eur(eva_pre)}, ΔEVA={_fmt_eur(deva)}. "
        f"PD={_pct(pd_)}, LGD={_pct(lgd)}, DPD={dpd:.0f}."
    )
    return code, txt, governance


# ===========================================================
# 1) MICRO
# ===========================================================
def _run_micro_inference(
    model_path: str,
    portfolio_path: str,
    vecnorm_path: Optional[str],
    risk_posture: str,
    tag_suffix: str,
    device: str = "auto",
    seed: int = 42,
    deterministic: bool = True,
) -> Tuple[pd.DataFrame, str]:
    cfg_inf = InferenceConfig(
        model_path=model_path,
        portfolio_path=portfolio_path,
        device=device,
        seed=seed,
        deterministic=deterministic,
        tag=f"coord_micro_{tag_suffix}",
        vecnormalize_path=vecnorm_path,
        risk_posture=risk_posture,
    )

    logger.info(f"[MICRO] Lanzando inferencia (risk_posture={risk_posture})…")
    out_dir_micro = run_inference(cfg_inf)

    xlsx_path = os.path.join(out_dir_micro, "decisiones_explicadas.xlsx")
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"No se encontró decisiones_explicadas.xlsx en {out_dir_micro}")

    df_micro = pd.read_excel(xlsx_path)
    df_micro = _ensure_loan_id_column(df_micro)
    logger.info(f"[MICRO] Decisiones cargadas: {xlsx_path} ({len(df_micro):,} préstamos).")
    return df_micro, out_dir_micro


# ===========================================================
# 2) MACRO
# ===========================================================
def _run_macro_inference(
    model_path_portfolio: str,
    portfolio_path: str,
    risk_posture: str,
    loan_model_path: Optional[str],
    tag_suffix: str,
    n_steps: int,
    top_k: int,
    device: str = "auto",
    seed: int = 42,
    deterministic: bool = True,
    vecnormalize_portfolio_path: Optional[str] = None,
    vecnormalize_loan_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    cfg_port = PortfolioInferenceConfig(
        model_path=model_path_portfolio,
        portfolio_path=portfolio_path,
        device=device,
        seed=seed,
        deterministic=deterministic,
        tag=f"coord_portfolio_{tag_suffix}",
        n_steps=n_steps,
        top_k=top_k,
        save_final_excel=True,
        risk_posture=risk_posture,
        loan_model_path=loan_model_path,
        vecnormalize_portfolio_path=vecnormalize_portfolio_path,
        vecnormalize_loan_path=vecnormalize_loan_path,
    )

    logger.info(f"[MACRO] Lanzando inferencia cartera (risk_posture={risk_posture})…")
    out_dir_macro = run_portfolio_inference(cfg_port)

    traj_path = os.path.join(out_dir_macro, "trajectory_portfolio.csv")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"No se encontró trajectory_portfolio.csv en {out_dir_macro}")

    df_steps = pd.read_csv(traj_path)
    logger.info(f"[MACRO] Trayectoria macro: {traj_path} ({len(df_steps):,} pasos).")
    return df_steps, out_dir_macro


def _split_ids_cell(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, float) and np.isnan(v):
        return []
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(";")]
    return [p for p in parts if p and p.lower() != "nan"]


def _build_macro_actions_per_loan(df_steps: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Construye “macro assignment” desde trajectory_portfolio.csv.
    SOLO ids tocados (sell/restructure). El resto se considera NO_ASIGNADO.
    """
    macro_info: Dict[str, Dict[str, Any]] = {}
    if df_steps is None or df_steps.empty:
        return macro_info

    for _, row in df_steps.iterrows():
        step = int(_safe_float(row.get("step", 0), 0))
        rationale = _safe_str(row.get("rationale_fin", ""))
        combo = _safe_str(row.get("combination_note", ""))

        sold_ids = _split_ids_cell(row.get("sold_ids", ""))
        restruct_ids = _split_ids_cell(row.get("restructured_ids", ""))

        for lid in sold_ids:
            info = macro_info.setdefault(
                str(lid),
                {"macro_action": "VENDER", "steps_sold": [], "steps_restructured": [], "rationales": []},
            )
            info["macro_action"] = "VENDER"
            info["steps_sold"].append(step)
            if rationale or combo:
                info["rationales"].append(f"[step {step}] {rationale} | {combo}")

        for lid in restruct_ids:
            info = macro_info.setdefault(
                str(lid),
                {"macro_action": "REESTRUCTURAR", "steps_sold": [], "steps_restructured": [], "rationales": []},
            )
            if info.get("macro_action") != "VENDER":
                info["macro_action"] = "REESTRUCTURAR"
            info["steps_restructured"].append(step)
            if rationale or combo:
                info["rationales"].append(f"[step {step}] {rationale} | {combo}")

    return macro_info


# ===========================================================
# 3) COMBINACIÓN MICRO + MACRO
# ===========================================================
def _combine_decisions(
    df_micro: pd.DataFrame,
    macro_actions: Dict[str, Dict[str, Any]],
    risk_posture: str,
    n_steps: int,
    top_k: int,
) -> pd.DataFrame:
    df = _ensure_loan_id_column(df_micro)

    # Preservar micro
    if "Accion" in df.columns:
        df["Accion_micro"] = df["Accion"]
    elif "action" in df.columns:
        df["Accion_micro"] = df["action"]
    else:
        df["Accion_micro"] = ""

    if "Explanation" in df.columns:
        df["Explanation_micro"] = df["Explanation"]
    else:
        df["Explanation_micro"] = ""

    n_rows = len(df)
    if n_rows == 0:
        logger.warning("⚠️ _combine_decisions(): df_micro vacío → devolviendo vacío.")
        return df

    def _pad_or_trim(lst, fill):
        """Garantiza len(lst)==n_rows para asignación segura a df[col]."""
        if len(lst) == n_rows:
            return lst
        if len(lst) == 0:
            logger.warning(f"⚠️ _combine_decisions(): lista vacía detectada → rellenando con '{fill}'.")
            return [fill] * n_rows
        if len(lst) < n_rows:
            logger.warning(f"⚠️ _combine_decisions(): len(lista)={len(lst)} < n_rows={n_rows} → padding con '{fill}'.")
            return lst + [fill] * (n_rows - len(lst))
        logger.warning(f"⚠️ _combine_decisions(): len(lista)={len(lst)} > n_rows={n_rows} → truncando.")
        return lst[:n_rows]

    def _safe_bool_cell(v) -> bool:
        """Parse robusto para booleanos leídos de Excel/CSV (evita bool('False')==True)."""
        if v is None:
            return False
        try:
            if pd.isna(v):
                return False
        except Exception:
            pass
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return bool(int(v))
        s = str(v).strip().lower()
        if s in {"true", "t", "1", "yes", "y", "si", "sí"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return bool(v)

    # Outputs acumulados
    accion_final_list: List[str] = []
    accion_macro_list: List[str] = []
    macro_selected_list: List[bool] = []
    convergencia_list: List[str] = []

    explanation_micro_list: List[str] = []
    explanation_macro_list: List[str] = []
    explanation_final_list: List[str] = []

    reason_code_list: List[str] = []
    rationale_cib_list: List[str] = []
    macro_evidence_list: List[str] = []
    governance_list: List[str] = []

    # Métricas persistidas
    fire_sale_list: List[bool] = []
    price_to_ead_list: List[float] = []
    price_ratio_ead_list: List[float] = []
    pnl_ratio_book_list: List[float] = []
    book_value_list: List[float] = []
    pnl_book_list: List[float] = []
    fire_sale_threshold_list: List[float] = []
    fire_sale_enabled_list: List[bool] = []
    fire_sale_triggered_list: List[bool] = []
    fire_sale_triggers_list: List[str] = []
    price_ratio_book_list: List[float] = []
    fire_sale_threshold_book_list: List[float] = []

    missing_viability_list: List[bool] = []

    g = _get_guardrails(risk_posture)
    hurdle = _safe_float(getattr(cfg.CONFIG.regulacion, "hurdle_rate", 0.0))
    rp_l = (risk_posture or "balanceado").strip().lower()

    # Loop por préstamo
    for _, r in df.iterrows():
        loan_id = str(r.get("loan_id", _get_loan_id_from_row(r)) or "")
        accion_micro = _normalize_action(r.get("Accion_micro", "")) or "MANTENER"

        macro_info = macro_actions.get(loan_id)
        macro_selected = bool(macro_info is not None)

        # Macro action: neutral si no seleccionado (no inventamos NO_ASIGNADO)
        if not macro_selected:
            macro_action = accion_micro
            macro_rationale = (
                f"Macro not applied (n_steps={n_steps}, top_k={top_k}) ⇒ "
                f"macro_action defaults to micro action (no portfolio steering)."
            )
        else:
            macro_action = _normalize_action(macro_info.get("macro_action", "")) or "MANTENER"
            macro_rationale = (
                " || ".join(macro_info.get("rationales", []) or [])
                or "Macro selected (no rationale text)."
            )

        # Inputs base
        ead = _safe_float(r.get("EAD", 0.0))
        rw = _safe_float(r.get("RW", np.nan), np.nan)

        price = _safe_float(r.get("precio_optimo", r.get("price", np.nan)), np.nan)
        pnl = _safe_float(r.get("pnl", r.get("PnL", np.nan)), np.nan)
        cap_rel = _safe_float(r.get("capital_liberado", r.get("capital_release", np.nan)), np.nan)
        deva = _safe_float(r.get("ΔEVA", r.get("delta_eva", r.get("EVA_gain", np.nan))), np.nan)
        rorwa_pre = _safe_float(r.get("RORWA_pre", r.get("RORWA", np.nan)), np.nan)

        # Hurdle test
        micro_meets_hurdle = False
        if hurdle > 0 and not np.isnan(rorwa_pre):
            try:
                micro_meets_hurdle = (float(rorwa_pre) >= float(hurdle))
            except Exception:
                micro_meets_hurdle = False

        # Viabilidad inputs (preferimos *_pre)
        pti = r.get("PTI_post", r.get("PTI", None))
        dscr = r.get("DSCR_post", r.get("DSCR", None))

        pti_pre = r.get("PTI_pre", r.get("PTI_post", r.get("PTI", None)))
        dscr_pre = r.get("DSCR_pre", r.get("DSCR_post", r.get("DSCR", None)))

        if isinstance(pti_pre, float) and np.isnan(pti_pre):
            pti_pre = None
        if isinstance(dscr_pre, float) and np.isnan(dscr_pre):
            dscr_pre = None

        # Feasible restructure (numérica)
        dscr_min = float(g["DSCR_MIN"])
        pti_max = g.get("PTI_MAX", None)
        restruct_feasible = True

        if dscr is not None and not (isinstance(dscr, float) and np.isnan(dscr)):
            try:
                restruct_feasible &= (float(dscr) >= dscr_min)
            except Exception:
                pass

        if pti_max is not None and pti is not None and not (isinstance(pti, float) and np.isnan(pti)):
            try:
                restruct_feasible &= (float(pti) <= float(pti_max))
            except Exception:
                pass

        if not np.isnan(deva):
            restruct_feasible &= (deva >= float(g["MIN_DEVA_RESTRUCT"]))
        else:
            restruct_feasible &= False

        # RC10 missing viability (segment-aware)
        seg_hint = _safe_str(r.get("segmento_banco", r.get("segment_raw", r.get("segment", ""))), "").strip().lower()
        needs_pti = any(k in seg_hint for k in ("mortgage", "hipotec", "retail", "minorista", "consumer", "consumo"))
        needs_dscr = not needs_pti

        missing_pti = _is_missing(pti_pre)
        missing_dscr = _is_missing(dscr_pre)

        missing_viability_inputs = (needs_pti and missing_pti) or (needs_dscr and missing_dscr)
        if missing_viability_inputs:
            restruct_feasible = False

        # Fire-sale robusto (preferir simulador micro; fallback mínimo)
        price_ratio_ead = _safe_float(r.get("price_ratio_ead", np.nan), np.nan)
        if np.isnan(price_ratio_ead):
            if (ead > 0) and (not np.isnan(price)):
                price_ratio_ead = float(price / ead)

        book_value = _safe_float(r.get("book_value", np.nan), np.nan)
        pnl_book = _safe_float(r.get("pnl_book", np.nan), np.nan)
        ratio_book = _safe_float(r.get("price_ratio_book", np.nan), np.nan)
        thr_book = _safe_float(r.get("fire_sale_threshold_book", np.nan), np.nan)

        lgd = _safe_float(r.get("LGD", np.nan), np.nan)

        if np.isnan(book_value):
            cov = _safe_float(r.get("coverage_rate", np.nan), np.nan)
            if np.isfinite(cov) and cov > 1.0:
                cov = cov / 100.0
            if np.isfinite(cov) and (ead > 0):
                cov = float(np.clip(cov, 0.0, 1.0))
                book_value = ead * (1.0 - cov)
            elif (ead > 0) and (not np.isnan(lgd)):
                book_value = ead * (1.0 - float(np.clip(lgd, 0.0, 1.0)))

        if np.isnan(pnl_book):
            if (np.isfinite(price) and np.isfinite(book_value) and book_value > 0):
                pnl_book = float(price - book_value)
            else:
                pnl_book = pnl

        if np.isnan(ratio_book) and np.isfinite(price) and np.isfinite(book_value) and book_value > 0:
            ratio_book = float(price / book_value)

        pnl_ratio_book = np.nan
        if np.isfinite(book_value) and book_value > 0 and np.isfinite(pnl_book):
            pnl_ratio_book = float(pnl_book / book_value)

        thr_ead = _safe_float(r.get("fire_sale_threshold", np.nan), np.nan)
        min_price_ratio_ead = float(g["MIN_PRICE_TO_EAD"])
        if np.isfinite(thr_ead):
            min_price_ratio_ead = max(min_price_ratio_ead, float(thr_ead))

        fire_sale_sim = _safe_bool_cell(r.get("fire_sale", r.get("Fire_Sale", False)))

        fire_sale_enabled = (rp_l in ("prudencial", "balanceado"))
        fire_sale_triggered = False
        triggers: List[str] = []

        if fire_sale_enabled:
            if fire_sale_sim:
                fire_sale_triggered = True
                triggers.append("SIM_FLAG")

            if np.isfinite(thr_book) and np.isfinite(ratio_book):
                if ratio_book < float(thr_book):
                    fire_sale_triggered = True
                    triggers.append("PX_BOOK")
            else:
                if (not np.isnan(price_ratio_ead)) and (price_ratio_ead < float(min_price_ratio_ead)):
                    fire_sale_triggered = True
                    triggers.append("PX_EAD")

            max_pnl_ratio_book = float(g.get("MAX_FIRE_SALE_PNL_RATIO_BOOK", -0.75))
            if (not np.isnan(pnl_ratio_book)) and (pnl_ratio_book < max_pnl_ratio_book):
                fire_sale_triggered = True
                triggers.append("PNL_BOOK_RATIO")

            max_pnl_abs = float(g.get("MAX_FIRE_SALE_PNL_ABS", -50_000.0))
            if (not np.isnan(pnl_book)) and (pnl_book <= max_pnl_abs):
                fire_sale_triggered = True
                triggers.append("PNL_ABS")

        fire_sale = bool(fire_sale_triggered)
        price_to_ead = (price / ead) if (ead > 0 and not np.isnan(price)) else np.nan

        # Arbitraje: macro wins salvo guardrails DUROS
        accion_candidate = accion_micro
        proposed_action = accion_micro

        if macro_selected:
            if accion_micro == macro_action:
                accion_candidate = accion_micro
            else:
                accion_candidate = macro_action

            proposed_action = accion_candidate

            # Guardrail DURO 1: prudencial/balanceado no vende si fire_sale
            if accion_candidate == "VENDER" and fire_sale and rp_l in ("prudencial", "balanceado"):
                accion_candidate = accion_micro
                if accion_micro == "VENDER":
                    accion_candidate = "REESTRUCTURAR" if restruct_feasible else "MANTENER"

            # Guardrail DURO 2: no adoptar REESTRUCTURAR si no feasible
            if accion_candidate == "REESTRUCTURAR" and not restruct_feasible:
                accion_candidate = accion_micro
                if accion_micro == "REESTRUCTURAR":
                    accion_candidate = "MANTENER"

        # RC10: si propones reestructurar pero faltan inputs => mantener
        if accion_candidate == "REESTRUCTURAR" and missing_viability_inputs:
            accion_candidate = "MANTENER"

        # Última línea prudencial/balanceado: si aún queda vender con fire_sale => degradar
        blocked_sell_fire_sale = False
        if (accion_candidate == "VENDER") and (rp_l in ("prudencial", "balanceado")) and fire_sale:
            blocked_sell_fire_sale = True
            accion_candidate = "REESTRUCTURAR" if (restruct_feasible and (not missing_viability_inputs)) else "MANTENER"

        accion_final = accion_candidate

        flags = {
            "restruct_feasible": restruct_feasible,
            "fire_sale": fire_sale,
            "micro_meets_hurdle": micro_meets_hurdle,
            "sell_requested": (accion_micro == "VENDER") or (macro_action == "VENDER"),
            "blocked_sell_fire_sale": blocked_sell_fire_sale,
            "missing_viability_inputs": bool(missing_viability_inputs),
            "proposed_action": proposed_action,
        }

        # Convergencia (audit)
        if not macro_selected:
            convergencia = "MACRO_NOT_APPLIED"
        else:
            if accion_micro == macro_action:
                convergencia = "AGREE_MICRO_MACRO"
            else:
                convergencia = "MACRO_WINS" if accion_final == macro_action else "GUARDRAIL_OVERRIDE"

        # Evidence micro (si viene vacío, lo construimos)
        exp_micro_raw = _safe_str(r.get("Explanation_micro", ""))
        steps_micro = _safe_str(r.get("Explain_Steps", ""))
        micro_detail = exp_micro_raw.strip() or steps_micro.strip()

        if (not micro_detail) or (micro_detail == "❓"):
            pead_txt = "NA"
            try:
                if (ead > 0) and (not np.isnan(price_to_ead)):
                    pead_txt = f"{float(price_to_ead):.3f}"
            except Exception:
                pead_txt = "NA"

            micro_detail = (
                f"loan_id={loan_id}, EAD={_fmt_eur(ead)}, RW={rw}, RORWA_pre={_pct(rorwa_pre) if not np.isnan(rorwa_pre) else 'NA'}, "
                f"ΔEVA={_fmt_eur(deva) if not np.isnan(deva) else 'NA'}, "
                f"CapRel={_fmt_eur(cap_rel) if not np.isnan(cap_rel) else 'NA'}, Price/EAD={pead_txt}, "
                f"P&L={_fmt_eur(pnl) if not np.isnan(pnl) else 'NA'}, DSCR={dscr}, PTI={pti}, "
                f"FireSale={fire_sale}, RestructFeasible={restruct_feasible}."
            )

        reason_code, rationale_cib, governance = _cib_rationale_row(
            r=r,
            risk_posture=risk_posture,
            accion_micro=accion_micro,
            macro_action=macro_action,
            accion_final=accion_final,
            macro_rationale=(macro_rationale if macro_selected else ""),
            flags=flags,
        )

        explanation_micro = f"Micro={accion_micro}. Evidence: {micro_detail}"
        explanation_macro = f"Macro={macro_action}. Evidence: {macro_rationale}"
        explanation_final = f"{rationale_cib} | Governance={governance} | Convergencia={convergencia}"

        # Append (SIEMPRE una vez por fila)
        accion_macro_list.append(macro_action)
        macro_selected_list.append(bool(macro_selected))
        convergencia_list.append(convergencia)
        accion_final_list.append(accion_final)

        explanation_micro_list.append(explanation_micro)
        explanation_macro_list.append(explanation_macro)
        explanation_final_list.append(explanation_final)

        reason_code_list.append(reason_code)
        rationale_cib_list.append(rationale_cib)
        macro_evidence_list.append(macro_rationale if macro_selected else "")
        governance_list.append(governance)

        fire_sale_list.append(bool(fire_sale))
        fire_sale_threshold_list.append(float(min_price_ratio_ead))
        fire_sale_threshold_book_list.append(float(thr_book) if np.isfinite(thr_book) else np.nan)

        price_to_ead_list.append(float(price_to_ead) if not np.isnan(price_to_ead) else np.nan)
        price_ratio_ead_list.append(float(price_ratio_ead) if not np.isnan(price_ratio_ead) else np.nan)

        pnl_ratio_book_list.append(float(pnl_ratio_book) if not np.isnan(pnl_ratio_book) else np.nan)
        book_value_list.append(float(book_value) if not np.isnan(book_value) else np.nan)
        pnl_book_list.append(float(pnl_book) if not np.isnan(pnl_book) else np.nan)

        fire_sale_enabled_list.append(bool(fire_sale_enabled))
        fire_sale_triggered_list.append(bool(fire_sale_triggered))
        fire_sale_triggers_list.append(";".join(triggers) if triggers else "")

        price_ratio_book_list.append(float(ratio_book) if np.isfinite(ratio_book) else np.nan)
        missing_viability_list.append(bool(missing_viability_inputs))

    # saneo final de longitudes
    accion_macro_list = _pad_or_trim(accion_macro_list, "NO_ASIGNADO")
    macro_selected_list = _pad_or_trim(macro_selected_list, False)
    convergencia_list = _pad_or_trim(convergencia_list, "MACRO_NOT_APPLIED")
    accion_final_list = _pad_or_trim(accion_final_list, "MANTENER")

    explanation_micro_list = _pad_or_trim(explanation_micro_list, "Micro=NA")
    explanation_macro_list = _pad_or_trim(explanation_macro_list, "Macro=NO_ASIGNADO")
    explanation_final_list = _pad_or_trim(explanation_final_list, "Fallback")

    reason_code_list = _pad_or_trim(reason_code_list, "RC00_FALLBACK")
    rationale_cib_list = _pad_or_trim(rationale_cib_list, "Fallback")
    macro_evidence_list = _pad_or_trim(macro_evidence_list, "")
    governance_list = _pad_or_trim(governance_list, "Fallback")

    fire_sale_list = _pad_or_trim(fire_sale_list, False)
    price_to_ead_list = _pad_or_trim(price_to_ead_list, np.nan)
    price_ratio_ead_list = _pad_or_trim(price_ratio_ead_list, np.nan)
    pnl_ratio_book_list = _pad_or_trim(pnl_ratio_book_list, np.nan)
    book_value_list = _pad_or_trim(book_value_list, np.nan)
    pnl_book_list = _pad_or_trim(pnl_book_list, np.nan)
    price_ratio_book_list = _pad_or_trim(price_ratio_book_list, np.nan)
    fire_sale_threshold_book_list = _pad_or_trim(fire_sale_threshold_book_list, np.nan)

    fire_sale_threshold_list = _pad_or_trim(fire_sale_threshold_list, np.nan)
    fire_sale_enabled_list = _pad_or_trim(fire_sale_enabled_list, bool(rp_l in ("prudencial", "balanceado")))
    fire_sale_triggered_list = _pad_or_trim(fire_sale_triggered_list, False)
    fire_sale_triggers_list = _pad_or_trim(fire_sale_triggers_list, "")
    missing_viability_list = _pad_or_trim(missing_viability_list, False)

    # Columnas finales
    df["Accion_macro"] = accion_macro_list
    df["Macro_Selected"] = macro_selected_list
    df["Convergencia_Caso"] = convergencia_list

    df["Accion_final"] = accion_final_list
    df["Accion"] = df["Accion_final"]  # legacy

    # IMPORTANTÍSIMO: mantener alias usados por KPI/reporting
    if "decision_final" in df.columns:
        df["decision_final"] = df["Accion_final"]
    else:
        df["decision_final"] = df["Accion_final"]

    if "Decision_Final" in df.columns:
        df["Decision_Final"] = df["Accion_final"]

    df["Explanation_micro"] = explanation_micro_list
    df["Explanation_macro"] = explanation_macro_list
    df["Explanation_final"] = explanation_final_list
    df["Explanation"] = df["Explanation_final"]  # legacy

    df["Reason_Code"] = reason_code_list
    df["Rationale_CIB"] = rationale_cib_list
    df["Macro_Evidence"] = macro_evidence_list
    df["Decision_Governance"] = governance_list

    df["Fire_Sale"] = fire_sale_list
    df["Price_to_EAD"] = price_to_ead_list
    df["price_ratio_ead"] = price_ratio_ead_list
    df["pnl_ratio_book"] = pnl_ratio_book_list
    df["book_value"] = book_value_list
    df["pnl_book"] = pnl_book_list
    df["price_ratio_book"] = price_ratio_book_list
    df["fire_sale_threshold_book"] = fire_sale_threshold_book_list
    df["fire_sale_threshold"] = fire_sale_threshold_list
    df["FireSale_Enabled"] = fire_sale_enabled_list
    df["FireSale_Triggered"] = fire_sale_triggered_list
    df["FireSale_Triggers"] = fire_sale_triggers_list
    df["Missing_Viability_Inputs"] = missing_viability_list

    # Reorden visual (enforce_schema reordenará canónicas primero)
    preferred = [
        "loan_id",
        "segment",
        "EAD",
        "RW",
        "Accion_micro",
        "Accion_macro",
        "Macro_Selected",
        "Accion_final",
        "decision_final",
        "Convergencia_Caso",
        "Reason_Code",
        "Decision_Governance",
        "Missing_Viability_Inputs",
        "Rationale_CIB",
        "Macro_Evidence",
        "Fire_Sale",
        "FireSale_Enabled",
        "FireSale_Triggered",
        "FireSale_Triggers",
        "Price_to_EAD",
        "fire_sale_threshold",
        "Explanation_micro",
        "Explanation_macro",
        "Explanation_final",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    # ===========================================================
    # HARD GUARDRAILS POST-PROCESADO (bank-ready, schema-stable)
    # ===========================================================

    # 1) Posture gate estricto (solo si Accion_final=REESTRUCTURAR)
    try:
        posture_key = str(risk_posture or "balanceado").lower().strip()
        g2 = DEFAULT_GUARDRAILS.get(posture_key, DEFAULT_GUARDRAILS["balanceado"])
        dscr_min_g = float(g2.get("DSCR_MIN", 1.10))
        pti_max_g = float(g2.get("PTI_MAX", 0.40))

        accion_u = df["Accion_final"].astype(str).str.upper()
        rest_mask = accion_u.eq("REESTRUCTURAR")

        pti_s = pd.to_numeric(df.get("PTI_post", np.nan), errors="coerce")
        dscr_s = pd.to_numeric(df.get("DSCR_post", np.nan), errors="coerce")

        has_dscr = rest_mask & dscr_s.notna()
        has_pti = rest_mask & dscr_s.isna() & pti_s.notna()
        no_gate = rest_mask & dscr_s.isna() & pti_s.isna()

        bad_gate = (has_dscr & (dscr_s < dscr_min_g)) | (has_pti & (pti_s > pti_max_g)) | no_gate

        if bad_gate.any():
            df.loc[bad_gate, "Accion_final"] = "MANTENER"
            df.loc[bad_gate, "Accion"] = "MANTENER"
            df.loc[bad_gate, "decision_final"] = "MANTENER"
            if "Decision_Final" in df.columns:
                df.loc[bad_gate, "Decision_Final"] = "MANTENER"

            if "Decision_Governance" in df.columns:
                df.loc[bad_gate, "Decision_Governance"] = "HARD_GUARDRAIL_OVERRIDE_RESTRUCT_GATE"
            else:
                df["Decision_Governance"] = ""
                df.loc[bad_gate, "Decision_Governance"] = "HARD_GUARDRAIL_OVERRIDE_RESTRUCT_GATE"

            if "Reason_Code" in df.columns:
                df.loc[bad_gate, "Reason_Code"] = "RC12_RESTRUCT_BLOCKED_STRICT_POSTURE_GATE"
    except Exception as e:
        logger.warning(f"⚠ Guardrail gate postura (DSCR/PTI) no aplicado: {e}")

    # 2) Fire-sale hard guardrail (NUNCA SELL si Price_to_EAD < fire_sale_threshold, en TODAS las posturas)
    try:
        accion_u = df["Accion_final"].astype(str).str.upper()
        sell_mask = accion_u.eq("VENDER")

        pead = pd.to_numeric(df.get("Price_to_EAD", np.nan), errors="coerce")
        thr = pd.to_numeric(df.get("fire_sale_threshold", np.nan), errors="coerce")

        fs_mask = sell_mask & pead.notna() & thr.notna() & (pead < thr)

        if fs_mask.any():
            df.loc[fs_mask, "Sell_Blocked"] = True

            if "Sell_Blocked_Reason" not in df.columns:
                df["Sell_Blocked_Reason"] = pd.Series([""] * len(df), index=df.index, dtype="string")
            else:
                df["Sell_Blocked_Reason"] = df["Sell_Blocked_Reason"].astype("string").fillna("")

            df.loc[fs_mask, "Sell_Blocked_Reason"] = "FIRE_SALE_PRICE_TO_EAD_LT_THRESHOLD"

            if "Decision_Governance" not in df.columns:
                df["Decision_Governance"] = ""
            df.loc[fs_mask, "Decision_Governance"] = "HARD_GUARDRAIL_OVERRIDE_SELL"

            if "Reason_Code" in df.columns:
                df.loc[fs_mask, "Reason_Code"] = "RC02_SELL_BLOCKED_FIRE_SALE"
            else:
                df["Reason_Code"] = ""
                df.loc[fs_mask, "Reason_Code"] = "RC02_SELL_BLOCKED_FIRE_SALE"

            df.loc[fs_mask, "FireSale_Triggered"] = True
            if "FireSale_Triggers" in df.columns:
                def _append_trigger(x):
                    x = "" if pd.isna(x) else str(x).strip()
                    add = "Price_to_EAD<threshold"
                    if not x:
                        return add
                    return x if add in x else (x + "; " + add)
                df.loc[fs_mask, "FireSale_Triggers"] = df.loc[fs_mask, "FireSale_Triggers"].apply(_append_trigger)

            # Fallback de acción: REESTRUCTURAR si viable y no faltan inputs; si no, MANTENER
            if "Missing_Viability_Inputs" in df.columns:
                mvi = df["Missing_Viability_Inputs"].fillna(True).astype(bool)
            else:
                mvi = pd.Series(True, index=df.index)

            if "restruct_viable" in df.columns:
                rv = df["restruct_viable"].fillna(False).astype(bool)
                can_restruct = rv & (~mvi)
            else:
                # Fallback: usamos DSCR/PTI_post si existen
                posture_key = str(risk_posture or "balanceado").lower().strip()
                g2 = DEFAULT_GUARDRAILS.get(posture_key, DEFAULT_GUARDRAILS["balanceado"])
                dscr_min_g = float(g2.get("DSCR_MIN", 1.10))
                pti_max_g = float(g2.get("PTI_MAX", 0.40))

                pti_s = pd.to_numeric(df.get("PTI_post", np.nan), errors="coerce")
                dscr_s = pd.to_numeric(df.get("DSCR_post", np.nan), errors="coerce")

                ok_dscr = dscr_s.notna() & (dscr_s >= dscr_min_g)
                ok_pti = dscr_s.isna() & pti_s.notna() & (pti_s <= pti_max_g)
                can_restruct = (ok_dscr | ok_pti) & (~mvi)

            df.loc[fs_mask & can_restruct, "Accion_final"] = "REESTRUCTURAR"
            df.loc[fs_mask & (~can_restruct), "Accion_final"] = "MANTENER"

            df.loc[fs_mask, "Accion"] = df.loc[fs_mask, "Accion_final"]
            df.loc[fs_mask, "decision_final"] = df.loc[fs_mask, "Accion_final"]
            if "Decision_Final" in df.columns:
                df.loc[fs_mask, "Decision_Final"] = df.loc[fs_mask, "Accion_final"]

    except Exception as e:
        logger.warning(f"⚠ Fire-sale post-guardrail no aplicado: {e}")

    # ===========================================================
    # Contrafactual vs Realizado (coherencia para summary)
    # ===========================================================
    ratio_total = _get_ratio_total_capital()

    if "capital_release_cf" not in df.columns:
        df["capital_release_cf"] = np.nan

    if "RWA_pre" in df.columns:
        rwa_pre = pd.to_numeric(df["RWA_pre"], errors="coerce")
    else:
        ead_s = pd.to_numeric(df.get("EAD", np.nan), errors="coerce")
        rw_s = pd.to_numeric(df.get("RW", np.nan), errors="coerce")
        rwa_pre = ead_s * rw_s

    df["capital_release_cf"] = pd.to_numeric(df["capital_release_cf"], errors="coerce")
    df.loc[df["capital_release_cf"].isna(), "capital_release_cf"] = (rwa_pre * float(ratio_total))

    df["pnl_realized"] = 0.0
    df["capital_release_realized"] = 0.0

    accion_final_u = df["Accion_final"].astype(str).str.upper()
    sell_mask2 = accion_final_u.eq("VENDER")
    rest_mask2 = accion_final_u.eq("REESTRUCTURAR")

    if "pnl" in df.columns:
        df.loc[sell_mask2, "pnl_realized"] = pd.to_numeric(df.loc[sell_mask2, "pnl"], errors="coerce").fillna(0.0)

    df.loc[sell_mask2, "capital_release_realized"] = pd.to_numeric(
        df.loc[sell_mask2, "capital_release_cf"], errors="coerce"
    ).fillna(0.0)

    if "capital_liberado" in df.columns:
        df.loc[rest_mask2, "capital_release_realized"] = pd.to_numeric(
            df.loc[rest_mask2, "capital_liberado"], errors="coerce"
        ).fillna(0.0)

    return df


# ===========================================================
# 4) PIPELINE COORDINADO (UNA POSTURA)
# ===========================================================
def run_coordinator_inference(
    model_micro: str,
    portfolio_path: str,
    vecnorm_micro_path: Optional[str],
    model_macro: Optional[str],
    risk_posture: str,
    n_steps: int,
    top_k: int,
    tag: str,
    base_output_dir: Optional[str] = None,
    device: str = "auto",
    seed: int = 42,
    deterministic: bool = True,
    export_audit_csv: bool = False,
    vecnorm_macro_path: Optional[str] = None,
    vecnorm_loan_path: Optional[str] = None,
) -> Tuple[str, str]:
    if not os.path.exists(model_micro):
        raise FileNotFoundError(f"Modelo MICRO no encontrado: {model_micro}")
    if not os.path.exists(portfolio_path):
        raise FileNotFoundError(f"Cartera no encontrada: {portfolio_path}")

    vecnorm_micro_path, vecnorm_macro_path, vecnorm_loan_path = _sanitize_vecnorm_paths(
        vn_micro=vecnorm_micro_path,
        vn_macro=vecnorm_macro_path,
        vn_loan=vecnorm_loan_path,
    )

    if model_macro is None:
        model_macro = _pick_default_macro_model()

    df_port = _load_portfolio_df(portfolio_path)

    # 1) MICRO
    df_micro, _ = _run_micro_inference(
        model_path=model_micro,
        portfolio_path=portfolio_path,
        vecnorm_path=vecnorm_micro_path,
        risk_posture=risk_posture,
        tag_suffix=tag,
        device=device,
        seed=seed,
        deterministic=deterministic,
    )

    # 2) MACRO
    macro_actions: Dict[str, Dict[str, Any]] = {}
    if model_macro is not None and os.path.exists(model_macro):
        df_steps, _ = _run_macro_inference(
            model_path_portfolio=model_macro,
            portfolio_path=portfolio_path,
            risk_posture=risk_posture,
            loan_model_path=model_micro,
            tag_suffix=tag,
            n_steps=n_steps,
            top_k=top_k,
            device=device,
            seed=seed,
            deterministic=deterministic,
            vecnormalize_portfolio_path=vecnorm_macro_path,
            vecnormalize_loan_path=vecnorm_loan_path,
        )
        macro_actions = _build_macro_actions_per_loan(df_steps)
    else:
        logger.warning("⚠️ No se ha encontrado modelo MACRO; se usarán solo decisiones micro.")

    # 3) COMBINAR
    df_final = _combine_decisions(
        df_micro,
        macro_actions,
        risk_posture=risk_posture,
        n_steps=n_steps,
        top_k=top_k,
    )
    df_final = _ensure_loan_id_column(df_final)
    df_final = _fill_from_portfolio(df_final, df_port)

    # 3b) schema stable
    df_final = enforce_schema(df_final)

    # 4) EXPORTAR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    posture_suffix = (risk_posture or "balanceado").lower().strip()

    if base_output_dir:
        out_dir_coord = os.path.join(base_output_dir, f"{ts}_{tag}_{posture_suffix}")
    else:
        out_dir_coord = os.path.join(REPORTS_DIR, f"coordinated_inference_{tag}_{ts}_{posture_suffix}")
    os.makedirs(out_dir_coord, exist_ok=True)

    excel_final = os.path.join(out_dir_coord, f"decisiones_finales_{posture_suffix}.xlsx")
    export_styled_excel(df_final, excel_final)

    if export_audit_csv:
        csv_final = os.path.join(out_dir_coord, f"decisiones_audit_{posture_suffix}.csv")
        df_final.to_csv(csv_final, index=False, encoding="utf-8-sig")
        logger.info(f"🧾 Audit CSV: {csv_final}")

    # Logging de convergencia (rápido)
    try:
        n = len(df_final)
        sel = int(pd.Series(df_final.get("Macro_Selected", False)).sum())
        mw = int((df_final.get("Convergencia_Caso", "") == "MACRO_WINS").sum()) if "Convergencia_Caso" in df_final.columns else 0
        gr = int((df_final.get("Convergencia_Caso", "") == "GUARDRAIL_OVERRIDE").sum()) if "Convergencia_Caso" in df_final.columns else 0
        logger.info(f"📌 Macro_Selected: {sel}/{n} ({(sel/max(n,1))*100:.1f}%) | MACRO_WINS={mw} | GUARDRAIL_OVERRIDE={gr}")
    except Exception:
        pass

    logger.info(f"✅ Inferencia coordinada ({risk_posture}) completada.")
    logger.info(f"📊 Excel final: {excel_final}")
    return out_dir_coord, excel_final


# ===========================================================
# 5) PIPELINE MULTI-POSTURA
# ===========================================================
def run_coordinator_inference_multi_posture(cfg_base: CoordinatorInferenceConfig) -> List[str]:
    outputs: List[str] = []
    postures = ["prudencial", "balanceado", "desinversion"]

    runs_root = cfg_base.base_output_dir or os.path.join(REPORTS_DIR, "runs")
    os.makedirs(runs_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base_dir = os.path.join(runs_root, f"{ts}_{cfg_base.tag}_MULTI")
    os.makedirs(run_base_dir, exist_ok=True)

    deliverable_dir = os.path.join(runs_root, f"{ts}_{cfg_base.tag}_DELIVERABLE")
    os.makedirs(deliverable_dir, exist_ok=True)

    logger.info(f"▶ MULTI-POSTURE RUN ROOT: {run_base_dir}")
    logger.info(f"📦 DELIVERABLE ROOT:      {deliverable_dir}")
    logger.info(f"  • deliverable_only:    {cfg_base.deliverable_only}")
    logger.info(f"  • export_audit_csv:    {cfg_base.export_audit_csv}")

    import glob  # local import (simple y robusto)

    for posture in postures:
        logger.info(f"▶ Ejecutando COORDINATOR (risk_posture={posture})…")

        posture_base = os.path.join(run_base_dir, posture)
        os.makedirs(posture_base, exist_ok=True)

        out_dir, excel_path = run_coordinator_inference(
            model_micro=cfg_base.model_path_micro,
            portfolio_path=cfg_base.portfolio_path,
            vecnorm_micro_path=cfg_base.vecnormalize_path_micro,

            model_macro=cfg_base.model_path_macro,
            risk_posture=posture,

            n_steps=int(cfg_base.n_steps),
            top_k=int(cfg_base.top_k),

            tag=f"{cfg_base.tag}_{posture}",
            base_output_dir=posture_base,

            device=cfg_base.device,
            seed=int(cfg_base.seed),
            deterministic=bool(cfg_base.deterministic),

            export_audit_csv=bool(cfg_base.export_audit_csv),

            vecnorm_macro_path=cfg_base.vecnormalize_path_macro,
            vecnorm_loan_path=cfg_base.vecnormalize_path_loan,
        )

        outputs.append(out_dir)

        # --- Copia Excel al DELIVERABLE ---
        dest_excel = os.path.join(deliverable_dir, f"decisiones_finales_{posture}.xlsx")
        if excel_path and os.path.exists(excel_path):
            shutil.copy2(excel_path, dest_excel)
            logger.info(f"  ✓ Excel ({posture}) => {dest_excel}")
        else:
            logger.warning(f"  ⚠ No se encontró excel_path para postura={posture}: {excel_path}")

        # --- Copia AUDIT CSV al DELIVERABLE (bank-ready) ---
        if bool(cfg_base.export_audit_csv):
            if out_dir and os.path.exists(out_dir):
                preferred = glob.glob(os.path.join(out_dir, "**", f"decisiones_audit_{posture}.csv"), recursive=True)
                candidates = preferred or glob.glob(os.path.join(out_dir, "**", "decisiones_audit_*.csv"), recursive=True)

                if candidates:
                    src_audit = candidates[0]
                    dest_audit = os.path.join(deliverable_dir, f"decisiones_audit_{posture}.csv")
                    shutil.copy2(src_audit, dest_audit)
                    logger.info(f"  ✓ Audit CSV ({posture}) => {dest_audit}")
                else:
                    logger.warning(f"  ⚠ No se encontró audit CSV en out_dir para postura={posture}: {out_dir}")
            else:
                logger.warning(f"  ⚠ out_dir inválido para postura={posture}: {out_dir}")

    logger.info("✅ DELIVERABLE generado (3 Excels).")

    if cfg_base.deliverable_only:
        try:
            shutil.rmtree(run_base_dir, ignore_errors=True)
            logger.info(f"🧹 Limpieza deliverable-only: eliminado run_dir={run_base_dir}")
        except Exception as e:
            logger.warning(f"⚠ No se pudo eliminar run_dir={run_base_dir}: {e}")
        return [deliverable_dir]

    return outputs + [deliverable_dir]


# ===========================================================
# 6) CLI
# ===========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Inferencia coordinada MICRO+MACRO (LoanEnv + PortfolioEnv) · Banco L1.5")

    p.add_argument("--model-micro", type=str, required=True, dest="model_micro", help="Ruta a best_model.zip (LoanEnv).")
    p.add_argument("--portfolio", type=str, required=True, help="Ruta a cartera (Excel/CSV).")

    p.add_argument(
        "--vn-micro",
        type=str,
        default=_pick_default_micro_vecnorm(),
        dest="vn_micro",
        help="Ruta VecNormalize MICRO (LoanEnv). Recomendado: models/vecnormalize_loan.pkl",
    )
    p.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        dest="vecnorm",
        help="(DEPRECATED) Alias legacy de --vn-micro (para compatibilidad con .bat antiguos).",
    )

    p.add_argument(
        "--model-macro",
        type=str,
        default=_pick_default_macro_model(),
        dest="model_macro",
        help="Ruta a best_model_portfolio.zip (PortfolioEnv).",
    )
    p.add_argument(
        "--vn-macro",
        type=str,
        default=_pick_default_macro_vecnorm(),
        dest="vn_macro",
        help="Ruta VecNormalize MACRO (PortfolioEnv). Recomendado: models/vecnormalize_portfolio.pkl",
    )
    p.add_argument(
        "--vn-loan",
        type=str,
        default=_pick_default_micro_vecnorm(),
        dest="vn_loan",
        help="Ruta VecNormalize LOAN (para micro re-ranking en macro).",
    )

    p.add_argument("--risk-posture", type=str, choices=["prudencial", "balanceado", "desinversion"], default="balanceado")
    p.add_argument("--n-steps", type=int, default=1, dest="n_steps")
    p.add_argument("--top-k", type=int, default=5, dest="top_k")
    p.add_argument("--tag", type=str, default="run1")
    p.add_argument("--all-postures", action="store_true")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p.add_argument("--non-deterministic", dest="deterministic", action="store_false")

    p.add_argument(
        "--keep-all",
        action="store_true",
        default=False,
        help="Si se activa, devuelve también carpetas intermedias (no solo DELIVERABLE).",
    )
    p.add_argument("--export-audit-csv", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()

    # Backward compatibility: --vecnorm (legacy) -> --vn-micro (nuevo)
    if (not getattr(args, "vn_micro", None)) and getattr(args, "vecnorm", None):
        args.vn_micro = args.vecnorm

    deliverable_only = (not bool(args.keep_all))

    if getattr(args, "all_postures", False):
        cfg_base = CoordinatorInferenceConfig(
            model_path_micro=args.model_micro,
            portfolio_path=args.portfolio,
            vecnormalize_path_micro=args.vn_micro if (args.vn_micro and os.path.exists(args.vn_micro)) else None,
            model_path_macro=args.model_macro if (args.model_macro and os.path.exists(args.model_macro)) else None,
            vecnormalize_path_macro=args.vn_macro if (args.vn_macro and os.path.exists(args.vn_macro)) else None,
            vecnormalize_path_loan=args.vn_loan if (args.vn_loan and os.path.exists(args.vn_loan)) else None,
            device=args.device,
            seed=args.seed,
            deterministic=bool(args.deterministic),
            tag=args.tag,
            n_steps=args.n_steps,
            top_k=args.top_k,
            deliverable_only=deliverable_only,
            export_audit_csv=bool(args.export_audit_csv),
        )
        outs = run_coordinator_inference_multi_posture(cfg_base)
        logger.info(f"🏁 Multi-postura completado. Outputs: {outs}")
    else:
        out_dir, excel_path = run_coordinator_inference(
            model_micro=args.model_micro,
            portfolio_path=args.portfolio,
            vecnorm_micro_path=args.vn_micro if (args.vn_micro and os.path.exists(args.vn_micro)) else None,
            model_macro=args.model_macro if (args.model_macro and os.path.exists(args.model_macro)) else None,
            risk_posture=args.risk_posture,
            n_steps=args.n_steps,
            top_k=args.top_k,
            tag=args.tag,
            base_output_dir=None,
            device=args.device,
            seed=args.seed,
            deterministic=bool(args.deterministic),
            export_audit_csv=bool(args.export_audit_csv),
            vecnorm_macro_path=args.vn_macro if (args.vn_macro and os.path.exists(args.vn_macro)) else None,
            vecnorm_loan_path=args.vn_loan if (args.vn_loan and os.path.exists(args.vn_loan)) else None,
        )
        logger.info(f"🏁 Inferencia coordinada completada. Carpeta: {out_dir}")
        logger.info(f"📌 Excel final: {excel_path}")


if __name__ == "__main__":
    main()
