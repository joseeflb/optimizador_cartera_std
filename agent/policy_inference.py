# -*- coding: utf-8 -*-
# ============================================
# agent/policy_inference.py — Inferencia de políticas PPO (Banco L1.5 + trazabilidad)
# (CORREGIDO v3.6.3 · Logging FIX + Fire-Sale Threshold dinámico + Gate DSCR + KEEP boost + Safety)
# ============================================

"""
POC — OPTIMIZADOR DE CARTERAS EN DEFAULT (Método Estándar · Basilea III)
Inferencia del modelo PPO sobre carteras reales o sintéticas.
Autor: José María Fernández-Ladreda Ballvé

Heurística financiera PRIORITARIA + PPO solo como desempate en banda ambigua.

PATCH v3.6.3 (CRÍTICO):
- Logging: FileHandler explícito para 'logs/policy_inference.log' (soluciona logs vacíos).
- Fire-sale: Threshold book-aware inyectado según postura + severidad (evita bloqueo masivo 200/200).
- Gate DSCR: Usa DSCR_MIN de estrategia en vez de hardcode 1.0.
- Scoring KEEP: Añadido bonus de opcionalidad para Secured/Low-LGD.
- EVA gain pct: denominador con suelo económico (evita “mejoras infinitas” cuando EVA≈0).
- Robustez: control de NaN/inf en métricas y scores.
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import dataclasses
import numpy as np
import pandas as pd

import config as cfg

# --- Schema enforcement (columnas estables bank-ready) ---
try:
    from reports.schema import enforce_schema  # type: ignore
except Exception:
    def enforce_schema(df):  # fallback no-op
        return df


# -------------------------------------------------------------
# Logging y paths
# -------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("logs", "reports", "models"):
    os.makedirs(os.path.join(ROOT_DIR, sub), exist_ok=True)

LOG_DIR = os.path.join(ROOT_DIR, "logs")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# --- LOGGING FIX: Logger específico con FileHandler explícito ---
logger = logging.getLogger("policy_inference")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

_fh = logging.FileHandler(os.path.join(LOG_DIR, "policy_inference.log"), encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
logger.addHandler(_sh)

logger.propagate = False  # True si quieres también en main.log


# -------------------------------------------------------------
# Imports RL y optimizadores
# -------------------------------------------------------------
from env.loan_env import LoanEnv
from optimizer.restructure_optimizer import optimize_restructure
from optimizer.price_simulator import simulate_npl_price

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise SystemExit("❌ Faltan dependencias RL. Ejecuta install_requirements_smart.py.") from e

CFG = cfg.CONFIG


# -------------------------------------------------------------
# Configuración de inferencia
# -------------------------------------------------------------
@dataclass
class InferenceConfig:
    model_path: str
    portfolio_path: str
    device: str = "auto"
    seed: int = 42
    deterministic: bool = True
    tag: str = "policy"
    vecnormalize_path: Optional[str] = None
    risk_posture: str = "balanceado"  # "prudencial" | "balanceado" | "desinversion"

    persist_outputs: bool = True
    output_dir: Optional[str] = None


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _device_auto(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _translate_action(a: int) -> str:
    return {0: "MANTENER", 1: "REESTRUCTURAR", 2: "VENDER"}.get(int(a), "DESCONOCIDA")


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        return a / b if b not in (0, 0.0, None) else default
    except Exception:
        return default


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        if key not in row.index:
            return float(default)
        v = pd.to_numeric(row.get(key), errors="coerce")
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _price_to_ead(price: float, ead: float) -> float:
    return _safe_div(price, ead, default=np.nan)


def _is_fire_sale_legacy(posture: str, pnl_sell: float, price: float, ead: float) -> bool:
    posture = (posture or "balanceado").lower()
    if posture == "prudencial":
        max_loss = -50_000.0
        min_px_ead = 0.20
    elif posture == "balanceado":
        max_loss = -75_000.0
        min_px_ead = 0.18
    else:  # desinversion
        max_loss = -250_000.0
        min_px_ead = 0.10

    px_ead = _price_to_ead(price, ead)
    if (not np.isnan(px_ead)) and (px_ead < min_px_ead):
        return True
    if pnl_sell <= max_loss:
        return True
    return False


def _get_fire_sale_threshold_for_posture(posture: str, *, secured: bool, dpd: float, lgd: float) -> float:
    """
    Threshold Price/Book calibrado para NPL:
    - prudencial: bloquea ventas MUY castigadas (no “casi todas”)
    - balanceado: algo más permisivo
    - desinversion: muy permisivo
    Ajustes: secured / dpd alto / lgd alto -> aceptamos menor Price/Book.
    """
    p = (posture or "").lower().strip()
    base = {
        "prudencial": 0.28,
        "balanceado": 0.22,
        "desinversion": 0.15,
    }.get(p, 0.22)

    if secured:
        base -= 0.02
    if dpd >= 360:
        base -= 0.02
    if lgd >= 0.75:
        base -= 0.02

    return float(min(0.40, max(0.10, base)))


def _sale_pack_from_row(row: pd.Series, posture: str) -> Dict[str, Any]:
    ead = _safe_float(row.get("EAD", 0.0))
    lgd = _safe_float(row.get("LGD", 0.60))
    dpd = _safe_float(row.get("DPD", 180.0))

    secured = False
    try:
        if pd.notna(row.get("secured", np.nan)):
            secured = bool(row.get("secured"))
    except Exception:
        secured = False

    thr_book = _get_fire_sale_threshold_for_posture(posture, secured=secured, dpd=dpd, lgd=lgd)

    pdv = row.get("PD", None)
    try:
        pdv = None if pd.isna(pdv) else float(pdv)
    except Exception:
        pdv = None

    seg = str(row.get("segmento_banco", row.get("segment", "CORPORATE"))).strip()
    rating = str(row.get("rating", "BBB")).strip()

    book_value = row.get("book_value", None)
    coverage_rate = row.get("coverage_rate", None)

    try:
        book_value = None if pd.isna(book_value) else float(book_value)
    except Exception:
        book_value = None

    try:
        coverage_rate = None if pd.isna(coverage_rate) else float(coverage_rate)
    except Exception:
        coverage_rate = None

    return simulate_npl_price(
        ead=ead,
        lgd=lgd,
        pd=pdv,
        dpd=dpd,
        segment=seg,
        secured=secured,
        rating=rating,
        book_value=book_value,
        coverage_rate=coverage_rate,
        fire_sale_price_ratio_book=thr_book,
    )


# -------------------------------------------------------------
# VecNormalize robusto (MICRO)
# -------------------------------------------------------------
def _pick_default_vn_loan_path() -> Optional[str]:
    cands = [
        os.path.join(MODELS_DIR, "vecnormalize_loan.pkl"),
        os.path.join(MODELS_DIR, "vecnormalize_micro.pkl"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    legacy = os.path.join(MODELS_DIR, "vecnormalize_final.pkl")
    if os.path.exists(legacy):
        logger.warning("⚠️ Usando VN legacy para LOAN: models/vecnormalize_final.pkl. Recomendado: vecnormalize_loan.pkl.")
        return legacy

    legacy2 = os.path.join(MODELS_DIR, "best_model_vecnormalize.pkl")
    if os.path.exists(legacy2):
        logger.warning("⚠️ Usando VN legacy2 para LOAN: models/best_model_vecnormalize.pkl. Recomendado: vecnormalize_loan.pkl.")
        return legacy2

    return None


def _vn_shape_matches_env(vn: VecNormalize, dummy_env: DummyVecEnv) -> bool:
    try:
        env_shape = getattr(dummy_env.observation_space, "shape", None)
        vn_mean = getattr(getattr(vn, "obs_rms", None), "mean", None)
        vn_shape = getattr(vn_mean, "shape", None)
        if env_shape is None or vn_shape is None:
            return True
        return tuple(env_shape) == tuple(vn_shape)
    except Exception:
        return True


def _load_vecnormalize_loan(vn_path: Optional[str], dummy_env: DummyVecEnv) -> Optional[VecNormalize]:
    if not vn_path:
        return None
    if not os.path.exists(vn_path):
        logger.warning(f"⚠️ VecNormalize LOAN no existe: {vn_path}")
        return None
    try:
        vn = VecNormalize.load(vn_path, dummy_env)
        vn.training = False
        vn.norm_reward = False

        if not _vn_shape_matches_env(vn, dummy_env):
            env_shape = getattr(dummy_env.observation_space, "shape", None)
            vn_mean = getattr(getattr(vn, "obs_rms", None), "mean", None)
            vn_shape = getattr(vn_mean, "shape", None)
            logger.warning(f"⚠️ VecNormalize LOAN INVALIDADO por mismatch: env={env_shape} vs vn={vn_shape} | {vn_path}")
            return None

        logger.info(f"🔄 VecNormalize LOAN cargado: {vn_path}")
        return vn
    except Exception as e:
        logger.warning(f"⚠️ VecNormalize LOAN incompatible: {vn_path} | {e}")
        return None


# -------------------------------------------------------------
# Mapeos consistentes con config/generate (PPO obs)
# -------------------------------------------------------------
_RATING_NUM: Dict[str, int] = {"AAA": 9, "AA": 8, "A": 7, "BBB": 6, "BB": 5, "B": 4, "CCC": 3}
_SEG_ID: Dict[str, int] = {seg.value.upper(): i for i, seg in enumerate(cfg.Segmento)}

SEG_ALIASES = {
    "LARGE CORPORATE": "CORPORATE",
    "MIDCAP": "SME",
    "PYME": "SME",
    "MINORISTA": "RETAIL",
    "HIPOTECARIO": "MORTGAGE",
    "CONSUMO": "CONSUMER",
    "OTRO": "OTHER",
}


def _rating_num(r: str) -> int:
    return _RATING_NUM.get(str(r).upper(), 6)


def _segmento_id_from_any(row: pd.Series) -> int:
    cand = None
    for col in ["segmento_banco", "segment_raw", "segment", "segmento_id"]:
        if col in row and pd.notna(row[col]):
            cand = str(row[col]).strip().upper()
            break
    if cand is None:
        return _SEG_ID.get("CORPORATE", 0)
    cand = SEG_ALIASES.get(cand, cand)
    return _SEG_ID.get(cand, _SEG_ID.get("CORPORATE", 0))


def _build_raw_obs_row(row: pd.Series) -> np.ndarray:
    """
    IMPORTANTE: esto debe coincidir con LoanEnv._get_obs() (orden y variables).
    Esta versión usa 10 features e incluye 'secured' como 0/1.
    """
    EAD = _num(row, "EAD", 0.0)
    PD = _num(row, "PD", 0.0)
    LGD = _num(row, "LGD", 0.0)
    RW = _num(row, "RW", 1.0)
    EVA = _num(row, "EVA", 0.0)

    if "RORWA" in row.index and pd.notna(row.get("RORWA")):
        RORWA = _num(row, "RORWA", 0.0)
    else:
        rate = _num(row, "rate", 0.0)
        COST_FUND = 0.006
        NI = EAD * rate - PD * LGD * EAD - EAD * COST_FUND
        RWA = EAD * RW
        RORWA = NI / RWA if RWA > 0 else 0.0

    rating_num = float(_rating_num(row.get("rating", "BBB")))
    segmento_id = float(_segmento_id_from_any(row))

    secured = 0.0
    try:
        if pd.notna(row.get("secured", np.nan)):
            secured = 1.0 if bool(row.get("secured")) else 0.0
    except Exception:
        secured = 0.0

    dpd = _num(row, "DPD", 120.0)
    DPD30 = dpd / 30.0

    obs = np.array([EAD, PD, LGD, RW, EVA, RORWA, rating_num, segmento_id, secured, DPD30], dtype=np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return obs.reshape(1, -1)


# -------------------------------------------------------------
# Carga de cartera y modelo
# -------------------------------------------------------------
def harmonize_portfolio_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if cfg.ID_COL not in df.columns:
        if "id" in df.columns:
            df[cfg.ID_COL] = df["id"].astype(str)
        else:
            df[cfg.ID_COL] = [f"loan_{i}" for i in range(len(df))]

    if "ingreso_mensual" not in df.columns and "monthly_income" in df.columns:
        df["ingreso_mensual"] = df["monthly_income"]
    if "cashflow_operativo_mensual" not in df.columns and "monthly_cfo" in df.columns:
        df["cashflow_operativo_mensual"] = df["monthly_cfo"]
    if "cuota_mensual" not in df.columns and "monthly_payment" in df.columns:
        df["cuota_mensual"] = df["monthly_payment"]

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

    if "book_value" not in df.columns:
        if "coverage_rate" in df.columns and "EAD" in df.columns:
            cov = pd.to_numeric(df["coverage_rate"], errors="coerce")
            cov = np.where(cov > 1.0, cov / 100.0, cov)
            cov = pd.Series(cov).clip(0, 1)
            df["book_value"] = pd.to_numeric(df["EAD"], errors="coerce") * (1.0 - cov)
        elif "LGD" in df.columns and "EAD" in df.columns:
            lgd = pd.to_numeric(df["LGD"], errors="coerce").clip(0, 1)
            df["book_value"] = pd.to_numeric(df["EAD"], errors="coerce") * (1.0 - lgd)

    if "segment" in df.columns:
        df["segment"] = df["segment"].astype(str).str.strip()

    return df


def load_portfolio_any(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Cartera no encontrada: {path}")

    ext = os.path.splitext(path.lower())[1]
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df = harmonize_portfolio_schema(df)

    required_min = [cfg.ID_COL, "EAD", "PD", "LGD", "RW", "EVA", "DPD"]
    missing = [c for c in required_min if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Columnas mínimas faltantes en cartera: {missing}")

    if "rating_num" not in df.columns:
        df["rating_num"] = df.get("rating", "BBB").apply(_rating_num).astype(float)

    if "segmento_id" not in df.columns:
        df["segmento_id"] = df.apply(_segmento_id_from_any, axis=1).astype(float)

    if "RONA" not in df.columns or "RORWA" not in df.columns:
        COST_FUND = 0.006
        rate = pd.to_numeric(df.get("rate", 0.0), errors="coerce").fillna(0.0)
        ead = pd.to_numeric(df["EAD"], errors="coerce").fillna(0.0)
        pdv = pd.to_numeric(df["PD"], errors="coerce").fillna(0.0)
        lgd = pd.to_numeric(df["LGD"], errors="coerce").fillna(0.0)
        rw = pd.to_numeric(df["RW"], errors="coerce").fillna(1.0)

        el = ead * pdv * lgd
        ni = ead * rate - el - ead * COST_FUND
        rwa = ead * rw

        if "RONA" not in df.columns:
            df["RONA"] = np.where(ead > 0, ni / ead, 0.0)
        if "RORWA" not in df.columns:
            df["RORWA"] = np.where(rwa > 0, ni / rwa, 0.0)

    logger.info(f"📥 Cartera cargada correctamente ({len(df):,} préstamos)")
    return df


def load_policy(model_path: str, cfg_inf: InferenceConfig) -> Tuple[PPO, Optional[VecNormalize]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modelo PPO no encontrado: {model_path}")

    dummy_env = DummyVecEnv([lambda: LoanEnv()])

    vn_path = cfg_inf.vecnormalize_path or _pick_default_vn_loan_path()
    vn_env = _load_vecnormalize_loan(vn_path, dummy_env) if vn_path else None

    device_final = _device_auto(cfg_inf.device)
    model = PPO.load(model_path, device=device_final)
    logger.info(f"🤖 Modelo PPO cargado desde: {model_path} [device={device_final}]")

    if vn_env is None:
        logger.warning("⚠️ PPO sin VecNormalize (se usará obs raw).")
    return model, vn_env


def _select_strategy_for_posture(posture: str):
    posture = (posture or "balanceado").lower()
    if posture == "prudencial":
        prof = cfg.BankProfile.PRUDENTE
    elif posture == "desinversion":
        prof = cfg.BankProfile.DESINVERSION
    else:
        prof = cfg.BankProfile.BALANCEADO

    strat = cfg.BANK_STRATEGIES[prof]
    reward = cfg.RewardParams.from_strategy(strat)
    return prof, strat, reward


# -------------------------------------------------------------
# Núcleo de inferencia por postura
# -------------------------------------------------------------
def _run_inference_for_posture(
    cfg_inf: InferenceConfig,
    out_dir: Optional[str],
    suffix: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Si cfg_inf.persist_outputs=False o out_dir=None:
      - NO exporta excel/summary
      - Devuelve (df_dec, "", "")
    """
    posture = (getattr(cfg_inf, "risk_posture", None) or "balanceado").lower()
    bank_profile, bank_strat, reward_cfg = _select_strategy_for_posture(posture)

    regulacion = cfg.CONFIG.regulacion
    hurdle = float(getattr(regulacion, "hurdle_rate", 0.0))
    ratio_total = float(regulacion.required_total_capital_ratio())

    W_EVA = float(reward_cfg.w_eva)
    W_CAPITAL = float(reward_cfg.w_capital)
    W_PNL = float(reward_cfg.w_pnl)
    PENALTY_FIRE_SALE = float(reward_cfg.penalty_fire_sale)
    PNL_PENALTY_SCALE = float(reward_cfg.pnl_penalty_scale)

    EVA_STRONGLY_NEG_EUR = float(getattr(bank_strat, "eva_strongly_neg", -50_000.0))
    EVA_MIN_IMPROVEMENT_EUR = float(getattr(bank_strat, "eva_min_improvement", 10_000.0))
    EVA_AMBIGUOUS_BAND = float(getattr(bank_strat, "eva_ambiguous_band", 10_000.0))

    PTI_MAX = float(getattr(bank_strat, "esfuerzo_bajo", 0.40))
    PTI_CRITICO = float(getattr(bank_strat, "esfuerzo_alto", 0.55))
    DSCR_MIN = float(getattr(bank_strat, "dscr_min", 1.10))

    EVA_MIN_IMPROVEMENT_PCT = 0.05
    RORWA_MARGIN = 0.002

    PD_ALTO = 0.70
    LGD_ALTO = 0.70

    logger.info(
        f"🏦 Inferencia MICRO — posture={posture} | profile={bank_profile.value} | strategy={bank_strat.name} "
        f"| W_EVA={W_EVA:.2f} W_CAP={W_CAPITAL:.2f} W_PNL={W_PNL:.2f} | persist_outputs={bool(cfg_inf.persist_outputs)}"
    )

    df = load_portfolio_any(cfg_inf.portfolio_path)

    model: Optional[PPO] = None
    vn_env: Optional[VecNormalize] = None
    try:
        model, vn_env = load_policy(cfg_inf.model_path, cfg_inf)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo cargar el modelo PPO; se usará lógica financiera pura: {e}")

    decisions: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        loan_id = row[cfg.ID_COL]

        base_eva = _safe_float(row.get("EVA", 0.0))
        base_rorwa = _safe_float(row.get("RORWA", 0.0))
        base_ead = _safe_float(row.get("EAD", 0.0))
        base_rw = _safe_float(row.get("RW", 1.0))
        base_rwa = _safe_float(row.get("RWA", base_ead * base_rw))

        base_rate = _safe_float(row.get("rate", 0.0))
        base_pd = _safe_float(row.get("PD", 0.0))
        base_lgd = _safe_float(row.get("LGD", 0.0))
        base_seg = row.get("segment", None)
        base_rat = row.get("rating", None)
        base_rona = _safe_float(row.get("RONA", row.get("Effective_RONA_pct", 0.0)))

        base_dpd = _safe_float(row.get("DPD", 120.0))
        base_secured = False
        try:
            if pd.notna(row.get("secured", np.nan)):
                base_secured = bool(row.get("secured"))
        except Exception:
            base_secured = False

        ingreso_mensual = row.get("ingreso_mensual", None)
        cfo_m = row.get("cashflow_operativo_mensual", None)

        pti_base = _safe_float(row.get("PTI", np.nan), default=np.nan)
        dscr_base = _safe_float(row.get("DSCR", np.nan), default=np.nan)
        pti_base = None if not np.isfinite(pti_base) else float(pti_base)
        dscr_base = None if not np.isfinite(dscr_base) else float(dscr_base)

        pti_pre = pti_base
        dscr_pre = dscr_base

        finance_reason_steps: List[str] = []

        restruct: Optional[Dict[str, Any]] = None
        try:
            restruct = optimize_restructure(
                ead=base_ead,
                rate=base_rate,
                pd=base_pd,
                lgd=base_lgd,
                rw=base_rw,
                ingreso_mensual=ingreso_mensual,
                cashflow_operativo_mensual=cfo_m,
                esfuerzo_max=PTI_MAX,
                dscr_min=DSCR_MIN,
                hurdle=hurdle,
            )
        except Exception as e:
            logger.warning(f"⚠️ Error optimize_restructure loan_id={loan_id}: {e}")
            restruct = None

        eva_post = base_eva
        eva_gain = 0.0
        rorwa_post = base_rorwa
        rorwa_gain = 0.0
        rwa_post = base_rwa
        pti_post = None
        dscr_post = None
        cured = False
        plazo_optimo = None
        tasa_nueva = None
        quita = None
        restruct_ok = False

        if restruct is not None:
            eva_post = _safe_float(restruct.get("EVA_post", base_eva))
            eva_gain = _safe_float(restruct.get("EVA_gain", eva_post - base_eva))
            rorwa_post = _safe_float(restruct.get("RORWA_post", base_rorwa))
            rorwa_gain = _safe_float(restruct.get("RORWA_gain", rorwa_post - base_rorwa))
            rwa_post = _safe_float(restruct.get("RWA_post", base_rwa))

            pti_post = restruct.get("PTI_post", restruct.get("PTI", None))
            dscr_post = restruct.get("DSCR_post", restruct.get("DSCR", None))

            try:
                pti_post = None if pti_post is None or (isinstance(pti_post, float) and np.isnan(pti_post)) else float(pti_post)
            except Exception:
                pass
            try:
                dscr_post = None if dscr_post is None or (isinstance(dscr_post, float) and np.isnan(dscr_post)) else float(dscr_post)
            except Exception:
                pass

            cured = bool(restruct.get("cured", False))
            plazo_optimo = restruct.get("plazo_optimo", None)
            tasa_nueva = restruct.get("tasa_nueva", None)
            quita = restruct.get("quita", None)
            restruct_ok = bool(restruct.get("ok", False))

        # EVA gain pct con suelo económico: evita “mejoras infinitas” cuando EVA≈0
        den = max(abs(base_eva), 0.02 * max(base_ead, 1.0))
        eva_gain_pct = abs(eva_gain) / (den + 1e-6)

        ambiguous_eva_band = (
            abs(base_eva) < EVA_AMBIGUOUS_BAND
            and EVA_STRONGLY_NEG_EUR < base_eva < EVA_MIN_IMPROVEMENT_EUR
        )

        if base_eva > 0 and base_rorwa >= (hurdle + RORWA_MARGIN):
            finance_reason_steps.append(f"EVA_pre={base_eva:,.0f}€ > 0 y RORWA_pre={base_rorwa:.2%} ≥ hurdle ({hurdle:.2%}).")
            base_state = "BUENO"
        elif base_eva <= EVA_STRONGLY_NEG_EUR or base_rorwa < (hurdle - RORWA_MARGIN):
            finance_reason_steps.append(f"EVA_pre={base_eva:,.0f}€ muy bajo o RORWA_pre={base_rorwa:.2%} < hurdle-δ ({hurdle-RORWA_MARGIN:.2%}).")
            base_state = "MALO"
        else:
            finance_reason_steps.append(f"Banda intermedia: EVA_pre={base_eva:,.0f}€, RORWA_pre={base_rorwa:.2%} ~ hurdle ({hurdle:.2%}).")
            base_state = "AMBIGUO"

        restruct_viable = False
        if restruct is not None:
            if posture == "prudencial":
                cond_eva = (eva_gain > EVA_MIN_IMPROVEMENT_EUR or eva_gain_pct > EVA_MIN_IMPROVEMENT_PCT)
            elif posture == "balanceado":
                cond_eva = (eva_gain > EVA_MIN_IMPROVEMENT_EUR or eva_gain_pct > EVA_MIN_IMPROVEMENT_PCT) and (eva_post > min(0.0, base_eva))
            else:
                cond_eva = (eva_gain > 2.0 * EVA_MIN_IMPROVEMENT_EUR) and (eva_post > 0)

            cond_pti = (pti_post is None) or (float(pti_post) <= PTI_MAX)
            cond_dscr = (dscr_post is None) or (float(dscr_post) >= DSCR_MIN)
            cond_cure = cured or (eva_post > base_eva and rwa_post <= base_rwa * 1.05)

            restruct_viable = bool(cond_eva and cond_pti and cond_dscr and cond_cure)

            finance_reason_steps.append(
                f"Reestructura: EVA_post={eva_post:,.0f}€ (ΔEVA={eva_gain:,.0f}€, +{eva_gain_pct*100:.1f}%), "
                f"RWA_post≈{rwa_post:,.0f}, PTI_post={pti_post if pti_post is not None else 'NA'}, "
                f"DSCR_post={dscr_post if dscr_post is not None else 'NA'}, cured={cured}."
            )

        if pti_post is not None and float(pti_post) > PTI_CRITICO:
            restruct_viable = False
            finance_reason_steps.append(f"PTI_post={float(pti_post):.2f} > {PTI_CRITICO:.2f} → esfuerzo inasumible, reestructura descartada.")
        if dscr_post is not None and float(dscr_post) < DSCR_MIN:
            restruct_viable = False
            finance_reason_steps.append(f"DSCR_post={float(dscr_post):.2f} < {DSCR_MIN:.2f} → flujo insuficiente, reestructura descartada.")

        # Venta NPL (fuente única)
        price_for_sell: Dict[str, Any] = {}
        try:
            price_for_sell = _sale_pack_from_row(row, posture)
        except Exception as e:
            logger.warning(f"⚠️ Error simulate_npl_price loan_id={loan_id}: {e}")
            price_for_sell = {}

        precio_opt = _safe_float(price_for_sell.get("precio_optimo", 0.0))
        pnl_sell = _safe_float(price_for_sell.get("pnl", 0.0))
        capital_lib_venta = _safe_float(price_for_sell.get("capital_liberado", base_rwa * ratio_total))

        resumen = price_for_sell.get("resumen", {}) or {}
        p5_sale = _safe_float(resumen.get("p5", np.nan), default=np.nan)
        p50_sale = _safe_float(resumen.get("p50", np.nan), default=np.nan)
        p95_sale = _safe_float(resumen.get("p95", np.nan), default=np.nan)

        price_ratio_ead = _safe_float(price_for_sell.get("price_ratio_ead", np.nan), default=np.nan)
        if (not np.isfinite(price_ratio_ead)) and (base_ead > 0):
            try:
                price_ratio_ead = float(precio_opt / base_ead)
            except Exception:
                price_ratio_ead = np.nan

        price_ratio_book = _safe_float(price_for_sell.get("price_ratio_book", np.nan), default=np.nan)
        thr_book = _safe_float(price_for_sell.get("fire_sale_threshold_book", np.nan), default=np.nan)

        fire_sale_enabled = posture in ("prudencial", "balanceado")

        fire_sale_source = "SIM"
        if (not np.isfinite(price_ratio_book)) or (not np.isfinite(thr_book)):
            fire_sale = _is_fire_sale_legacy(posture, pnl_sell=pnl_sell, price=precio_opt, ead=base_ead)
            fire_sale_source = "LEGACY"
        else:
            fire_sale = bool(price_for_sell.get("fire_sale", False))
            fire_sale_source = "SIM"

        fire_sale_triggered = bool(fire_sale)

        sell_blocked = bool(fire_sale_enabled and fire_sale_triggered)
        sell_blocked_reason = ""
        if sell_blocked:
            if fire_sale_source == "SIM":
                prb_txt = f"{price_ratio_book:.2f}" if np.isfinite(price_ratio_book) else "NA"
                thb_txt = f"{thr_book:.2f}" if np.isfinite(thr_book) else "NA"
                sell_blocked_reason = f"FIRE_SALE(sim): Price/Book={prb_txt} < thr_book={thb_txt} (or flag)"
            else:
                px_ead = _price_to_ead(precio_opt, base_ead)
                px_txt = f"{px_ead:.2f}" if np.isfinite(px_ead) else "NA"
                sell_blocked_reason = f"FIRE_SALE(legacy): px/EAD={px_txt} or P&L abs threshold"
            finance_reason_steps.append(f"Guardrail fire-sale: {sell_blocked_reason} → venta bloqueada.")

        risk_extremo = (base_pd >= PD_ALTO and base_lgd >= LGD_ALTO and not restruct_viable)
        if risk_extremo:
            finance_reason_steps.append(f"Riesgo extremo NPL (PD={base_pd:.1%}, LGD={base_lgd:.1%}) sin reestructura viable → sesgo hacia VENDER.")

        # Normalizaciones
        ead_scale = max(base_ead, 1.0)
        eva_pre_n = _safe_div(base_eva, ead_scale)
        deva_re_n = _safe_div((eva_post - base_eva), ead_scale)
        cap_rel_sell_n = _safe_div(capital_lib_venta, ead_scale)

        pnl_scaled = pnl_sell / max(PNL_PENALTY_SCALE, 1.0)

        if bank_profile == cfg.BankProfile.DESINVERSION:
            sell_alpha = 1.0
        elif bank_profile == cfg.BankProfile.BALANCEADO:
            sell_alpha = 0.8
        else:
            sell_alpha = 0.6

        # Gating venta
        if bank_profile == cfg.BankProfile.PRUDENTE:
            sell_allowed = ((base_state == "MALO" and not restruct_viable and not ambiguous_eva_band) or risk_extremo)
        elif bank_profile == cfg.BankProfile.BALANCEADO:
            sell_allowed = ((base_state == "MALO" and not restruct_viable) or risk_extremo)
        else:
            sell_allowed = (base_eva < 0) or risk_extremo

        if sell_blocked:
            sell_allowed = False

        # Scores
        scores: Dict[str, float] = {}

        score_keep = 0.0
        if base_eva > 0:
            score_keep += W_EVA * eva_pre_n

        if base_secured and base_lgd < 0.40:
            score_keep += 0.05 * W_EVA

        if base_eva < 0 and base_eva > EVA_STRONGLY_NEG_EUR:
            score_keep += 0.02 * W_EVA

        scores["MANTENER"] = float(score_keep)

        if restruct is not None and restruct_ok and restruct_viable:
            delta_rwa_re = max(base_rwa - rwa_post, 0.0)
            capital_lib_re = delta_rwa_re * ratio_total
            cap_rel_re_n = _safe_div(capital_lib_re, ead_scale)

            score_restruct = W_EVA * deva_re_n + W_CAPITAL * cap_rel_re_n
            if cured and posture in ("prudencial", "balanceado"):
                score_restruct += 0.10 * W_EVA * abs(eva_pre_n)

            scores["REESTRUCTURAR"] = float(score_restruct)

        eva_gain_sell_n = _safe_div(-base_eva, ead_scale)
        score_sell = sell_alpha * W_EVA * eva_gain_sell_n + W_CAPITAL * cap_rel_sell_n + W_PNL * pnl_scaled

        if base_eva > 0:
            score_sell -= PENALTY_FIRE_SALE * abs(eva_pre_n)

        if pnl_scaled < 0 and posture in ("prudencial", "balanceado"):
            score_sell -= 0.50 * abs(pnl_scaled)

        if base_state == "MALO" and not restruct_viable and base_eva <= EVA_STRONGLY_NEG_EUR:
            score_sell += 0.10 * W_EVA * abs(eva_pre_n)

        if risk_extremo:
            score_sell += 0.05 * (W_EVA * abs(eva_pre_n) + W_CAPITAL * cap_rel_sell_n)

        if not sell_allowed:
            finance_reason_steps.append("Venta bloqueada por política/guardrails → score_sell desactivado.")
            score_sell = -1e12

        scores["VENDER"] = float(score_sell)

        # Safety: evita NaN/inf en scores
        for k, v in list(scores.items()):
            if not np.isfinite(float(v)):
                scores[k] = -1e12

        decision_financial = max(scores, key=scores.get)
        best_score = scores[decision_financial]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else best_score
        score_margin = best_score - second_best_score

        finance_reason_steps.append(
            f"Scores(fin) — KEEP={scores.get('MANTENER', float('nan')):,.6f}, "
            f"REST={scores.get('REESTRUCTURAR', float('nan')):,.6f}, "
            f"SELL={scores.get('VENDER', float('nan')):,.6f}. "
            f"Mejor(fin): {decision_financial} (margen={score_margin:,.6f})."
        )

        if base_state == "BUENO" and decision_financial == "VENDER":
            finance_reason_steps.append("Guardrail: préstamo BUENO → no se vende; se fuerza MANTENER.")
            decision_financial = "MANTENER"

        decision_override: Optional[str] = None
        if base_eva >= EVA_MIN_IMPROVEMENT_EUR and dscr_base is not None and dscr_base >= DSCR_MIN:
            decision_override = "MANTENER"
            finance_reason_steps.append("Override perfil: zona cómoda (EVA y DSCR) → MANTENER.")
        elif base_eva <= EVA_STRONGLY_NEG_EUR:
            if bank_profile == cfg.BankProfile.DESINVERSION:
                decision_override = "VENDER"
                finance_reason_steps.append("Override DESINVERSION: EVA muy negativa → VENDER.")
            elif bank_profile == cfg.BankProfile.PRUDENTE:
                if restruct_viable:
                    decision_override = "REESTRUCTURAR"
                    finance_reason_steps.append("Override PRUDENTE: reestructurar antes de vender.")
                else:
                    decision_override = "VENDER"
                    finance_reason_steps.append("Override PRUDENTE: sin reestructura viable → VENDER.")

        decision_ppo: Optional[str] = None
        decision_final = decision_override or decision_financial

        used_ppo = False
        if model is not None and len(scores) > 1:
            if abs(score_margin) < 0.0005 or ambiguous_eva_band:
                raw_obs = _build_raw_obs_row(row)
                try:
                    obs = vn_env.normalize_obs(raw_obs) if vn_env is not None else raw_obs
                    a_pred, _ = model.predict(obs, deterministic=cfg_inf.deterministic)
                    decision_ppo = _translate_action(int(np.squeeze(a_pred)))
                    used_ppo = True
                    finance_reason_steps.append(f"Desempate PPO → sugiere {decision_ppo}.")

                    if decision_ppo in scores and scores[decision_ppo] >= best_score - 0.10 * abs(best_score):
                        decision_final = decision_ppo
                        finance_reason_steps.append(f"PPO dentro de tolerancia → se adopta {decision_final}.")
                except Exception as e:
                    finance_reason_steps.append(f"No se pudo usar PPO ({e}); se mantiene {decision_final}.")

        if decision_final is None:
            decision_final = "VENDER"
            finance_reason_steps.append("Fallback: acción no determinada → VENDER por prudencia.")

        final_rationale = " | ".join(
            [
                f"Financial={decision_financial}",
                f"Override={decision_override}" if decision_override else "Override=None",
                f"PPO={decision_ppo}" if used_ppo else "PPO=not_used",
                f"Final={decision_final}",
            ]
        )

        decision: Dict[str, Any] = {
            "loan_id": loan_id,
            "segment": base_seg,
            "rating": base_rat,

            "Accion": decision_final,

            "decision_financial": decision_financial,
            "decision_override": decision_override or "",
            "decision_ppo": decision_ppo or "",
            "decision_final": decision_final,
            "final_rationale": final_rationale,

            "EAD": base_ead,
            "PD": base_pd,
            "LGD": base_lgd,
            "RW": base_rw,
            "DPD": base_dpd,
            "secured": base_secured,
            "rate": base_rate,

            "EVA_pre": base_eva,
            "RWA_pre": base_rwa,
            "RORWA_pre": base_rorwa,
            "Effective_RONA_pct": base_rona,

            "PTI_pre": pti_pre,
            "DSCR_pre": dscr_pre,

            "base_state": base_state,
            "restruct_viable": bool(restruct_viable),
            "risk_extremo": bool(risk_extremo),

            "fire_sale": bool(fire_sale),
            "fire_sale_source": str(fire_sale_source),
            "FireSale_Enabled": bool(fire_sale_enabled),
            "Sell_Blocked": bool(sell_blocked),
            "Sell_Blocked_Reason": str(sell_blocked_reason),

            "Score_keep": scores.get("MANTENER", np.nan),
            "Score_restruct": scores.get("REESTRUCTURAR", np.nan),
            "Score_sell": scores.get("VENDER", np.nan),

            "bank_profile": bank_profile.value,
            "bank_strategy": bank_strat.name,

            "precio_optimo": precio_opt,
            "pnl": pnl_sell,
            "p5": p5_sale,
            "p50": p50_sale,
            "p95": p95_sale,
            "capital_release_cf": capital_lib_venta,

            "price_ratio_ead": price_ratio_ead,

            "book_value": _safe_float(price_for_sell.get("book_value", np.nan), default=np.nan),
            "book_value_source": str(price_for_sell.get("book_value_source", "")),
            "price_ratio_book": _safe_float(price_for_sell.get("price_ratio_book", np.nan), default=np.nan),
            "fire_sale_threshold_book": _safe_float(price_for_sell.get("fire_sale_threshold_book", np.nan), default=np.nan),

            "pnl_book": _safe_float(price_for_sell.get("pnl_book", np.nan), default=np.nan),

            "pnl_realized": 0.0,
            "capital_release_realized": 0.0,
        }

        decision["capital_liberado"] = 0.0

        if decision_final == "REESTRUCTURAR" and restruct is not None and restruct_viable:
            decision["EVA_post"] = eva_post
            decision["ΔEVA"] = eva_gain
            decision["EVA_gain"] = eva_gain
            decision["RORWA_post"] = rorwa_post
            decision["RORWA_gain"] = rorwa_gain
            decision["RWA_post"] = rwa_post
            decision["ok"] = restruct_ok
            decision["msg"] = restruct.get("msg", "sin mejora")

            decision["plazo_optimo"] = plazo_optimo
            decision["tasa_nueva"] = tasa_nueva
            decision["quita"] = quita

            decision["PTI_post"] = pti_post
            decision["DSCR_post"] = dscr_post
            decision["cured"] = cured

            delta_rwa = max(base_rwa - rwa_post, 0.0)
            capital_lib_restruct = delta_rwa * ratio_total
            decision["capital_liberado"] = capital_lib_restruct

            finance_reason_steps += [
                f"Acción final REESTRUCTURAR: EVA_post={eva_post:,.0f}€ (ΔEVA={eva_gain:,.0f}€), "
                f"RWA_post≈{rwa_post:,.0f} → cap_lib≈{capital_lib_restruct:,.0f}€."
            ]

        elif decision_final == "VENDER":
            decision["capital_liberado"] = capital_lib_venta
            decision["EVA_post"] = 0.0
            decision["ΔEVA"] = -base_eva

            decision["PTI_post"] = pti_pre
            decision["DSCR_post"] = dscr_pre

            finance_reason_steps += [
                f"Acción final VENDER: precio={precio_opt:,.0f}€ (px/EAD={_price_to_ead(precio_opt, base_ead):.2f}), "
                f"P&L(book)={pnl_sell:,.0f}€, cap_lib≈{capital_lib_venta:,.0f}€."
            ]

        else:
            decision["EVA_post"] = base_eva
            decision["ΔEVA"] = 0.0

            decision["PTI_post"] = pti_pre
            decision["DSCR_post"] = dscr_pre

            capital_bloq = base_rwa * ratio_total
            finance_reason_steps += [f"Acción final MANTENER: capital bloqueado≈{capital_bloq:,.0f}€ (RWA_pre×ratio)."]

        if decision_final == "VENDER":
            decision["pnl_realized"] = float(decision.get("pnl", 0.0) or 0.0)
            decision["capital_release_realized"] = float(decision.get("capital_release_cf", 0.0) or 0.0)
        elif decision_final == "REESTRUCTURAR":
            decision["pnl_realized"] = 0.0
            decision["capital_release_realized"] = float(decision.get("capital_liberado", 0.0) or 0.0)
        else:
            decision["pnl_realized"] = 0.0
            decision["capital_release_realized"] = 0.0

        decision["Explain_Steps"] = " → ".join(finance_reason_steps)
        decisions.append(decision)

    df_dec = pd.DataFrame(decisions)
    df_dec = enforce_schema(df_dec)

    if (not bool(getattr(cfg_inf, "persist_outputs", True))) or (out_dir is None) or (not str(out_dir).strip()):
        return df_dec, "", ""

    os.makedirs(out_dir, exist_ok=True)

    from reports.export_styled_excel import export_styled_excel  # type: ignore

    excel_name = "decisiones_explicadas.xlsx" if suffix is None else f"decisiones_{suffix}.xlsx"
    summary_name = "summary.csv" if suffix is None else f"summary_{suffix}.csv"

    excel_path = os.path.join(out_dir, excel_name)
    export_styled_excel(df_dec, excel_path)
    logger.info(f"✅ Resultados exportados a {excel_path}")

    summary_path = os.path.join(out_dir, summary_name)
    df_summary = pd.DataFrame(
        {
            "timestamp": [_now_tag()],
            "label": [cfg_inf.tag],
            "risk_posture": [cfg_inf.risk_posture],
            "bank_profile": [bank_profile.value],
            "strategy": [bank_strat.name],
            "n_loans": [len(df_dec)],
            "evamean": [float(df_dec["EVA_post"].mean()) if "EVA_post" in df_dec.columns else float(df_dec["EVA_pre"].mean())],
            "delta_eva_mean": [float(df_dec["ΔEVA"].mean()) if "ΔEVA" in df_dec.columns else 0.0],
            "capital_liberado_mean": [float(df_dec["capital_liberado"].mean()) if "capital_liberado" in df_dec.columns else 0.0],
            "pct_keep": [float((df_dec["decision_final"] == "MANTENER").mean())],
            "pct_restructure": [float((df_dec["decision_final"] == "REESTRUCTURAR").mean())],
            "pct_sell": [float((df_dec["decision_final"] == "VENDER").mean())],
            "cnt_fire_sale_blocked": [int(df_dec["Sell_Blocked"].sum()) if "Sell_Blocked" in df_dec.columns else 0],
            "cnt_risk_extremo": [int(df_dec["risk_extremo"].sum()) if "risk_extremo" in df_dec.columns else 0],
        }
    )
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"💾 Resumen agregado guardado en: {summary_path}")

    return df_dec, excel_path, summary_path


def run_inference_df(cfg_inf: InferenceConfig) -> pd.DataFrame:
    cfg_local = dataclasses.replace(cfg_inf, persist_outputs=False)
    df_dec, _, _ = _run_inference_for_posture(cfg_local, out_dir=None, suffix=None)
    return df_dec


def run_inference(cfg_inf: InferenceConfig) -> str:
    if not bool(getattr(cfg_inf, "persist_outputs", True)):
        _ = run_inference_df(cfg_inf)
        return ""

    ts = _now_tag()
    out_dir = cfg_inf.output_dir or os.path.join(REPORTS_DIR, f"inference_{ts}_{cfg_inf.tag}")
    os.makedirs(out_dir, exist_ok=True)

    _run_inference_for_posture(cfg_inf, out_dir, suffix=None)
    return out_dir


def run_inference_multi_posture(cfg_base: InferenceConfig) -> List[str]:
    if not bool(getattr(cfg_base, "persist_outputs", True)):
        for postura in ("prudencial", "balanceado", "desinversion"):
            _ = run_inference_df(dataclasses.replace(cfg_base, risk_posture=postura))
        return []

    ts = _now_tag()
    out_dir = cfg_base.output_dir or os.path.join(REPORTS_DIR, f"inference_{ts}_{cfg_base.tag}")
    os.makedirs(out_dir, exist_ok=True)

    outputs: List[str] = []
    escenarios = [
        ("prudencial", "policy_prudencial"),
        ("balanceado", "policy_balanceado"),
        ("desinversion", "policy_desinversion"),
    ]

    for postura, tag in escenarios:
        cfg_inf = dataclasses.replace(cfg_base, risk_posture=postura, tag=tag, persist_outputs=True)
        logger.info(f"▶ Ejecutando inferencia (MICRO) — postura={postura}…")
        _, excel_path, _ = _run_inference_for_posture(cfg_inf, out_dir, suffix=postura)
        outputs.append(excel_path)

    return outputs


def parse_args() -> InferenceConfig:
    p = argparse.ArgumentParser(description="Inferencia de política PPO (STD Basilea III · Banco L1.5)")
    p.add_argument("--model", type=str, default=os.path.join(MODELS_DIR, "best_model.zip"))
    p.add_argument("--portfolio", type=str, default=os.path.join(ROOT_DIR, "data", "portfolio_synth.xlsx"))
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--deterministic", dest="deterministic", action="store_true", help="Forzar predicción determinista.")
    p.add_argument("--stochastic", dest="deterministic", action="store_false", help="Permitir predicción estocástica.")
    p.set_defaults(deterministic=True)

    p.add_argument("--tag", type=str, default="policy")
    p.add_argument(
        "--risk-posture",
        type=str,
        choices=["prudencial", "balanceado", "desinversion"],
        default="balanceado",
        help="Perfil de riesgo para esta inferencia (por defecto: balanceado)",
    )

    p.add_argument("--vn", type=str, default="", help="Path VecNormalize LOAN (opcional).")
    p.add_argument("--no-persist", action="store_true", default=False, help="No exporta ni crea carpetas (solo cálculo).")
    p.add_argument("--out-dir", type=str, default="", help="Carpeta de salida (solo si persist).")

    a = p.parse_args()

    vn = a.vn.strip() or None
    if vn and not os.path.exists(vn):
        logger.warning(f"⚠️ --vn indicado pero no existe: {vn}. Se ignorará.")
        vn = None

    out_dir = a.out_dir.strip() or None

    return InferenceConfig(
        model_path=a.model,
        portfolio_path=a.portfolio,
        device=a.device,
        seed=a.seed,
        deterministic=bool(a.deterministic),
        tag=a.tag,
        vecnormalize_path=vn,
        risk_posture=a.risk_posture,
        persist_outputs=(not bool(a.no_persist)),
        output_dir=out_dir,
    )


if __name__ == "__main__":
    cfg_inf = parse_args()
    out_dir = run_inference(cfg_inf)
    if out_dir:
        logger.info(f"🏁 Inferencia completada. Reporte: {out_dir}")
    else:
        logger.info("🏁 Inferencia completada (no-persist).")
