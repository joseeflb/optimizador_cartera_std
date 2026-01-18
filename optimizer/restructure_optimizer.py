# ============================================
# optimizer/restructure_optimizer.py — Optimizador híbrido (local + global)
# v2.2 (BANK-READY · gates por segmento · cure no laxo · venta vs book coherente)
#
# Cambios clave:
#  - ✅ Gates duros por segmento:
#       * Retail/Mortgage/Consumer: PTI <= esfuerzo_max (obligatorio)
#       * Corporate/SME/Bank/Sov/Other: DSCR >= dscr_min (obligatorio)
#    (si falta la métrica necesaria -> NO se propone reestructura, evita “cheat code”)
#  - ✅ PD interpretada como forward al horizonte (SENS.horizon_months), hazard consistente
#  - ✅ Cure logic más bankable: RW baja SOLO si cured y tras cure_window_months
#  - ✅ Auditoría: before/after + capital_release estimado por cambio EAD/RW
#  - ✅ Venta: usa simulate_npl_price con book_value/coverage_rate; campos coherentes con price_simulator v2.x
# ============================================

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from optimizer.price_simulator import simulate_npl_price

logger = logging.getLogger("restructure_optimizer")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

HURDLE = float(cfg.CONFIG.regulacion.hurdle_rate)
RESTR = cfg.CONFIG.reestructura
SENS = cfg.CONFIG.sensibilidad_reestructura

# coherencia con generator/env
COST_FUND_ANNUAL = 0.006

try:
    REWARD = cfg.CONFIG.reward_params
except AttributeError:
    REWARD = getattr(cfg.CONFIG, "reward", cfg.CONFIG)

REG = cfg.CONFIG.regulacion
if hasattr(REG, "required_total_capital_ratio") and callable(REG.required_total_capital_ratio):
    CAP_RATIO = float(REG.required_total_capital_ratio())
else:
    base = float(getattr(REG, "total_capital_min", 0.08))
    buf = getattr(getattr(REG, "buffers", None), "total_buffer", lambda: 0.0)()
    CAP_RATIO = float(base + buf)


# -----------------------------
# utilidades numéricas robustas
# -----------------------------
def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is None


def _to_float_or_nan(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _norm_str(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
         .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    return s


def _parse_segment(segment: Any) -> cfg.Segmento:
    if isinstance(segment, cfg.Segmento):
        return segment
    s = _norm_str(segment)
    if s in {"corporate", "corp", "large corporate", "large_corporate", "largecorporate"}:
        return cfg.Segmento.CORPORATE
    if s in {"sme", "pyme", "pymes", "midcap", "mid cap", "mid-cap"}:
        return cfg.Segmento.SME
    if "mortgage" in s or "hipotec" in s:
        return cfg.Segmento.MORTGAGE
    if s in {"retail", "minorista"}:
        return cfg.Segmento.RETAIL
    if s in {"consumer", "consumo"}:
        return cfg.Segmento.CONSUMER
    if "bank" in s or s == "banco":
        return cfg.Segmento.BANK
    if "sovereign" in s or "soberan" in s:
        return cfg.Segmento.SOVEREIGN
    if "leasing" in s:
        return cfg.Segmento.LEASING
    if "project" in s and "finance" in s:
        return cfg.Segmento.OTHER
    try:
        return cfg.Segmento[str(segment).upper()]
    except Exception:
        return cfg.Segmento.CORPORATE


def _segment_is_retail(seg: cfg.Segmento) -> bool:
    return seg in {cfg.Segmento.RETAIL, cfg.Segmento.MORTGAGE, cfg.Segmento.CONSUMER}


def _segment_is_corporate(seg: cfg.Segmento) -> bool:
    return seg in {
        cfg.Segmento.CORPORATE,
        cfg.Segmento.SME,
        cfg.Segmento.BANK,
        cfg.Segmento.SOVEREIGN,
        cfg.Segmento.LEASING,
        cfg.Segmento.OTHER,
    }


def _coerce_rw(rw: Any) -> float:
    try:
        x = float(rw)
    except Exception:
        return float("nan")
    if x > 10.0:
        x = x / 100.0
    return float(x)


def _rw_default_discrete(seg: cfg.Segmento, rating: str = "BBB") -> float:
    """
    Consistente con el resto del pipeline: DEFAULT => 1.00/1.50 discreto.
    """
    try:
        rw_engine = cfg.CONFIG.basel_map.resolve_rw(
            segmento=seg,
            rating=str(rating or "BBB").upper(),
            estado=cfg.EstadoCredito.DEFAULT,
            secured_by_mortgage=(seg == cfg.Segmento.MORTGAGE),
        )
        rw_engine = _coerce_rw(rw_engine)
        if np.isfinite(rw_engine):
            return 1.00 if rw_engine < 1.25 else 1.50
    except Exception:
        pass
    return 1.00 if seg == cfg.Segmento.MORTGAGE else 1.50


# -----------------------------
# PD forward al horizonte (hazard mensual consistente)
# -----------------------------
def hazard_from_pd_forward(pd_horizon: float, horizon_months: int) -> float:
    """
    Interpreta PD como probabilidad acumulada de default en el horizonte (meses).
    Devuelve hazard mensual constante: PD = 1 - exp(-h * horizon_months).
    """
    horizon_months = int(max(1, horizon_months))
    pd_horizon = float(np.clip(pd_horizon, 1e-9, 0.999))
    return -math.log(1.0 - pd_horizon) / float(horizon_months)


def pd_forward_from_hazard(h: float, horizon_months: int) -> float:
    horizon_months = int(max(1, horizon_months))
    return float(1.0 - math.exp(-float(h) * float(horizon_months)))


def cuota_anualidad(P: float, tasa_anual: float, n_meses: int) -> float:
    i_m = tasa_anual / 12.0
    if n_meses <= 0:
        return 0.0
    if abs(i_m) < 1e-12:
        return P / n_meses
    return P * (i_m * (1.0 + i_m) ** n_meses) / ((1.0 + i_m) ** n_meses - 1.0)


@dataclass
class RestructureResult:
    plazo_optimo: int
    tasa_nueva: float
    quita: float
    EVA_post: float
    RWA_post: float
    RORWA_post: float
    EVA_gain: float
    RORWA_gain: float
    ok: bool
    msg: str


def compute_eva_base(
    ead: float,
    rate: float,
    pd: float,
    lgd: float,
    rw: float,
    hurdle: float = HURDLE,
    horizon_months: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    EVA base coherente con PD forward al horizonte.
    Modelo mínimo:
      - saldo constante (default, sin amortización)
      - EL mensual vía hazard derivado de PD(horizonte)
      - coste de funding mensual
      - cargo de capital mensual: (hurdle/12)*RWA
    """
    horizon = int(horizon_months or int(getattr(SENS, "horizon_months", 24)))
    horizon = max(1, horizon)

    ead = float(ead)
    rate = float(rate)
    pd = float(pd)
    lgd = float(lgd)
    rw = float(rw)

    i_m = rate / 12.0
    h_m = hazard_from_pd_forward(pd, horizon)  # PD forward al horizonte
    p_m = 1.0 - math.exp(-h_m)

    eva_m: List[float] = []
    rwa_m: List[float] = []
    el_m: List[float] = []
    ni_m: List[float] = []

    saldo = ead
    for _ in range(horizon):
        interes = saldo * i_m
        fund = saldo * (COST_FUND_ANNUAL / 12.0)
        el = p_m * lgd * saldo

        rwa = saldo * rw
        cap_charge = (hurdle / 12.0) * rwa

        ni = interes - fund - el
        eva_m.append(ni - cap_charge)

        rwa_m.append(rwa)
        el_m.append(el)
        ni_m.append(ni)

    eva_total = float(np.sum(eva_m))
    rwa_avg = float(np.mean(rwa_m)) if rwa_m else 0.0
    el_total = float(np.sum(el_m))

    years = horizon / 12.0
    net_income_annual = (float(np.sum(ni_m)) / years) if years > 0 else 0.0
    rorwa_annual = (net_income_annual / rwa_avg) if rwa_avg > 0 else 0.0

    return eva_total, float(rorwa_annual), float(rwa_avg), float(el_total)


def optimize_restructure(
    *,
    ead: float,
    rate: float,
    pd: float,
    lgd: float,
    hurdle: float = HURDLE,
    rw: float | None = None,
    segment: Optional[str] = None,
    rating: str = "BBB",

    # Compat:
    ingreso_mensual: float | None = None,                 # retail/mortgage
    cashflow_operativo_mensual: float | None = None,      # corporate
    dscr_min: float = 1.10,
    permite_quita: bool = True,
    max_quita: float = 0.15,
    esfuerzo_max: float = 0.40,
    **_: Any,
) -> Dict[str, Any]:
    horizon = max(1, int(getattr(SENS, "horizon_months", 24)))

    seg_enum = _parse_segment(segment)
    rating_u = str(rating or "BBB").upper()

    # RW discreto si no lo pasan (consistente con pipeline)
    if rw is None or not np.isfinite(_coerce_rw(rw)):
        rw = _rw_default_discrete(seg_enum, rating_u)
    rw = float(rw)

    eva_base, rorwa_base, rwa_base, el_base = compute_eva_base(
        ead=ead,
        rate=rate,
        pd=pd,
        lgd=lgd,
        rw=rw,
        hurdle=hurdle,
        horizon_months=horizon,
    )

    # Viabilidad por segmento: requerimos data real (no proxy) para gates duros
    is_retail = _segment_is_retail(seg_enum)
    is_corp = _segment_is_corporate(seg_enum)

    ingreso_ok = (ingreso_mensual is not None) and (not _is_nan(ingreso_mensual)) and (float(ingreso_mensual) > 0)
    cfo_ok = (cashflow_operativo_mensual is not None) and (not _is_nan(cashflow_operativo_mensual)) and (float(cashflow_operativo_mensual) > 0)

    if is_retail and (not ingreso_ok):
        return {
            "ok": False,
            "msg": "sin income -> retail no viable",
            "segment": seg_enum.name,
            "EVA_base": float(eva_base),
            "RORWA_base": float(rorwa_base),
            "RWA_base": float(rwa_base),
            "EL_base_horizon": float(el_base),
        }

    if is_corp and (not cfo_ok):
        return {
            "ok": False,
            "msg": "sin CFO -> corporate no viable",
            "segment": seg_enum.name,
            "EVA_base": float(eva_base),
            "RORWA_base": float(rorwa_base),
            "RWA_base": float(rwa_base),
            "EL_base_horizon": float(el_base),
        }

    ingreso = float(ingreso_mensual) if ingreso_ok else float("nan")
    cfo_m = float(cashflow_operativo_mensual) if cfo_ok else float("nan")

    best = RestructureResult(
        plazo_optimo=0,
        tasa_nueva=rate,
        quita=0.0,
        EVA_post=eva_base,
        RWA_post=rwa_base,
        RORWA_post=rorwa_base,
        EVA_gain=0.0,
        RORWA_gain=0.0,
        ok=False,
        msg="sin mejora",
    )
    best_extra: Dict[str, Any] = {}

    # Admin/quita costs (one-off)
    admin_cost_abs = float(getattr(REWARD, "restructure_admin_cost_abs", 0.0))
    quita_cost_bps = float(getattr(REWARD, "restructure_cost_quita_bps", 0.0))

    rw_perf_guess = float(getattr(SENS, "rw_perf_guess", 1.0))
    pd_cure_th = float(getattr(SENS, "pd_cure_threshold", 0.20))
    cure_window = int(max(0, int(getattr(SENS, "cure_window_months", 3))))

    for plazo_anios in RESTR.plazo_anios_grid:
        n_meses = int(plazo_anios * 12)

        for tasa in RESTR.tasa_anual_grid:
            for quita in (RESTR.quita_grid if permite_quita else [0.0]):
                if quita > max_quita:
                    continue

                ead_n = float(ead * (1.0 - quita))
                if ead_n <= 0:
                    continue

                # Sensibilidades PD/LGD (PD forward al horizonte)
                lgd_n = float(np.clip(lgd * (1.0 - SENS.lgd_reduction_per_quita * quita), 0.0, 1.0))

                # PD post: reducimos hazard mensual para que PD_forward(horizon) baje de forma consistente
                h0 = hazard_from_pd_forward(pd, horizon)
                h_post = h0 * (1.0 - SENS.pd_reduction_per_year * plazo_anios) * (1.0 - SENS.pd_reduction_per_quita * quita)
                h_post = float(max(h_post, 1e-12))
                pd_post = float(np.clip(pd_forward_from_hazard(h_post, horizon), 1e-9, 0.999))

                cuota = float(cuota_anualidad(ead_n, tasa, n_meses))

                # -------------------------
                # Gates duros por segmento
                # -------------------------
                pti = float("nan")
                dscr = float("nan")

                if is_retail:
                    pti = float(cuota / max(1e-6, ingreso))
                    if pti > esfuerzo_max:
                        continue

                if is_corp:
                    dscr = float(cfo_m / max(1e-6, cuota))
                    if dscr < dscr_min:
                        continue

                # Si es retail y además hay CFO (raro), o corp con income, lo dejamos como info, no gate
                if not is_retail and ingreso_ok:
                    pti = float(cuota / max(1e-6, ingreso))
                if not is_corp and cfo_ok:
                    dscr = float(cfo_m / max(1e-6, cuota))

                # -------------------------
                # Simulación de EVA mensual (con amortización)
                # -------------------------
                saldo = float(ead_n)
                i_m = float(tasa / 12.0)

                h_m = float(h_post)  # hazard mensual post (constante)
                p_m = 1.0 - math.exp(-h_m)

                eva_m: List[float] = []
                rwa_m: List[float] = []
                ni_m: List[float] = []
                el_m: List[float] = []

                cured = bool(pd_post < pd_cure_th)
                current_rw = float(rw)

                for m in range(1, horizon + 1):
                    if saldo <= 1e-8:
                        eva_m.append(0.0)
                        rwa_m.append(0.0)
                        ni_m.append(0.0)
                        el_m.append(0.0)
                        continue

                    interes = saldo * i_m
                    fund = saldo * (COST_FUND_ANNUAL / 12.0)

                    # Política de pagos: ventana de cura con interés-only para no “curar” por magia
                    pago = cuota if m > cure_window else interes
                    amort = max(pago - interes, 0.0)
                    saldo = max(saldo - amort, 0.0)

                    el = p_m * lgd_n * saldo
                    el_m.append(el)

                    # RW baja solo si cured y tras ventana de cura
                    if cured and (m > cure_window) and (rw_perf_guess > 0):
                        current_rw = float(rw_perf_guess)

                    rwa = saldo * current_rw
                    rwa_m.append(rwa)

                    cap_charge = (hurdle / 12.0) * rwa
                    ni = interes - fund - el
                    ni_m.append(ni)
                    eva_m.append(ni - cap_charge)

                rwa_avg = float(np.mean(rwa_m)) if rwa_m else 0.0

                # costes one-off
                quita_cost = (quita_cost_bps / 10_000.0) * (quita * float(ead))
                eva_total = float(np.sum(eva_m)) - admin_cost_abs - quita_cost

                years = horizon / 12.0
                net_income_total = float(np.sum(ni_m)) - admin_cost_abs - quita_cost
                net_income_annual = (net_income_total / years) if years > 0 else 0.0
                rorwa_annual = (net_income_annual / rwa_avg) if rwa_avg > 0 else 0.0

                if eva_total > best.EVA_post:
                    # capital release aproximado (instantáneo) por cambio EAD/RW en cured/no-cured
                    rw_end = float(rw_perf_guess) if cured else float(rw)
                    rwa0_inst = float(ead * rw)
                    rwa1_inst = float(ead_n * rw_end)
                    capital_release_est = float(max(0.0, (rwa0_inst - rwa1_inst) * CAP_RATIO))

                    best = RestructureResult(
                        plazo_optimo=int(n_meses),
                        tasa_nueva=float(tasa),
                        quita=float(quita),
                        EVA_post=float(eva_total),
                        RWA_post=float(rwa_avg),
                        RORWA_post=float(rorwa_annual),
                        EVA_gain=float(eva_total - eva_base),
                        RORWA_gain=float(rorwa_annual - rorwa_base),
                        ok=bool(eva_total > eva_base),
                        msg="OK" if eva_total > eva_base else "sin mejora",
                    )
                    best_extra = {
                        "segment": seg_enum.name,
                        "rating": rating_u,
                        "cuota_mensual_post": float(cuota),
                        "PTI_post": None if math.isnan(pti) else float(pti),
                        "DSCR_post": None if math.isnan(dscr) else float(dscr),
                        "EL_total_horizon": float(np.sum(el_m)),
                        "cured": bool(cured),
                        "PD_post": float(pd_post),
                        "LGD_post": float(lgd_n),
                        "EAD_post": float(ead_n),
                        "RW_post_end": float(rw_end),
                        "admin_cost": float(admin_cost_abs),
                        "quita_cost": float(quita_cost),
                        "capital_release_est": float(capital_release_est),
                        "horizon_months": int(horizon),
                        "gate_type": "PTI" if is_retail else "DSCR",
                        "gate_params": {
                            "esfuerzo_max": float(esfuerzo_max),
                            "dscr_min": float(dscr_min),
                            "cure_window_months": int(cure_window),
                            "pd_cure_threshold": float(pd_cure_th),
                        },
                    }

    out = best.__dict__
    out.update(
        best_extra
        or {
            "segment": seg_enum.name,
            "rating": rating_u,
            "cuota_mensual_post": None,
            "PTI_post": None,
            "DSCR_post": None,
            "EL_total_horizon": None,
            "cured": False,
            "PD_post": None,
            "LGD_post": None,
            "EAD_post": None,
            "RW_post_end": None,
            "admin_cost": float(admin_cost_abs),
            "quita_cost": None,
            "capital_release_est": 0.0,
            "horizon_months": int(horizon),
            "gate_type": "PTI" if is_retail else "DSCR",
            "gate_params": {"esfuerzo_max": float(esfuerzo_max), "dscr_min": float(dscr_min)},
        }
    )

    out["EVA_base"] = float(eva_base)
    out["RORWA_base"] = float(rorwa_base)
    out["RWA_base"] = float(rwa_base)
    out["EL_base_horizon"] = float(el_base)
    out["msg"] = "OK (mejora significativa)" if out.get("ok") else "sin mejora"

    return out


def _evaluate_sale_prudential(
    *,
    ead: float,
    pd: float,
    lgd: float,
    rw: float,
    segment: Optional[str] = None,
    rating: str = "BBB",
    dpd: Optional[float] = None,
    # contable
    book_value: Optional[float] = None,
    coverage_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Venta: usa simulate_npl_price (bank-ready) con book_value/coverage_rate.
    Penalización fire-sale basada en Price/Book (no por Price/EAD).
    """
    price = simulate_npl_price(
        ead=ead,
        lgd=lgd,
        pd=pd,
        dpd=dpd,
        segment=segment,
        rating=rating,
        secured=bool(_parse_segment(segment) == cfg.Segmento.MORTGAGE),
        book_value=book_value,
        coverage_rate=coverage_rate,
    )

    if not price.get("ok", False):
        return {"ok": False, "msg": "price_simulator failed", "sim_raw": price}

    precio_neto = float(price.get("precio_optimo", 0.0))

    # --- Book usado realmente por el simulador (fuente única)
    book_used = price.get("book_value", None)
    if book_used is None or _is_nan(book_used) or float(book_used) <= 0:
        # fallback coherente si por algún motivo no viniera
        if coverage_rate is not None and (not _is_nan(coverage_rate)) and float(ead) > 0:
            cr = float(coverage_rate)
            if cr > 1.0:
                cr = cr / 100.0
            cr = float(np.clip(cr, 0.0, 1.0))
            book_used = float(ead) * (1.0 - cr)
        else:
            book_used = float(ead) * (1.0 - float(np.clip(lgd, 0.0, 1.0)))
    book_used = float(max(book_used, 1e-9))

    # pnl contable vs ese book_used (el simulador ya devuelve pnl vs book; si no, lo derivamos)
    pnl_book = float(price.get("pnl", precio_neto - book_used))

    # --- Price/Book ratio (no confundir con Price/EAD)
    ratio_pb = price.get("price_ratio_book", None)
    if ratio_pb is None or (isinstance(ratio_pb, float) and not np.isfinite(ratio_pb)):
        ratio_pb = float(precio_neto / book_used) if book_used > 0 else float("nan")
    ratio_pb = float(ratio_pb) if np.isfinite(ratio_pb) else float("nan")

    # --- Threshold book-aware: NUNCA default 0.08
    thr_pb = price.get("fire_sale_threshold_book", None)
    if thr_pb is None or (isinstance(thr_pb, float) and not np.isfinite(thr_pb)):
        thr_pb = float(getattr(cfg.CONFIG.precio_venta, "fire_sale_price_ratio_book", 0.85))
    thr_pb = float(thr_pb)

    fire_sale = bool(price.get("fire_sale", False))
    # Si por alguna razón no viene el flag, lo derivamos con ratio_pb
    if (not fire_sale) and np.isfinite(ratio_pb):
        fire_sale = bool(ratio_pb < thr_pb)

    gap_pb = float(max(0.0, thr_pb - ratio_pb)) if np.isfinite(ratio_pb) else 0.0

    penalty_fire_sale_w = float(getattr(REWARD, "penalty_fire_sale", 0.0))
    fire_sale_penalty_eur = float(penalty_fire_sale_w * float(ead) * gap_pb)

    eva_post_sale_adj = float(pnl_book - fire_sale_penalty_eur)

    # Capital liberado: viene del simulador (RW discreto coherente).
    capital_liberado = float(price.get("capital_liberado", float(ead) * float(rw) * CAP_RATIO))

    return {
        "ok": True,
        "precio_optimo": float(precio_neto),

        # ratios
        "price_ratio_ead": float(price.get("price_ratio_ead", (precio_neto / ead) if ead > 0 else 0.0)),
        "price_ratio_book": float(ratio_pb),

        # fire-sale
        "fire_sale_threshold_book": float(thr_pb),
        "fire_sale": bool(fire_sale),

        # contable
        "book_value": float(book_used),
        "pnl": float(pnl_book),          # ✅ alias principal (pipeline)
        "pnl_book": float(pnl_book),     # ✅ legacy/audit

        # penalización
        "fire_sale_penalty_eur": float(fire_sale_penalty_eur),
        "EVA_post_sale": float(pnl_book),
        "EVA_post_sale_adj": float(eva_post_sale_adj),

        # capital
        "capital_liberado": float(capital_liberado),         # ✅ alias principal
        "capital_liberado_venta": float(capital_liberado),   # ✅ legacy

        # percentiles si vienen
        "resumen": price.get("resumen", {}) or {},

        "sim_raw": price,
    }



def optimize_heuristic(
    df: pd.DataFrame,
    budget: float = 0.0,
    hurdle: float = HURDLE,
    esfuerzo_max: float = 0.40,
    dscr_min: float = 1.10,
) -> pd.DataFrame:
    logger.info("🧮 Ejecutando heurístico global Banco L1.5 (con lógica NPL)…")
    results: List[Dict[str, Any]] = []

    EVA_MIN_IMPROVEMENT_EUR = float(getattr(REWARD, "eva_min_improvement", 10_000.0))

    for row in df.itertuples(index=False):
        try:
            ead = float(getattr(row, "EAD"))
            rate = float(getattr(row, "rate"))
            pd_val = float(getattr(row, "PD"))
            lgd = float(getattr(row, "LGD"))
            seg = getattr(row, "segment", getattr(row, "segment_raw", "corporate"))
            rating = getattr(row, "rating", "BBB")
            dpd = float(getattr(row, "DPD", 180.0))
            loan_id = getattr(row, "loan_id", None)

            # RW (si falta, derivarlo discreto por segmento)
            rw_in = getattr(row, "RW", None)
            seg_enum = _parse_segment(seg)
            rw = _coerce_rw(rw_in)
            if not np.isfinite(rw):
                rw = _rw_default_discrete(seg_enum, str(rating or "BBB"))
            else:
                rw = 1.00 if rw < 1.25 else 1.50
            rw = float(rw)

            # -----------------------------
            # compatibilidad con generator
            # -----------------------------
            monthly_payment_pre = getattr(row, "monthly_payment", None)
            monthly_income = getattr(row, "monthly_income", None)
            monthly_cfo = getattr(row, "monthly_cfo", None)
            book_value = getattr(row, "book_value", None)
            coverage_rate = getattr(row, "coverage_rate", getattr(row, "coverage", None))

            # legacy (datasets antiguos)
            ingreso_mensual_legacy = getattr(row, "ingreso_mensual", None)
            cfo_m_legacy = getattr(row, "cashflow_operativo_mensual", None)

            ingreso_mensual = monthly_income if not _is_nan(monthly_income) else ingreso_mensual_legacy
            cfo_m = monthly_cfo if not _is_nan(monthly_cfo) else cfo_m_legacy

            cuota_pre = _to_float_or_nan(monthly_payment_pre)

            # PTI_pre / DSCR_pre (si tengo data)
            pti_pre = float("nan")
            dscr_pre = float("nan")
            if not math.isnan(cuota_pre) and cuota_pre > 0:
                if ingreso_mensual is not None and not _is_nan(ingreso_mensual):
                    pti_pre = float(cuota_pre / max(1e-6, float(ingreso_mensual)))
                if cfo_m is not None and not _is_nan(cfo_m):
                    dscr_pre = float(float(cfo_m) / max(1e-6, cuota_pre))

            eva_base, rorwa_base, rwa_base, _ = compute_eva_base(
                ead=ead,
                rate=rate,
                pd=pd_val,
                lgd=lgd,
                rw=rw,
                hurdle=hurdle,
                horizon_months=int(getattr(SENS, "horizon_months", 24)),
            )

            res_restr = optimize_restructure(
                ead=ead,
                rate=rate,
                pd=pd_val,
                lgd=lgd,
                rw=rw,
                segment=str(seg),
                rating=str(rating),
                ingreso_mensual=None if _is_nan(ingreso_mensual) else float(ingreso_mensual),
                cashflow_operativo_mensual=None if _is_nan(cfo_m) else float(cfo_m),
                esfuerzo_max=esfuerzo_max,
                dscr_min=dscr_min,
                hurdle=hurdle,
            )

            eva_restr = float(res_restr.get("EVA_post", -1e18))
            delta_eva_restr = float(res_restr.get("EVA_gain", 0.0))

            coste_admin = float(getattr(REWARD, "restructure_admin_cost_abs", 0.0))
            coste_quita = (float(getattr(REWARD, "restructure_cost_quita_bps", 0.0)) / 10_000.0) * (
                float(res_restr.get("quita", 0.0)) * ead
            )
            coste_total = float(coste_admin + coste_quita)

            sale_pack = None
            eva_sale = None
            eva_sale_adj = None
            delta_eva_sale_adj = None

            # Venta sólo se evalúa cuando EVA base es negativa (heurística)
            if eva_base < 0.0:
                sale_pack = _evaluate_sale_prudential(
                    ead=ead,
                    pd=pd_val,
                    lgd=lgd,
                    rw=rw,
                    segment=str(seg),
                    rating=str(rating),
                    dpd=dpd,
                    book_value=None if _is_nan(book_value) else float(book_value),
                    coverage_rate=None if _is_nan(coverage_rate) else float(coverage_rate),
                )
                if sale_pack.get("ok", False):
                    eva_sale = float(sale_pack["EVA_post_sale"])
                    eva_sale_adj = float(sale_pack["EVA_post_sale_adj"])
                    delta_eva_sale_adj = float(eva_sale_adj - eva_base)

            # decisión local
            strategy = "MANTENER"
            eva_choice = float(eva_base)
            delta_eva_choice = 0.0

            if eva_base >= 0.0:
                if res_restr.get("ok") and delta_eva_restr > EVA_MIN_IMPROVEMENT_EUR:
                    strategy = "REESTRUCTURAR"
                    eva_choice = float(eva_restr)
                    delta_eva_choice = float(delta_eva_restr)
            else:
                if eva_sale_adj is not None:
                    if res_restr.get("ok") and float(eva_restr) > float(eva_sale_adj):
                        strategy = "REESTRUCTURAR"
                        eva_choice = float(eva_restr)
                        delta_eva_choice = float(delta_eva_restr)
                    else:
                        strategy = "VENDER"
                        eva_choice = float(eva_sale_adj)
                        delta_eva_choice = float(delta_eva_sale_adj or 0.0)
                else:
                    if res_restr.get("ok") and float(delta_eva_restr) > 0:
                        strategy = "REESTRUCTURAR"
                        eva_choice = float(eva_restr)
                        delta_eva_choice = float(delta_eva_restr)

            eff_ratio = float("-inf")
            if strategy == "REESTRUCTURAR" and coste_total > 0 and delta_eva_restr > 0:
                eff_ratio = float(delta_eva_restr / coste_total)

            out_row: Dict[str, Any] = dict(res_restr)

            # enriquecimiento audit: pre/post coherente + campos de entrada
            out_row.update(
                {
                    "loan_id": loan_id,

                    # inputs viabilidad
                    "monthly_payment_pre": None if math.isnan(cuota_pre) else float(cuota_pre),
                    "monthly_income": None if _is_nan(ingreso_mensual) else float(ingreso_mensual),
                    "monthly_cfo": None if _is_nan(cfo_m) else float(cfo_m),
                    "PTI_pre": None if math.isnan(pti_pre) else float(pti_pre),
                    "DSCR_pre": None if math.isnan(dscr_pre) else float(dscr_pre),

                    # contable
                    "book_value": None if _is_nan(book_value) else float(book_value),
                    "coverage_rate": None if _is_nan(coverage_rate) else float(coverage_rate),

                    # base
                    "EVA_base": float(eva_base),
                    "RORWA_base": float(rorwa_base),
                    "RWA_base": float(rwa_base),

                    # candidatos
                    "EVA_restr": float(eva_restr) if np.isfinite(eva_restr) else None,
                    "ΔEVA_restr": float(delta_eva_restr),
                    "EVA_sale": eva_sale,
                    "EVA_sale_adj": eva_sale_adj,
                    "ΔEVA_sale_adj": delta_eva_sale_adj,

                    # decisión
                    "strategy_finance": strategy,
                    "EVA_choice": float(eva_choice),
                    "ΔEVA_choice": float(delta_eva_choice),
                    "coste_total": float(coste_total),
                    "eff_ratio": float(eff_ratio),

                    # venta: pack completo
                    "sale_pack": sale_pack,
                }
            )

            results.append(out_row)

        except Exception as e:
            logger.warning(f"⚠️ Error optimizando préstamo en heurístico: {e}")
            continue

    out = pd.DataFrame(results)

    if not budget or budget <= 0:
        out["chosen"] = ((out["strategy_finance"] == "REESTRUCTURAR") & (out["ΔEVA_choice"] > 0)).astype(int)
        return out

    out_restr = out[out["strategy_finance"] == "REESTRUCTURAR"].copy()
    out_restr = out_restr.sort_values("eff_ratio", ascending=False).reset_index(drop=True)

    chosen_mask = np.zeros(len(out), dtype=int)
    spent = 0.0

    # selección global por budget (greedy por eficiencia)
    for _, r in out_restr.iterrows():
        if float(r.get("ΔEVA_choice", 0.0)) <= 0:
            continue
        coste_i = float(r.get("coste_total", 0.0))
        if coste_i <= 0:
            continue
        if spent + coste_i <= budget:
            spent += coste_i
            # marca por loan_id si existe; si no, cae al índice
            lid = r.get("loan_id", None)
            if lid is not None:
                idxs = out.index[out["loan_id"] == lid]
                if len(idxs) > 0:
                    chosen_mask[out.index.get_loc(idxs[0])] = 1

    out["chosen"] = chosen_mask
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimizador de reestructuración (heurístico Banco L1.5)")
    ap.add_argument("--input", required=True, help="CSV o Excel de préstamos")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--out", default="reports/restructured_out.csv")
    args = ap.parse_args()

    df = pd.read_excel(args.input) if args.input.lower().endswith(".xlsx") else pd.read_csv(args.input)
    df_out = optimize_heuristic(df, budget=args.budget)
    df_out.to_csv(args.out, index=False)
    logger.info(f"💾 Resultados guardados en {args.out}")


if __name__ == "__main__":
    main()
