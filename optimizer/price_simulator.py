# ============================================
# optimizer/price_simulator.py — Simulador de precio de venta (mercado secundario NPL)
# v2.2 (bank-ready · anti-arbitraje · coherente con book & capital release)
# ============================================
from __future__ import annotations

import math
import logging
from enum import Enum, auto
from typing import Dict, Optional, Tuple, Any, List

import numpy as np

import config as cfg

logger = logging.getLogger("price_simulator")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class MarketScenario(Enum):
    BULL = auto()
    BASE = auto()
    STRESS = auto()


class BuyerType(Enum):
    SPECIAL_SITUATIONS = auto()
    SERVICER = auto()
    BANK = auto()
    PRIVATE_EQUITY = auto()


CONFIG = cfg.CONFIG.precio_venta
REG = cfg.CONFIG.regulacion

REWARD = getattr(cfg.CONFIG, "reward", None)
if REWARD is None:
    REWARD = getattr(cfg.CONFIG, "reward_params", cfg.CONFIG)

# CAP_RATIO robusto (mismo criterio que otros módulos)
if hasattr(REG, "required_total_capital_ratio") and callable(REG.required_total_capital_ratio):
    _CAP_RATIO = float(REG.required_total_capital_ratio())
else:
    base = float(getattr(REG, "total_capital_min", 0.08))
    buf = getattr(getattr(REG, "buffers", None), "total_buffer", lambda: 0.0)()
    _CAP_RATIO = float(base + buf)


# -------------------------
# Helpers robustos
# -------------------------
def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(cfg.GLOBAL_SEED if seed is None else seed)


def _norm_str(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
         .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    return s


def _is_na(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(np.isnan(float(x)))
    except Exception:
        return False


def _coerce_rw(rw: Any) -> float:
    """Normaliza RW a multiplicador: acepta 1.5 o 150."""
    try:
        x = float(rw)
    except Exception:
        return float("nan")
    if x > 10.0:
        x = x / 100.0
    return float(x)


def _rw_default_discrete(seg_enum: cfg.Segmento, rating: str) -> float:
    """
    Discretización coherente con LoanEnv/PortfolioEnv:
      DEFAULT => RW discreto 1.00 o 1.50 (sin valores intermedios)
    """
    try:
        rw_engine = cfg.CONFIG.basel_map.resolve_rw(
            segmento=seg_enum,
            rating=str(rating or "BBB").upper(),
            estado=cfg.EstadoCredito.DEFAULT,
            secured_by_mortgage=(seg_enum == cfg.Segmento.MORTGAGE),
        )
        rw_engine = _coerce_rw(rw_engine)
        if np.isfinite(rw_engine):
            return 1.00 if rw_engine < 1.25 else 1.50
    except Exception:
        pass

    # fallback conservador
    return 1.00 if seg_enum == cfg.Segmento.MORTGAGE else 1.50


def _parse_segment(segment: Any) -> cfg.Segmento:
    """
    Mapea etiquetas legacy del generador / Excel a Enum Segmento.
    Evita que 'Large Corporate' o 'MidCap' caigan a fallback silencioso.
    """
    if isinstance(segment, cfg.Segmento):
        return segment

    s = _norm_str(segment)
    if s in {"corporate", "corp", "large corporate", "large_corporate", "largecorporate"}:
        return cfg.Segmento.CORPORATE
    if s in {"midcap", "mid cap", "mid-cap"}:
        return cfg.Segmento.SME
    if s in {"sme", "pyme", "pymes"}:
        return cfg.Segmento.SME
    if "project" in s and "finance" in s:
        return cfg.Segmento.OTHER
    if "mortgage" in s or "hipotec" in s:
        return cfg.Segmento.MORTGAGE

    try:
        return cfg.Segmento[str(segment).upper()]
    except Exception:
        return cfg.Segmento.CORPORATE


def _dpd_penalty(dpd: float, a: float = 0.08) -> float:
    dpd = 0.0 if _is_na(dpd) else float(max(float(dpd), 0.0))
    return float(1.0 / (1.0 + a * math.log1p(dpd)))


def _secured_mult(secured: bool, cfgp) -> float:
    if secured:
        return 1.0
    try:
        mean_legal = float(np.mean(cfgp.coste_legal_estimado_abs))
    except Exception:
        mean_legal = 5000.0
    return float(0.85 - min(mean_legal, 20000.0) / 200000.0)


def _pd_discount(pd_prob: Optional[float]) -> float:
    if pd_prob is None or _is_na(pd_prob):
        return 1.0
    try:
        p = float(pd_prob)
    except Exception:
        return 1.0
    p = float(np.clip(p, 0.0, 1.0))
    return float(1.0 - 0.30 * p)


def _rating_mult(rating: Any) -> float:
    """
    Descuento suave por rating (para precio, no para RW).
    Mantiene continuidad y evita discontinuidades.
    """
    r = str(rating or "BBB").strip().upper()
    m = {
        "AAA": 1.00,
        "AA": 0.99,
        "A": 0.98,
        "BBB": 0.97,
        "BB": 0.95,
        "B": 0.92,
        "CCC": 0.88,
    }
    return float(m.get(r, 0.95))


def _tail_metrics(samples: np.ndarray) -> Dict[str, float]:
    if len(samples) == 0:
        return {"p5": np.nan, "p50": np.nan, "p95": np.nan}
    return {
        "p5": float(np.quantile(samples, 0.05)),
        "p50": float(np.quantile(samples, 0.50)),
        "p95": float(np.quantile(samples, 0.95)),
    }


def _scenario_recovery_multiplier(scn: MarketScenario) -> float:
    if scn == MarketScenario.BULL:
        return 1.05
    if scn == MarketScenario.STRESS:
        return 0.70
    return 1.00


def _scenario_volatility(scn: MarketScenario) -> float:
    if scn == MarketScenario.BULL:
        return 0.15
    if scn == MarketScenario.STRESS:
        return 0.30
    return 0.20


def _price_caps(ead: float, base_recovery: float, secured: bool, dpd: float) -> Tuple[float, float]:
    """
    Caps por EAD/recovery para evitar outliers y discontinuidades.
    NOTA: No capamos directamente por book para no distorsionar casos de sobre-cobertura;
          anti-arbitraje se gestiona por book_value derivado y caps suaves por fuente de book.
    """
    min_ratio = 0.03 if secured else 0.02
    min_price = min_ratio * ead

    max_ratio_ead = 0.80 if secured else 0.60
    dpd = float(max(dpd, 0.0))

    if dpd > 720:
        max_ratio_ead *= 0.65
    elif dpd > 360:
        max_ratio_ead *= 0.80
    elif dpd > 180:
        max_ratio_ead *= 0.90

    max_price_ead = max_ratio_ead * ead
    max_price_recovery = 0.90 * base_recovery

    max_price = max(0.0, min(max_price_ead, max_price_recovery))
    if max_price < min_price:
        max_price = min_price * 1.05

    return float(min_price), float(max_price)


def _derive_book_value(
    *,
    ead: float,
    lgd: float,
    pd_prob: Optional[float],
    book_value: Optional[float],
    coverage_rate: Optional[float],
) -> Tuple[float, str]:
    """
    Deriva book_value de forma conservadora y coherente:
      - Si book_value viene informado: se usa.
      - Si coverage_rate viene: book = EAD*(1-coverage).
      - Si no: proxy = EAD*(1 - clip(PD*LGD)) (IFRS9-like ECL proxy).
        Esto evita infra-estimar book (anti-arbitraje) y alinea con economía PD/LGD del RL.
    """
    if book_value is not None and not _is_na(book_value):
        bv = float(max(float(book_value), 1e-9))
        return bv, "provided"

    if coverage_rate is not None and not _is_na(coverage_rate):
        cr = float(coverage_rate)
        if cr > 1.0:
            cr = cr / 100.0
        cr = float(np.clip(cr, 0.0, 1.0))
        bv = float(max(ead * (1.0 - cr), 1e-9))
        return bv, "coverage_rate"

    # proxy por PD*LGD; si PD no viene, asumimos NPL alto (conservador)
    if pd_prob is None or _is_na(pd_prob):
        pd_proxy = 0.85
    else:
        pd_proxy = float(np.clip(float(pd_prob), 0.0, 1.0))

    cov_proxy = float(np.clip(pd_proxy * float(np.clip(lgd, 0.0, 1.0)), 0.05, 0.95))
    bv = float(max(ead * (1.0 - cov_proxy), 1e-9))
    return bv, "proxy_pd_lgd"


def simulate_npl_price(
    *,
    ead: float | None = None,
    lgd: float | None = None,
    segment: str | None = None,
    rating: str = "BBB",
    pd: float | None = None,
    dpd: float | None = None,
    secured: bool = False,
    # contable: para pnl/fire-sale vs book
    book_value: float | None = None,
    coverage_rate: float | None = None,
    escenarios: Tuple[str, ...] = ("BULL", "BASE", "STRESS"),
    pesos: Tuple[float, ...] = (0.2, 0.6, 0.2),
    n_mc: int = 3000,
    seed: Optional[int] = None,
    buyer_type: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    # -------------------------
    # Compat parse inputs
    # -------------------------
    if dpd is None:
        dpd = kwargs.get("dpd") or kwargs.get("DPD") or kwargs.get("days_past_due")
    if pd is None:
        pd = kwargs.get("pd") or kwargs.get("PD")
    if ead is None:
        ead = kwargs.get("EAD") or kwargs.get("ead")
    if lgd is None:
        lgd = kwargs.get("LGD") or kwargs.get("lgd")
    if segment is None:
        segment = (
            kwargs.get("segment")
            or kwargs.get("SEGMENT")
            or kwargs.get("segment_raw")
            or kwargs.get("segmento_banco")
        )

    if book_value is None:
        book_value = kwargs.get("book_value") or kwargs.get("BOOK_VALUE") or kwargs.get("BookValue") or kwargs.get("book")
    if coverage_rate is None:
        coverage_rate = kwargs.get("coverage_rate") or kwargs.get("COVERAGE_RATE") or kwargs.get("coverage")

    # -------------------------
    # Sanitización
    # -------------------------
    ead = float(ead or 0.0)
    lgd = float(np.clip(lgd or 0.60, 0.0, 1.0))
    dpd = float(max(float(dpd or 180.0), 0.0))

    if ead <= 0:
        return {"precio_optimo": 0.0, "ok": False, "msg": "EAD<=0"}

    base_recovery = float(ead * (1.0 - lgd))
    seg_enum = _parse_segment(segment)
    rating_u = str(rating or "BBB").strip().upper()

    # Book (robusto + anti-arbitraje si no viene informado)
    book_value, book_source = _derive_book_value(
        ead=ead,
        lgd=lgd,
        pd_prob=pd,
        book_value=book_value,
        coverage_rate=coverage_rate,
    )

    # -------------------------
    # Determinísticos (multiplicadores)
    # -------------------------
    alpha_seg = CONFIG.alpha_by_segment.get(seg_enum, (0.35, 0.55))
    alpha_mean = float(np.mean(alpha_seg))

    dpd_a = float(getattr(CONFIG, "dpd_penalty_a", getattr(CONFIG, "buyer_discount_rate", 0.08)))
    dpd_mult = _dpd_penalty(dpd, a=dpd_a)

    sec_mult = _secured_mult(bool(secured), CONFIG)
    pd_mult = _pd_discount(pd)
    rat_mult = _rating_mult(rating_u)

    det_mult = max(0.01, alpha_mean * dpd_mult * sec_mult * pd_mult * rat_mult)

    if buyer_type:
        bt = str(buyer_type).upper()
        if "SPECIAL" in bt or "SITUATION" in bt:
            det_mult *= 1.05
        elif "SERVICER" in bt:
            det_mult *= 0.95
        elif "BANK" in bt:
            det_mult *= 0.90
        elif "PRIVATE" in bt or "PE" in bt:
            det_mult *= 1.00

    # -------------------------
    # Monte Carlo por escenarios
    # -------------------------
    rng = _rng(seed)
    escenario_enum = {"BULL": MarketScenario.BULL, "BASE": MarketScenario.BASE, "STRESS": MarketScenario.STRESS}

    # Si pesos/escenarios vienen desalineados, truncamos al mínimo común
    k = min(len(escenarios), len(pesos)) if pesos else len(escenarios)
    escenarios_use = escenarios[:k]
    pesos_use = pesos[:k] if pesos else tuple([1.0 / max(1, k)] * k)

    total_weight = float(sum(pesos_use)) if pesos_use else 1.0
    min_price, max_price = _price_caps(ead, base_recovery, bool(secured), dpd)

    all_prices: List[np.ndarray] = []
    for scn_name, w in zip(escenarios_use, pesos_use):
        if w <= 0:
            continue
        scn = escenario_enum.get(str(scn_name).upper(), MarketScenario.BASE)

        scn_mu = _scenario_recovery_multiplier(scn)
        mean_price = float(base_recovery * det_mult * scn_mu)

        sigma = _scenario_volatility(scn)
        mu_shock = -0.5 * sigma**2

        n_samples = max(1, int(n_mc * (w / max(total_weight, 1e-12))))
        shocks = rng.lognormal(mean=mu_shock, sigma=sigma, size=n_samples)

        prices_scn = mean_price * shocks
        prices_scn = np.clip(prices_scn, min_price, max_price)
        all_prices.append(prices_scn)

    if not all_prices:
        return {"precio_optimo": 0.0, "ok": False, "msg": "No scenarios"}

    prices = np.concatenate(all_prices)
    metrics = _tail_metrics(prices)
    precio_bruto = float(np.mean(prices))

    # -------------------------
    # Costes de venta (tx + admin)
    # -------------------------
    sell_tx_cost_bps = float(getattr(REWARD, "sell_tx_cost_bps", 0.0))
    sell_admin_cost = float(getattr(REWARD, "sell_admin_cost_abs", 0.0))

    sell_tx_cost = (sell_tx_cost_bps / 10_000.0) * precio_bruto
    total_cost_tx = float(sell_tx_cost + sell_admin_cost)
    precio_neto = float(max(0.0, precio_bruto - total_cost_tx))

    # -------------------------
    # Anti-arbitraje (cuando book no viene informado)
    # -------------------------
    # Si book_value no es "provided", capamos el upside vs book para evitar PnL positivos "por construcción"
    # (sin distorsionar casos donde el book real sí justifica ganancia).
    if book_source == "proxy_pd_lgd":
        max_pb = float(getattr(CONFIG, "max_price_to_book_if_proxy", 1.05))
        precio_neto = float(min(precio_neto, book_value * max_pb))
    elif book_source == "coverage_rate":
        max_pb = float(getattr(CONFIG, "max_price_to_book_if_derived", 1.10))
        precio_neto = float(min(precio_neto, book_value * max_pb))

    # ✅ PnL contable (bank-ready): vs book_value (ya incluye costes tx)
    pnl_book = float(precio_neto - book_value)
    # métrica de referencia económica (legacy/auditoría)
    pnl_vs_recovery = float(precio_neto - base_recovery)

    # -------------------------
    # Capital release (STD en DEFAULT)
    # -------------------------
    rw_disc = float(_rw_default_discrete(seg_enum, rating_u))
    capital_release = float(ead * rw_disc * _CAP_RATIO)

    # -------------------------
    # Ratios (auditable)
    # -------------------------
    price_ratio_ead = float(precio_neto / ead) if ead > 0 else 0.0
    price_ratio_recovery = float(precio_neto / base_recovery) if base_recovery > 0 else 0.0
    price_ratio_book = float(precio_neto / book_value) if book_value > 0 else 0.0

    # Fire-sale de mercado vs book (umbral configurable; default 0.85)
    fire_sale_threshold_book = float(getattr(CONFIG, "fire_sale_price_ratio_book", 0.85))
    fire_sale = bool(price_ratio_book < fire_sale_threshold_book)

    pd_str = "NA" if pd is None else f"{float(pd):.2f}"

    logger.info(
        "🪙 Simulación precio NPL → "
        f"seg={seg_enum.name} | EAD={ead:,.0f} | LGD={lgd:.2f} | "
        f"PD_prob={pd_str} | RW_disc={rw_disc:.2f} | DPD={dpd:.0f} | "
        f"Book={book_value:,.2f} ({book_source}) | Price_neto={precio_neto:,.2f} | "
        f"Price/EAD={price_ratio_ead:.3f} | Price/Book={price_ratio_book:.3f} | "
        f"fire_sale={fire_sale} (thr={fire_sale_threshold_book:.2f})"
    )

    return {
        # precio neto (post costes) que consume el resto del pipeline
        "precio_optimo": float(precio_neto),
        "precio_bruto": float(precio_bruto),
        "resumen": metrics,

        # ✅ PnL contable (bank-ready)
        "pnl": float(pnl_book),
        "pnl_vs_recovery": float(pnl_vs_recovery),

        # capital release coherente con RWA=EAD*RW_disc y CAP_RATIO
        "capital_liberado": float(capital_release),
        "coste_tx": float(total_cost_tx),

        "price_ratio_ead": float(price_ratio_ead),
        "price_ratio_recovery": float(price_ratio_recovery),
        "price_ratio_book": float(price_ratio_book),

        "fire_sale": bool(fire_sale),
        "fire_sale_threshold_book": float(fire_sale_threshold_book),

        "book_value": float(book_value),
        "book_value_source": str(book_source),
        "base_recovery": float(base_recovery),

        # RW discreto para consistencia micro↔macro
        "rw": float(rw_disc),
        "total_capital_ratio": float(_CAP_RATIO),

        # señales para auditoría/diagnóstico
        "det_mult": float(det_mult),
        "dpd_mult": float(dpd_mult),
        "sec_mult": float(sec_mult),
        "pd_mult": float(pd_mult),
        "rating_mult": float(rat_mult),

        "ok": True,
        "msg": "OK",
    }
