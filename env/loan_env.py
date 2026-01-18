# -*- coding: utf-8 -*-
# ============================================
# env/loan_env.py — LoanEnv v6.4 (AUDIT-READY)
# (Banco L1.5 · STD · EVA-Optimal RL · NPL)
# ============================================
"""
POC — Entorno RL para optimización de carteras en default (Basilea III · Banco L1.5)

Acciones:
    0 = MANTENER
    1 = REESTRUCTURAR
    2 = VENDER

Observación (10 features, estable para PPO y policy_inference):
    [EAD, PD, LGD, RW, EVA, RONA, RORWA, rating_num, segmento_id, DPD/30]

Versión NPL:
    - Pensada para carteras de préstamos en default (DPD ≥ 90, PD alta)
    - RORWA puede ser negativo
    - Venta usa simulate_npl_price (P&L + capital liberado)
    - Umbrales prudenciales vía BankStrategy (esfuerzo_bajo / esfuerzo_alto / dscr_min)

CORRECCIONES CRÍTICAS (consistencia end-to-end):
    - rating_num alineado con policy_inference.py: AAA=9 ... CCC=3
    - segmento_id SIEMPRE numérico (float) y alineado con generate_portfolio.py (CORPORATE=1..OTHER=9)
    - En DEFAULT (STD): RW discreto 1.00 o 1.50 (no valores intermedios)
    - NI/EVA coherentes con generador: PD es “forward a horizonte” y EL se anualiza para NI (EL_annual)
    - Venta: se pasa book_value/coverage_rate al simulador para P&L contable (fire-sale vs book)
    - Reward: risk_proxy usa EL lifetime (no annual)
    - Sin escalado manual de observación por defecto: normalización vía VecNormalize si se usa

PATCHES v6.4 (calidad decisión):
    - ✅ Economía "true" centralizada (NI/RORWA/EVA) y consistente en estado/reward
    - ✅ EVA calculada como NI - hurdle*RWA (estable)
    - ✅ Observación: clip SOLO de RORWA feature (no de la economía)
    - ✅ Reestructura: evalúa RW_post (cure-aware) dentro del grid
    - ✅ Gate duro DSCR (viabilidad) además de PTI
    - ✅ Capital release también en reestructura (si baja RWA)
    - ✅ Fire-sale explícito por price/book y P&L contable (trazable en info)
    - ✅ Evita doble conteo de costes one-off en reestructura
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config as cfg
from optimizer.price_simulator import simulate_npl_price


# ---------------------------------------------------------------
# 🔧 Configuración
# ---------------------------------------------------------------
RESTRUCT_CFG = cfg.CONFIG.reestructura
SENS_CFG = cfg.CONFIG.sensibilidad_reestructura
ENV_CFG = cfg.CONFIG.env
REG_CFG = cfg.CONFIG.regulacion
HURDLE = float(REG_CFG.hurdle_rate)
BANK_STRAT = cfg.CONFIG.bank_strategy
REWARD_CFG = cfg.CONFIG.reward

# Funding cost consistente con generate_portfolio.py y resto del pipeline
COST_FUND = 0.006

# CAP_RATIO robusto (igual criterio que en otros módulos)
if hasattr(REG_CFG, "required_total_capital_ratio") and callable(REG_CFG.required_total_capital_ratio):
    CAP_RATIO = float(REG_CFG.required_total_capital_ratio())
else:
    base = float(getattr(REG_CFG, "total_capital_min", 0.08))
    buf = getattr(getattr(REG_CFG, "buffers", None), "total_buffer", lambda: 0.0)()
    CAP_RATIO = base + buf

cfg.set_all_seeds(cfg.GLOBAL_SEED)


class LoanEnv(gym.Env):
    """Entorno EVA-Optimal (STD · Basilea III · Banco L1.5) centrado en NPL."""

    metadata = {"render_modes": ["human"]}

    # -----------------------------------------------------------
    # INIT
    # -----------------------------------------------------------
    def __init__(self, loan_pool: Optional[List[Dict[str, Any]]] = None, seed: Optional[int] = None):
        super().__init__()
        self.rng = np.random.default_rng(seed or cfg.GLOBAL_SEED)
        self.loan_pool = loan_pool or []
        self.pool_index = 0
        self.steps = 0
        self.state: Dict[str, Any] = {}

        self.hurdle = HURDLE
        self.reward_cfg = REWARD_CFG
        self.restruct_cfg = RESTRUCT_CFG
        self.sens_cfg = SENS_CFG
        self.cfg = ENV_CFG
        self.bank_strategy = BANK_STRAT

        # ✅ Consistencia NI/EVA con el generador:
        # PD en NPL = forward a horizonte; EL es lifetime al horizonte y se anualiza para NI.
        self.horizon_years = float(max(1.0, (float(getattr(self.sens_cfg, "horizon_months", 24.0)) / 12.0)))

        # ENV_CFG puede ser dict o dataclass
        if isinstance(self.cfg, dict):
            self.n_actions = int(self.cfg.get("n_actions", 3))
            self.max_steps = int(self.cfg.get("max_steps", 50))
            self.normalize_obs = bool(self.cfg.get("normalize_obs", False))
        else:
            self.n_actions = int(getattr(self.cfg, "n_actions", 3))
            self.max_steps = int(getattr(self.cfg, "max_steps", 50))
            self.normalize_obs = bool(getattr(self.cfg, "normalize_obs", False))

        # Control de diversificación / mode collapse
        self.last_actions: List[int] = []
        self.max_last_actions = 50

        # Escalas P&L (coherente con policy_inference: pnl_scaled = pnl / scale)
        self.pnl_penalty_scale = float(getattr(self.reward_cfg, "pnl_penalty_scale", 1e5))

        # Espacios Gym
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

    # -----------------------------------------------------------
    # ✅ Coerciones robustas (audit-ready)
    # -----------------------------------------------------------
    def _segment_id_map(self) -> Dict[str, int]:
        """
        Mapa estable y alineado con generate_portfolio.py (fuente de verdad para RL):
          CORPORATE=1, SME=2, RETAIL=3, MORTGAGE=4, CONSUMER=5, SOVEREIGN=6, BANK=7, LEASING=8, OTHER=9
        """
        base = {
            "CORPORATE": 1,
            "SME": 2,
            "RETAIL": 3,
            "MORTGAGE": 4,
            "CONSUMER": 5,
            "SOVEREIGN": 6,
            "BANK": 7,
            "LEASING": 8,
            "OTHER": 9,
        }

        m: Dict[str, int] = dict(base)

        # Si el Enum tiene values (p.ej. "Corporate"), mapea también
        try:
            for seg in cfg.Segmento:
                key_name = str(getattr(seg, "name", "")).strip().upper()
                key_val = str(getattr(seg, "value", "")).strip().upper()
                if key_name in base:
                    m[key_name] = base[key_name]
                    if key_val:
                        m[key_val] = base[key_name]
        except Exception:
            pass

        # aliases típicos (legacy)
        m.update({
            "LARGE CORPORATE": m.get("CORPORATE", 1),
            "CORP": m.get("CORPORATE", 1),
            "MIDCAP": m.get("SME", 2),
            "MID CAP": m.get("SME", 2),
            "MID-CAP": m.get("SME", 2),
            "PYME": m.get("SME", 2),
            "PYMES": m.get("SME", 2),
            "MINORISTA": m.get("RETAIL", 3),
            "HIPOTECARIO": m.get("MORTGAGE", 4),
            "HIPOTECA": m.get("MORTGAGE", 4),
            "CONSUMO": m.get("CONSUMER", 5),
            "PROJECT FINANCE": m.get("OTHER", 9),
        })
        return m

    def _coerce_segmento_id(self, st: Dict[str, Any]) -> float:
        seg_map = self._segment_id_map()

        val = st.get("segmento_id", None)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        if isinstance(val, str) and val.strip():
            key = val.strip().upper()
            return float(seg_map.get(key, seg_map.get("CORPORATE", 1)))

        seg = st.get("segment", st.get("segmento", st.get("segmento_banco", None)))
        if isinstance(seg, str) and seg.strip():
            key = seg.strip().upper()
            return float(seg_map.get(key, seg_map.get("CORPORATE", 1)))

        return float(seg_map.get("CORPORATE", 1))

    def _coerce_rating_num(self, st: Dict[str, Any]) -> float:
        """
        Alineado con policy_inference.py:
          AAA=9, AA=8, A=7, BBB=6, BB=5, B=4, CCC=3
        """
        val = st.get("rating_num", None)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        rating = st.get("rating", st.get("rating_str", "BBB"))
        r = str(rating).strip().upper() if rating is not None else "BBB"
        rating_map = {"AAA": 9, "AA": 8, "A": 7, "BBB": 6, "BB": 5, "B": 4, "CCC": 3}
        return float(rating_map.get(r, 6))

    def _coerce_rw(self, rw: Any) -> float:
        """
        RW puede venir como:
          - 1.0 / 1.5 (multiplicador)
          - 100 / 150 (porcentaje)
        Normalizamos a multiplicador.
        """
        try:
            x = float(rw)
        except Exception:
            return float("nan")
        if x > 10.0:
            x = x / 100.0
        return float(x)

    def _resolve_rw_default_discrete(self, st: Dict[str, Any]) -> float:
        """
        Regla discreta para DEFAULT (STD simplificado):
          - Mortgage => 1.00
          - Resto    => 1.50 salvo que exista specific_adjustments_pct >= 20% => 1.00
        """
        seg_raw = str(st.get("segment", st.get("segmento_banco", "CORPORATE"))).strip().upper()
        rating_raw = str(st.get("rating", "BBB")).strip().upper()

        if "MORTGAGE" in seg_raw or seg_raw in {"HIPOTECARIO", "HIPOTECA"}:
            return 1.00

        adj_keys = [
            "specific_adjustments_pct",
            "specific_credit_risk_adjustments_pct",
            "provisions_pct",
            "provision_pct",
            "scr_adjustments_pct",
        ]
        adj = None
        for k in adj_keys:
            if k in st and st.get(k) is not None:
                adj = st.get(k)
                break

        adj_pct = 0.0
        try:
            if adj is not None:
                adj_pct = float(adj)
                if adj_pct > 1.0:
                    adj_pct = adj_pct / 100.0
        except Exception:
            adj_pct = 0.0

        rw = 1.00 if adj_pct >= 0.20 else 1.50

        # Si basel_map existe, se consulta pero se discretiza al final
        try:
            seg_enum = cfg.Segmento[seg_raw] if seg_raw in getattr(cfg.Segmento, "__members__", {}) else cfg.Segmento.CORPORATE
            rw_engine = cfg.CONFIG.basel_map.resolve_rw(
                segmento=seg_enum,
                rating=rating_raw,
                estado=cfg.EstadoCredito.DEFAULT,
                secured_by_mortgage=(seg_enum == cfg.Segmento.MORTGAGE),
            )
            rw_engine = self._coerce_rw(rw_engine)
            if np.isfinite(rw_engine):
                rw = 1.00 if rw_engine < 1.25 else 1.50
        except Exception:
            pass

        return float(rw)

    # -----------------------------------------------------------
    # Economía "true" (consistente y auditable)
    # -----------------------------------------------------------
    def _econ_true(self, ead: float, pd: float, lgd: float, rw: float, rate: float, one_off: float = 0.0) -> Dict[str, float]:
        """
        Economía consistente:
          - EL_lifetime = PD*LGD*EAD (PD forward a horizonte)
          - EL_annual   = EL_lifetime / horizon_years
          - NI          = EAD*rate - EL_annual - EAD*COST_FUND - one_off
          - EVA_true    = NI - hurdle*RWA
          - RORWA_true  = NI/RWA
        """
        rwa = float(ead * rw)
        el_life = float(pd * lgd * ead)
        el_ann = float(el_life / max(self.horizon_years, 1e-9))
        ni = float(ead * rate - el_ann - ead * COST_FUND - one_off)

        rorwa_true = 0.0 if rwa <= 1e-12 else float(ni / rwa)
        rona = 0.0 if ead <= 1e-12 else float(ni / ead)
        eva_true = float(ni - self.hurdle * rwa)

        return {
            "RWA": rwa,
            "EL": el_life,
            "EL_annual": el_ann,
            "NI": ni,
            "RORWA_true": rorwa_true,
            "RONA": rona,
            "EVA_true": eva_true,
        }

    # -----------------------------------------------------------
    # Selección de préstamo
    # -----------------------------------------------------------
    def _next_loan_from_pool(self) -> Dict[str, Any]:
        if not self.loan_pool:
            base = self._sample_random_state()
        else:
            loan = self.loan_pool[self.pool_index % len(self.loan_pool)]
            self.pool_index += 1
            base = dict(loan)
        return self._ensure_rich_state(base)

    # -----------------------------------------------------------
    # Enriquecimiento de estado
    # -----------------------------------------------------------
    def _ensure_rich_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        st = dict(st)

        rating_str = str(st.get("rating", "BBB")).strip().upper()
        seg_str = str(st.get("segment", st.get("segmento_banco", "CORPORATE"))).strip().upper()

        st["rating"] = rating_str
        st["segment"] = seg_str

        st["rating_num"] = self._coerce_rating_num(st)
        st["segmento_id"] = self._coerce_segmento_id(st)

        # Core
        ead = float(st.get("EAD", 0.0) or 0.0)
        pd = float(st.get("PD", 0.50) or 0.50)
        lgd = float(st.get("LGD", 0.60) or 0.60)
        rate = float(st.get("rate", 0.06) or 0.06)

        # RW: si viene informado lo normalizamos; si no, resolvemos discreto default
        rw_in = st.get("RW", None)
        rw = self._coerce_rw(rw_in) if rw_in is not None else float("nan")
        if not np.isfinite(rw):
            rw = self._resolve_rw_default_discrete(st)
        else:
            rw = 1.00 if rw < 1.25 else 1.50

        # DPD / meses_en_default
        st.setdefault("DPD", 180.0)
        dpd = float(st.get("DPD", 180.0) or 180.0)
        st["DPD"] = float(dpd)
        st["meses_en_default"] = max(int(dpd // 30), 0)

        # Secured: no inventamos; si falta, Mortgage => 1, resto => 0 (conservador)
        if "secured" not in st or st.get("secured") is None:
            st["secured"] = int("MORTGAGE" in seg_str or seg_str in {"HIPOTECARIO", "HIPOTECA"})
        else:
            st["secured"] = int(bool(st.get("secured")))

        # Cobertura / book (si viene, lo preservamos; si no, se computa en venta)
        if "coverage_rate" in st and st["coverage_rate"] is not None:
            try:
                cov = float(st["coverage_rate"])
                if cov > 1.0:
                    cov = cov / 100.0
                st["coverage_rate"] = float(np.clip(cov, 0.0, 1.0))
            except Exception:
                pass

        # Economía "true"
        econ = self._econ_true(
            ead=ead,
            pd=float(np.clip(pd, 0.01, 0.999)),
            lgd=float(np.clip(lgd, 0.0, 1.0)),
            rw=float(rw),
            rate=float(rate),
            one_off=0.0,
        )

        st["EAD"] = float(ead)
        st["PD"] = float(np.clip(pd, 0.01, 0.999))
        st["LGD"] = float(np.clip(lgd, 0.0, 1.0))
        st["RW"] = float(rw)
        st["RWA"] = float(econ["RWA"])
        st["rate"] = float(rate)

        st["EL"] = float(econ["EL"])
        st["EL_annual"] = float(econ["EL_annual"])
        st["NI"] = float(econ["NI"])

        st["RORWA"] = float(econ["RORWA_true"])  # true, sin clipping
        st["EVA"] = float(econ["EVA_true"])      # true, consistente
        st["RONA"] = float(econ["RONA"])

        # Ingreso y cuota (proxy)
        ingreso_mensual = float(st.get("ingreso_mensual", 0.0) or 0.0)
        if ingreso_mensual <= 0:
            ingreso_mensual = max(1.0, ead / 24_000.0)

        cuota = float(st.get("cuota_mensual", 0.0) or 0.0)
        if cuota <= 0 and ead > 0 and rate > 0:
            cuota = ead * rate / 12.0  # interés-only simplificado

        st["ingreso_mensual"] = float(ingreso_mensual)
        st["cuota_mensual"] = float(cuota)

        pti = float(cuota / max(1.0, ingreso_mensual))
        st["PTI"] = float(pti)
        st["DSCR"] = float(0.0 if pti <= 0 else 1.0 / pti)

        st.setdefault("cured", False)
        return st

    # -----------------------------------------------------------
    # Observación estable para PPO
    # -----------------------------------------------------------
    def _obs_from_state(self, st: Dict[str, Any]) -> np.ndarray:
        """
        Observación EXACTA: [EAD, PD, LGD, RW, EVA, RONA, RORWA, rating_num, segmento_id, DPD/30]
        Nota: RORWA se clippea SOLO como feature (estabilidad), no altera economía.
        """
        EAD = float(st.get("EAD", 0.0))
        PD = float(st.get("PD", 0.0))
        LGD = float(st.get("LGD", 0.0))
        RW = float(st.get("RW", 0.0))
        EVA = float(st.get("EVA", 0.0))
        RONA = float(st.get("RONA", 0.0))

        # Feature clip (no economía)
        RORWA_true = float(st.get("RORWA", 0.0))
        RORWA_obs = float(np.clip(RORWA_true, -0.30, 0.30))

        rating_num = self._coerce_rating_num(st)
        segmento_id = self._coerce_segmento_id(st)
        DPD_30 = float(st.get("DPD", 0.0)) / 30.0

        obs = np.array([EAD, PD, LGD, RW, EVA, RONA, RORWA_obs, rating_num, segmento_id, DPD_30], dtype=np.float32)

        # Si alguien fuerza normalize_obs=True en config, lo permitimos, pero NO recomendado con VecNormalize.
        if self.normalize_obs:
            scale = np.array([1e6, 1.0, 1.0, 1.5, 1e6, 1.0, 0.30, 9.0, 9.0, 24.0], dtype=np.float32)
            obs = obs / scale

        return obs

    # -----------------------------------------------------------
    # Generador aleatorio NPL
    # -----------------------------------------------------------
    def _sample_random_state(self) -> Dict[str, Any]:
        seg = self.rng.choice(["CORPORATE", "SME", "MORTGAGE", "CONSUMER"])
        rat = self.rng.choice(["BBB", "BB", "B", "CCC"], p=[0.25, 0.30, 0.30, 0.15])

        ead = float(np.exp(self.rng.normal(13.0, 0.7)))
        pd = float(np.clip(np.exp(self.rng.normal(-0.3, 0.6)), 0.30, 0.999))
        lgd = float(self.rng.uniform(0.4, 0.8))
        dpd = float(self.rng.integers(90, 720))

        return {"segment": seg, "rating": rat, "EAD": ead, "PD": pd, "LGD": lgd, "DPD": dpd}

    # -----------------------------------------------------------
    # Gym API — reset
    # -----------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.last_actions.clear()

        st = self._next_loan_from_pool()
        self.state = st
        obs = self._obs_from_state(st)

        info = {
            "metrics": {
                "EVA": float(st.get("EVA", 0.0)),
                "RORWA": float(st.get("RORWA", 0.0)),
                "capital_liberado": 0.0,
                "PTI": float(st.get("PTI", 0.0)),
                "DSCR": float(st.get("DSCR", 0.0)),
                "cured": bool(st.get("cured", False)),
                "EL": float(st.get("EL", 0.0)),
                "EL_annual": float(st.get("EL_annual", 0.0)),
                "NI": float(st.get("NI", 0.0)),
            }
        }
        return obs, info

    # -----------------------------------------------------------
    # Gym API — step
    # -----------------------------------------------------------
    def step(self, action: int):
        self.steps += 1
        st0 = self.state.copy()

        eva0 = float(st0.get("EVA", 0.0))
        rwa0 = float(st0.get("RWA", st0.get("EAD", 0.0) * st0.get("RW", 0.0)))
        pd0 = float(st0.get("PD", 0.0))
        lgd0 = float(st0.get("LGD", 0.0))
        ead0 = float(st0.get("EAD", 0.0))
        cured0 = bool(st0.get("cured", False))
        dpd0 = float(st0.get("DPD", 0.0))
        ingreso_mensual = float(st0.get("ingreso_mensual", max(1.0, ead0 / 24_000.0)))

        st1 = dict(st0)

        action = int(action)

        capital_release = 0.0
        restruct_cost = 0.0
        sell_tx_cost = 0.0

        best_restruct: Optional[Dict[str, Any]] = None

        pnl_sale = 0.0
        precio_venta = 0.0
        book_value = None
        cov = None
        price_to_book = 0.0
        fire_sale_flag = False

        # -----------------------
        # 0) MANTENER
        # -----------------------
        if action == 0:
            if cured0:
                # cured: no re-delinquir por tick; mantener DPD en 0 si ya lo está
                pd1 = float(np.clip(pd0 * 0.95, 0.01, 0.50))
                dpd1 = float(max(0.0, dpd0 - 30.0))
            else:
                drift = 1.0 + 0.02 * np.tanh(dpd0 / 360.0)
                pd1 = float(np.clip(pd0 * drift, 0.20, 0.999))
                dpd1 = float(dpd0 + 30.0)

            lgd1 = float(np.clip(lgd0 * (1 + 0.02 * self.rng.normal()), 0.0, 1.0))

            st1["PD"] = pd1
            st1["LGD"] = lgd1
            st1["DPD"] = dpd1
            st1["meses_en_default"] = max(0, int(dpd1 // 30))

        # -----------------------
        # 1) REESTRUCTURAR
        # -----------------------
        elif action == 1:
            best_gain = -1e18
            best_updates: Dict[str, Any] = {}

            esfuerzo_max = float(getattr(self.bank_strategy, "esfuerzo_alto", self.restruct_cfg.esfuerzo_umbral_alto))
            pd_min = float(self.restruct_cfg.pd_min)
            pd_max = float(self.restruct_cfg.pd_max)

            dscr_min = float(getattr(self.bank_strategy, "dscr_min", 1.0))
            pd_cure_th = float(getattr(self.sens_cfg, "pd_cure_threshold", 0.20))
            rw_perf_guess = float(getattr(self.sens_cfg, "rw_perf_guess", st0.get("RW", 1.5)))

            for plazo in self.restruct_cfg.plazo_anios_grid:
                for tasa in self.restruct_cfg.tasa_anual_grid:
                    for quita in self.restruct_cfg.quita_grid:
                        ead_c = ead0 * (1 - quita)

                        pd_c = pd0 * (1 - self.sens_cfg.pd_reduction_per_year * plazo)
                        pd_c *= (1 - self.sens_cfg.pd_reduction_per_quita * quita)
                        pd_c = float(np.clip(pd_c, pd_min, pd_max))

                        lgd_c = float(np.clip(
                            lgd0 * (1 - self.sens_cfg.lgd_reduction_per_quita * quita), 0.0, 1.0
                        ))

                        cuota_c = ead_c * tasa / 12.0 if ead_c > 0 else 0.0
                        pti_c = cuota_c / max(1.0, ingreso_mensual)
                        if pti_c > esfuerzo_max:
                            continue

                        dscr_c = 0.0 if pti_c <= 0 else 1.0 / pti_c
                        if dscr_c < dscr_min:
                            continue

                        # Cure-aware RW (para evaluar correctamente)
                        cured_candidate = bool(pd_c < pd_cure_th)
                        rw_post = float(rw_perf_guess) if cured_candidate else float(st0["RW"])
                        dpd_post = 0.0 if cured_candidate else 60.0
                        meses_post = 0 if cured_candidate else 2

                        admin = float(self.reward_cfg.restructure_admin_cost_abs)
                        quita_cost = (float(self.reward_cfg.restructure_cost_quita_bps) / 10_000.0) * (quita * ead0)
                        one_off = float(admin + quita_cost)

                        econ_c = self._econ_true(
                            ead=float(ead_c),
                            pd=float(pd_c),
                            lgd=float(lgd_c),
                            rw=float(rw_post),
                            rate=float(tasa),
                            one_off=float(one_off),
                        )

                        eva_c = float(econ_c["EVA_true"])
                        gain = eva_c - eva0

                        if gain > best_gain:
                            best_gain = gain
                            best_restruct = {
                                "plazo_anios": float(plazo),
                                "tasa": float(tasa),
                                "quita": float(quita),
                                "admin_cost": float(admin),
                                "quita_cost": float(quita_cost),
                                "one_off_total": float(one_off),
                                "PD_post": float(pd_c),
                                "LGD_post": float(lgd_c),
                                "EAD_post": float(ead_c),
                                "RW_post": float(rw_post),
                                "cured": bool(cured_candidate),
                                "DPD_post": float(dpd_post),
                                "meses_en_default_post": int(meses_post),
                                "cuota_post": float(cuota_c),
                                "PTI_post": float(pti_c),
                                "DSCR_post": float(dscr_c),
                                "EL_life_post": float(econ_c["EL"]),
                                "EL_annual_post": float(econ_c["EL_annual"]),
                                "NI_post": float(econ_c["NI"]),
                                "RORWA_post": float(econ_c["RORWA_true"]),
                                "EVA_post": float(eva_c),
                            }
                            best_updates = {
                                "EAD": float(ead_c),
                                "PD": float(pd_c),
                                "LGD": float(lgd_c),
                                "rate": float(tasa),
                                "RW": float(rw_post),
                                "DPD": float(dpd_post),
                                "meses_en_default": int(meses_post),
                                "cured": bool(cured_candidate),
                                "cuota_mensual": float(cuota_c),
                                "ingreso_mensual": float(ingreso_mensual),
                                "PTI": float(pti_c),
                                "DSCR": float(dscr_c),
                            }

            if best_restruct is not None:
                st1.update(best_updates)
                restruct_cost = float(best_restruct.get("one_off_total", 0.0))
            else:
                # si no hay solución viable, coste parcial (fricción) para desincentivar abuso
                restruct_cost = float(self.reward_cfg.restructure_admin_cost_abs) * 0.25

        # -----------------------
        # 2) VENDER
        # -----------------------
        elif action == 2:
            cov = float(st0.get("coverage_rate", 0.0) or 0.0)
            if cov > 1.0:
                cov = cov / 100.0
            cov = float(np.clip(cov, 0.0, 1.0))

            book_value = st0.get("book_value", None)
            if book_value is None:
                book_value = float(ead0 * (1.0 - cov))
            book_value = float(book_value)

            out_price = simulate_npl_price(
                ead=ead0,
                lgd=lgd0,
                pd=pd0,
                dpd=dpd0,
                segment=str(st0.get("segment", "CORPORATE")).upper(),
                secured=bool(st0.get("secured", False)),
                rating=str(st0.get("rating", "BBB")).upper(),
                book_value=book_value,
                coverage_rate=cov,
            )

            precio_venta = float(out_price.get("precio_optimo", 0.0))
            pnl_sale = float(out_price.get("pnl", 0.0))  # ya vs book_value
            capital_release = float(out_price.get("capital_liberado", rwa0 * CAP_RATIO))
            sell_tx_cost = float(out_price.get("coste_tx", 0.0))

            price_to_book = 0.0 if book_value <= 1e-12 else float(precio_venta / book_value)
            min_pb = float(getattr(self.reward_cfg, "min_price_to_book", 0.95))
            fire_sale_flag = bool(price_to_book < min_pb or pnl_sale < 0.0)

            # sale de balance (estado terminal) — preservamos segment/rating para obs estable
            st1.update(
                {
                    "EAD": 0.0,
                    "PD": 0.0,
                    "LGD": 0.0,
                    "RW": 0.0,
                    "rate": 0.0,
                    "RWA": 0.0,
                    "EL": 0.0,
                    "EL_annual": 0.0,
                    "NI": 0.0,
                    "EVA": 0.0,
                    "RORWA": 0.0,
                    "RONA": 0.0,
                    "DPD": 0.0,
                    "meses_en_default": 0,
                    "cured": False,
                    "PTI": 0.0,
                    "DSCR": 0.0,
                    "cuota_mensual": 0.0,
                    "ingreso_mensual": float(ingreso_mensual),
                }
            )

        # -------------------------------------------------------
        # Recalcular métricas solo si NO es venta
        # -------------------------------------------------------
        if action != 2:
            st1["RWA"] = float(st1["EAD"] * st1["RW"])

            econ1 = self._econ_true(
                ead=float(st1["EAD"]),
                pd=float(st1["PD"]),
                lgd=float(st1["LGD"]),
                rw=float(st1["RW"]),
                rate=float(st1["rate"]),
                one_off=float(restruct_cost),
            )

            st1["EL"] = float(econ1["EL"])
            st1["EL_annual"] = float(econ1["EL_annual"])
            st1["NI"] = float(econ1["NI"])
            st1["RORWA"] = float(econ1["RORWA_true"])
            st1["EVA"] = float(econ1["EVA_true"])
            st1["RONA"] = float(econ1["RONA"])

            # cuota/PTI/DSCR (si no venía ya)
            if "cuota_mensual" not in st1:
                st1["cuota_mensual"] = float(st1["EAD"] * st1["rate"] / 12.0 if st1["EAD"] > 0 else 0.0)

            st1["ingreso_mensual"] = float(ingreso_mensual)
            st1["PTI"] = float(st1["cuota_mensual"] / max(1.0, ingreso_mensual))
            st1["DSCR"] = float(0.0 if st1["PTI"] <= 0 else 1.0 / st1["PTI"])

            # Capital release también en reestructura si reduce RWA
            if action == 1 and rwa0 > 0.0:
                capital_release = max(0.0, (float(rwa0) - float(st1["RWA"])) * CAP_RATIO)

        # -------------------------------------------------------
        # Reward (coherente con policy_inference)
        # -------------------------------------------------------
        # Para venta, EVA post no tiene sentido económico; se premia por pnl + capital_release
        if action == 2:
            eva_gain = 0.0
        else:
            eva_gain = float(st1.get("EVA", 0.0) - eva0)

        rel_cap = float(capital_release / max(rwa0 * CAP_RATIO, 1e-9) if rwa0 > 0 else 0.0)

        esfuerzo_bajo = float(getattr(self.bank_strategy, "esfuerzo_bajo", self.restruct_cfg.esfuerzo_umbral_bajo))
        esfuerzo_alto = float(getattr(self.bank_strategy, "esfuerzo_alto", self.restruct_cfg.esfuerzo_umbral_alto))

        pti1 = float(st1.get("PTI", 0.0))
        excess_pti = max(0.0, pti1 - esfuerzo_bajo) / max(1e-6, esfuerzo_alto)
        pti_penalty = float(excess_pti * abs(eva0))

        # ✅ risk_proxy lifetime (no annual)
        risk_proxy = float(st1.get("EL", st1.get("PD", 0.0) * st1.get("LGD", 0.0) * st1.get("EAD", 0.0)))

        cure_bonus = 0.0
        if (not cured0) and bool(st1.get("cured", False)):
            cure_bonus = float(0.2 * abs(eva0 if eva0 != 0 else float(st1.get("EVA", 0.0))))

        # P&L escalado (venta)
        w_pnl = float(getattr(self.reward_cfg, "w_pnl", 0.0))
        pnl_scaled = float(pnl_sale / max(self.pnl_penalty_scale, 1.0))

        dscr_min = float(getattr(self.bank_strategy, "dscr_min", 1.0))
        dscr1 = float(st1.get("DSCR", 0.0))
        dscr_excess = max(0.0, dscr1 - dscr_min)
        w_dscr = float(getattr(self.reward_cfg, "w_dscr_bonus", 0.0))
        dscr_bonus = float(w_dscr * dscr_excess)

        r = (
            float(self.reward_cfg.w_eva) * eva_gain
            + float(self.reward_cfg.w_capital) * rel_cap * 1e4
            - float(self.reward_cfg.w_stab) * (risk_proxy / 1e4)
            - 0.1 * pti_penalty
            + cure_bonus
            + w_pnl * pnl_scaled
            + dscr_bonus
        )

        # Bonus por mantener EVA positiva (no tocar lo que va bien)
        if eva0 > 0 and action == 0:
            r += float(self.reward_cfg.maintain_bonus_eva_pos) * abs(eva0)

        # Venta cuando EVA era negativa (limpia value-destroyers)
        if eva0 < 0 and action == 2:
            scale = float(np.tanh(abs(eva0) / 1e5))
            r += float(self.reward_cfg.sell_bonus_eva_neg) * scale

        # Penalización explícita fire-sale (precio/book o pnl contable negativo)
        if action == 2 and fire_sale_flag:
            scale = float(np.tanh(abs(pnl_sale) / 1e5))
            r -= float(getattr(self.reward_cfg, "penalty_fire_sale", 0.0)) * scale

        # mode collapse penalty
        self.last_actions.append(int(action))
        if len(self.last_actions) > self.max_last_actions:
            self.last_actions.pop(0)
        if self.last_actions.count(1) / max(1, len(self.last_actions)) > 0.7:
            r -= float(self.reward_cfg.penalty_mode_collapse) * len(self.last_actions)

        r = float(np.clip(r, *self.reward_cfg.clip))

        # Gymnasium: terminated vs truncated
        terminated = bool(action == 2)
        truncated = bool((not terminated) and (self.steps >= self.max_steps))

        # Asegurar tipos
        st1["rating_num"] = self._coerce_rating_num(st1)
        st1["segmento_id"] = self._coerce_segmento_id(st1)

        self.state = st1
        obs = self._obs_from_state(st1)

        info = {
            "metrics": {
                "EVA": float(st1.get("EVA", 0.0)),
                "RORWA": float(st1.get("RORWA", 0.0)),
                "capital_liberado": float(capital_release),
                "PTI": float(st1.get("PTI", 0.0)),
                "DSCR": float(st1.get("DSCR", 0.0)),
                "cured": bool(st1.get("cured", False)),
                "precio_venta": float(precio_venta),
                "pnl_venta": float(pnl_sale),
                "book_value": float(book_value) if book_value is not None else 0.0,
                "coverage_rate": float(cov) if cov is not None else float(st0.get("coverage_rate", 0.0) or 0.0),
                "price_to_book": float(price_to_book),
                "fire_sale_flag": bool(fire_sale_flag),
                "EL": float(st1.get("EL", 0.0)),
                "EL_annual": float(st1.get("EL_annual", 0.0)),
                "NI": float(st1.get("NI", 0.0)),
            },
            "reward_breakdown": {
                "eva_gain": float(eva_gain),
                "rel_cap": float(rel_cap),
                "pti_penalty": float(pti_penalty),
                "risk_proxy_EL_lifetime": float(risk_proxy),
                "cure_bonus": float(cure_bonus),
                "pnl_scaled": float(pnl_scaled),
                "dscr_bonus": float(dscr_bonus),
                "dscr_excess": float(dscr_excess),
                "dscr_min": float(dscr_min),
                "fire_sale_flag": bool(fire_sale_flag),
                "final": float(r),
            },
            "best_restruct": best_restruct,
            "costs": {
                "restruct_one_off": float(restruct_cost),
                "sell_tx_cost": float(sell_tx_cost),
            },
            "last_action": int(action),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        return obs, float(r), terminated, truncated, info

    # -----------------------------------------------------------
    # Render
    # -----------------------------------------------------------
    def render(self):
        s = self.state
        print(
            f"[LoanEnv] {s.get('segment','?')}/{s.get('rating','?')} "
            f"EAD={s.get('EAD',0):.0f} PD={s.get('PD',0):.3f} LGD={s.get('LGD',0):.2f} "
            f"RW={s.get('RW',0):.2f} RORWA={s.get('RORWA',0):.3f} EVA={s.get('EVA',0):.0f} "
            f"PTI={s.get('PTI',0):.2f} DSCR={s.get('DSCR',0):.2f} "
            f"cured={s.get('cured',False)} DPD={s.get('DPD',0):.0f}"
        )
