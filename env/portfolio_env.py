# -*- coding: utf-8 -*-
# ============================================
# env/portfolio_env.py — PortfolioEnv v3.6 (micro↔macro aware, AUDIT-READY)
# (NPL-Ready · Banco L1.5 · Perfiles de banco · Re-ranking con PPO micro + VN robusto)
# ============================================
"""
Entorno RL MACRO para optimización de CARTERAS NPL (Non-Performing Loans)
Compatible con LoanEnv v6.3, simulate_npl_price y Basilea III STD.

Hardening v3.6 (correcciones relevantes):
  - ✅ Evita “stale aliases” import-time: bank_profile / bank_strategy / reward_cfg se refrescan en reset().
  - ✅ Micro re-ranking compatible con PolicyAdapter (micro ya normaliza):
      * Si ppo_micro expone atributos tipo adapter (model/vecnorm), NO re-normalizamos aquí.
      * Si ppo_micro es PPO “crudo”, soporta VN local (ruta o defaults) + validación shape (10,).
  - ✅ Gymnasium termination semantics:
      * terminated = cartera vacía
      * truncated = time-limit (max_steps) (solo si NOT terminated)
  - ✅ Maintain coherente con cured:
      * si cured=True → DPD se mantiene en 0 (no vuelve a delinquir por “time tick”)
  - ✅ Guardrails fire-sale:
      * Se evalúa contra BOOK VALUE (price/book) + pérdida contable, no solo price/EAD
      * Bloqueo prudente/balanceado salvo desinversión (o allow_fire_sale=True)
      * Umbral de pérdida en % de book (no € fijo): max_loss_eur = -max_loss_pct * book_value
  - ✅ EL/NI coherentes con generador y LoanEnv:
      * PD se interpreta como forward a horizonte
      * Para NI/EVA: EL_annual = (PD*LGD*EAD)/horizon_years
      * risk_proxy (estabilidad) usa EL lifetime (PD*LGD*EAD)
  - ✅ segmento_id alineado con pipeline (CORPORATE=1..OTHER=9), no enum-index 0..N
  - ✅ n_actions mínimo 12 (para el mapa 0..11 utilizado por policy_inference_portfolio)
  - ✅ Pesos/costes por postura: prioriza BankStrategy (fallback a CFG.reward)
  - ✅ Capital carry cost: usa cost_of_capital (no hurdle) y dt por step
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable

import os
import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

import config as cfg
from optimizer.price_simulator import simulate_npl_price

logger = logging.getLogger("portfolio_env")

# ---------------------------------------------------------------
# Configuración / paths
# ---------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Funding cost consistente con el resto del pipeline
COST_FUND = 0.006

cfg.set_all_seeds(cfg.GLOBAL_SEED)


def _cfg_get(obj: Any, key: str, default: Any) -> Any:
    """Acceso robusto a env_cfg dict o dataclass/obj."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _as_str_id(x: Any, fallback: str) -> str:
    if x is None:
        return fallback
    s = str(x).strip()
    return s if s else fallback


def _safe_bool(x: Any, default: bool = False) -> bool:
    try:
        if x is None:
            return default
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return bool(int(x))
        s = str(x).strip().lower()
        if s in ("true", "1", "yes", "y", "si", "sí"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
        return default
    except Exception:
        return default


def _coerce_bank_profile(v: Any):
    """Normaliza bank_profile para que SIEMPRE sea cfg.BankProfile.

    Problema típico: CFG.bank_profile puede venir como string ("prudencial", "balanceado", ...)
    y luego se compara contra cfg.BankProfile.<ENUM>. Si no se normaliza, las comparaciones
    fallan silenciosamente y el macro puede bloquear ventas/reestructuras por postura.
    """

    BP = getattr(cfg, "BankProfile", None)
    if BP is None:
        # fallback ultra-defensivo: devuelve string normalizado
        s = "" if v is None else str(v)
        return s.strip().lower()

    # ya es enum
    try:
        if isinstance(v, BP):
            return v
    except Exception:
        pass

    # normaliza string
    s = "" if v is None else str(v)
    s = s.strip().lower()
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
         .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    )
    s = s.replace("-", "_").replace(" ", "_")

    # mapeo tolerante
    if s in {"prudente", "prudencial", "prudent", "prudential"}:
        return BP.PRUDENTE
    if s in {"desinversion", "desinvertir", "divestment", "desinversionn"}:
        return BP.DESINVERSION
    if s in {"balanceado", "balanced", "neutral"}:
        return BP.BALANCEADO

    # último fallback
    return BP.BALANCEADO


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        loans_df: Optional[pd.DataFrame] = None,
        loan_dicts: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        top_k: int = 5,
        scenario: str = "baseline",
        ppo_micro: Optional[Any] = None,
        micro_deterministic: bool = True,
        micro_vecnormalize_path: Optional[str] = None,  # backward-compatible
        allow_fire_sale: Optional[bool] = None,          # backward-compatible
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed or cfg.GLOBAL_SEED)

        # scenario
        self.scenario = str(scenario or "baseline").lower()

        # PPO micro (puede ser PPO o PolicyAdapter)
        self.ppo_micro = ppo_micro
        self.micro_deterministic = bool(micro_deterministic)

        # micro VN
        self.micro_vec_env = None
        self.micro_vn = None
        self.micro_vn_path_override = micro_vecnormalize_path

        # refresh config aliases (incluye allow_fire_sale default)
        self.allow_fire_sale_override = allow_fire_sale
        self._refresh_cfg_aliases()

        self.top_k_base = int(top_k)

        # espacios
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_actions)

        # estado interno
        self.initial_portfolio: List[Dict[str, Any]] = []
        self.portfolio: List[Dict[str, Any]] = []

        self.steps: int = 0
        self.eva_history: List[float] = []

        # init micro normalizer (solo si hace falta)
        self._init_micro_normalizer()

        # carga cartera
        if loans_df is not None:
            self._load_from_dataframe(loans_df)
        elif loan_dicts is not None:
            self._load_from_dicts(loan_dicts)
        else:
            self._load_from_dicts(self._generate_synthetic_portfolio(n=200))

    # -----------------------------------------------------------------
    # ✅ Postura-aware params: prioriza BankStrategy (fallback reward_cfg)
    # -----------------------------------------------------------------
    def _p(self, key: str, default: float) -> float:
        """
        Devuelve parámetro por postura.
        1) bank_strategy.key si existe
        2) reward_cfg.key si existe
        3) default
        """
        try:
            bs = getattr(self, "bank_strategy", None)
            if bs is not None and hasattr(bs, key):
                v = getattr(bs, key)
                if v is not None:
                    return float(v)
        except Exception:
            pass
        try:
            rc = getattr(self, "reward_cfg", None)
            if rc is not None and hasattr(rc, key):
                v = getattr(rc, key)
                if v is not None:
                    return float(v)
        except Exception:
            pass
        return float(default)

    # -----------------------------------------------------------------
    # Config refresh (evita aliases stale)
    # -----------------------------------------------------------------
    def _refresh_cfg_aliases(self) -> None:
        CFG = cfg.CONFIG
        self.CFG = CFG
        self.reward_cfg = CFG.reward
        self.restruct_cfg = CFG.reestructura
        self.sens_cfg = CFG.sensibilidad_reestructura
        self.env_cfg = CFG.env
        self.reg_cfg = CFG.regulacion

        # ✅ horizonte coherente con generador/LoanEnv
        horizon_months = float(getattr(self.sens_cfg, "horizon_months", 24.0))
        self.horizon_years = float(max(1.0, horizon_months / 12.0))

        # bank posture (dinámico)
        self.bank_profile = _coerce_bank_profile(CFG.bank_profile)
        self.bank_strategy = CFG.bank_strategy

        # hurdle y cap ratio robustos
        self.hurdle = float(getattr(self.reg_cfg, "hurdle_rate", 0.0))

        if hasattr(self.reg_cfg, "required_total_capital_ratio") and callable(self.reg_cfg.required_total_capital_ratio):
            self.cap_ratio = float(self.reg_cfg.required_total_capital_ratio())
        else:
            base = float(getattr(self.reg_cfg, "total_capital_min", 0.08))
            buf_obj = getattr(self.reg_cfg, "buffers", None)
            buf = float(buf_obj.total_buffer()) if buf_obj is not None and hasattr(buf_obj, "total_buffer") else 0.0
            self.cap_ratio = base + buf

        # env dims
        self.max_steps = int(_cfg_get(self.env_cfg, "max_steps_portfolio", 30))
        self.normalize_obs = bool(_cfg_get(self.env_cfg, "normalize_obs_portfolio", True))
        self.state_dim = int(_cfg_get(self.env_cfg, "portfolio_state_dim", 308))
        self.n_actions = int(_cfg_get(self.env_cfg, "portfolio_n_actions", 12))

        # coherencia con el action-map 0..11
        if self.n_actions < 12:
            logger.warning(f"⚠️ portfolio_n_actions={self.n_actions} < 12. Forzando a 12 para compatibilidad.")
            self.n_actions = 12

        self.vol_window = int(_cfg_get(self.env_cfg, "portfolio_eva_vol_window", 8))

        # ✅ pesos por postura (fallback reward_cfg)
        self.w_vol = self._p("w_vol", float(getattr(self.reward_cfg, "w_vol", 0.0)))
        self.w_conc = self._p("w_concentration", float(getattr(self.reward_cfg, "w_concentration", 0.0)))
        self.w_cap_carry = self._p("w_cap_carry", float(getattr(self.reward_cfg, "w_cap_carry", 0.0)))

        # fire-sale behavior
        if self.allow_fire_sale_override is None:
            allow = bool(_cfg_get(self.env_cfg, "allow_fire_sale", False))
        else:
            allow = bool(self.allow_fire_sale_override)
        self.allow_fire_sale = bool(allow)

        # micro VN path (si no override, coge config)
        self.micro_vn_path = (
            self.micro_vn_path_override
            if self.micro_vn_path_override is not None
            else _cfg_get(self.env_cfg, "micro_vecnormalize_path", None)
        )

        # ✅ capital carry config
        self.cost_of_capital = float(getattr(self.reg_cfg, "cost_of_capital", 0.12))
        self.dt_years = float(_cfg_get(self.env_cfg, "dt_years_portfolio", 1.0 / 12.0))

    # -----------------------------------------------------------------
    # Micro: VN loader & validation
    # -----------------------------------------------------------------
    def _pick_default_micro_vn_path(self) -> Optional[str]:
        cands = [
            os.path.join(MODELS_DIR, "vecnormalize_loan.pkl"),
            os.path.join(MODELS_DIR, "vecnormalize_micro.pkl"),
            os.path.join(MODELS_DIR, "vecnormalize_final.pkl"),       # legacy
            os.path.join(MODELS_DIR, "best_model_vecnormalize.pkl"),  # legacy
        ]
        for p in cands:
            if os.path.exists(p):
                return p
        return None

    def _vn_shape_ok(self, vn: Any, expected_dim: int = 10) -> bool:
        try:
            mean = getattr(getattr(vn, "obs_rms", None), "mean", None)
            shp = getattr(mean, "shape", None)
            if shp is None:
                return False
            return tuple(shp) == (expected_dim,)
        except Exception:
            return False

    def _load_micro_vn(self, path: str) -> Optional[Any]:
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from env.loan_env import LoanEnv

            dummy = DummyVecEnv([lambda: LoanEnv()])
            vn = VecNormalize.load(path, dummy)
            vn.training = False
            vn.norm_reward = False

            if not self._vn_shape_ok(vn, expected_dim=10):
                logger.warning(f"⚠️ Micro VN invalidado por shape mismatch: {path}")
                return None

            logger.info(f"🔄 Micro VecNormalize cargado: {path}")
            return vn
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar Micro VecNormalize ({path}): {e}")
            return None

    def _micro_is_adapter(self) -> bool:
        """
        Heurística: si ppo_micro parece un PolicyAdapter (típico: .model y/o .vecnorm),
        asumimos que ya hace normalización y no tocamos VN aquí.
        """
        if self.ppo_micro is None:
            return False
        if hasattr(self.ppo_micro, "model") or hasattr(self.ppo_micro, "vecnorm"):
            return True
        return False

    def _init_micro_normalizer(self) -> None:
        # Si micro es adapter, no hacemos VN aquí (evita double-normalization)
        if self._micro_is_adapter():
            self.micro_vec_env = None
            self.micro_vn = None
            return

        # 1) intenta usar env del PPO (si existe)
        self.micro_vec_env = None
        if self.ppo_micro is not None and hasattr(self.ppo_micro, "get_env"):
            try:
                self.micro_vec_env = self.ppo_micro.get_env()
            except Exception:
                self.micro_vec_env = None

        # Si el env del modelo tiene normalize_obs y respeta shape, lo usamos
        if self.micro_vec_env is not None and hasattr(self.micro_vec_env, "normalize_obs"):
            try:
                dummy = np.zeros((1, 10), dtype=np.float32)
                out = self.micro_vec_env.normalize_obs(dummy)
                if isinstance(out, np.ndarray) and out.shape == (1, 10):
                    self.micro_vn = None
                    return
            except Exception:
                pass

        # 2) carga VN desde ruta/config/default
        vn_path = self.micro_vn_path or self._pick_default_micro_vn_path()
        if vn_path and os.path.exists(vn_path):
            self.micro_vn = self._load_micro_vn(vn_path)
        else:
            self.micro_vn = None

    # -----------------------------------------------------------------
    # Utilidad: top_k efectivo según perfil de banco
    # -----------------------------------------------------------------
    def _effective_top_k(self) -> int:
        k = self.top_k_base
        if self.bank_profile == cfg.BankProfile.DESINVERSION:
            return max(1, int(round(2.0 * k)))
        if self.bank_profile == cfg.BankProfile.PRUDENTE:
            return max(1, int(round(0.5 * k)))
        return max(1, k)

    def _eva_volatility(self) -> float:
        n = len(self.eva_history)
        if n < 2:
            return 0.0
        window = min(n, self.vol_window)
        recent = self.eva_history[-window:]
        return float(np.std(recent))

    # =====================================================================
    #                         CARGA DE CARTERA
    # =====================================================================
    def _load_from_dataframe(self, df: pd.DataFrame) -> None:
        self._load_from_dicts(df.to_dict(orient="records"))

    def _resolve_segment_rating(self, st: Dict[str, Any]) -> Tuple[str, str]:
        seg = None
        for k in ("segmento_banco", "segment_raw", "segment", "Segment", "SEGMENT"):
            if k in st and st.get(k) is not None and str(st.get(k)).strip():
                seg = str(st.get(k)).strip()
                break

        rat = None
        for k in ("rating", "Rating", "RATING", "rating_bucket"):
            if k in st and st.get(k) is not None and str(st.get(k)).strip():
                rat = str(st.get(k)).strip()
                break

        seg = (seg or "CORPORATE").strip().upper()
        rat = (rat or "BBB").strip().upper()
        return seg, rat

    def _load_from_dicts(self, loans: List[Dict[str, Any]]) -> None:
        enriched: List[Dict[str, Any]] = []
        for i, raw in enumerate(loans):
            st = dict(raw)

            base_id = st.get("loan_id", st.get("id", f"loan_{i}"))
            st["loan_id"] = _as_str_id(base_id, f"loan_{i}")

            seg, rat = self._resolve_segment_rating(st)
            st["segment"] = seg
            st["rating"] = rat

            dpd_in = _safe_float(st.get("DPD", raw.get("DPD", 180)), 180.0)
            st["DPD"] = max(dpd_in, 120.0)

            # mínimo 4 meses en default (consistente con NPL)
            st["meses_en_default"] = max(int(float(st["DPD"]) // 30), 4)

            st["estado"] = "DEFAULT"
            st["closed"] = False

            st = self._ensure_rich_state(st)

            # re-aplicar mínimo contractual de default
            if float(st.get("DPD", 0.0)) >= 120.0:
                st["meses_en_default"] = max(int(float(st["DPD"]) // 30), 4)

            enriched.append(st)

        self.initial_portfolio = enriched
        self._reset_portfolio_copy()

    def _reset_portfolio_copy(self) -> None:
        self.portfolio = [dict(l) for l in self.initial_portfolio]

    # =====================================================================
    #       PORTFOLIO SINTÉTICO — SOLO DEBUG
    # =====================================================================
    def _generate_synthetic_portfolio(self, n: int = 200) -> List[Dict[str, Any]]:
        segs = ["CORPORATE", "SME", "RETAIL", "MORTGAGE", "CONSUMER"]
        rats = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

        portfolio: List[Dict[str, Any]] = []
        for i in range(n):
            seg = str(self.rng.choice(segs))
            rat = str(self.rng.choice(rats, p=[0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05]))
            ead = float(np.exp(self.rng.normal(11.0, 0.8)))

            pdv = float(self.rng.uniform(0.30, 0.85))
            lgd = float(self.rng.uniform(0.40, 0.85))

            st = {
                "loan_id": f"synt_{i}",
                "segment": seg,
                "rating": rat,
                "EAD": ead,
                "PD": pdv,
                "LGD": lgd,
                "DPD": 180.0,
                "estado": "DEFAULT",
                "closed": False,
            }
            portfolio.append(st)
        return portfolio

    # =====================================================================
    #                 UTILIDADES FINANCIERAS / RICH STATE
    # =====================================================================
    def _eva(self, rorwa: float, rwa: float) -> float:
        return float(rwa * (rorwa - self.hurdle))

    def _rorwa(self, net_income: float, rwa: float) -> float:
        if rwa <= 1e-12:
            return 0.0
        return float(net_income / rwa)

    # -----------------------------
    # Mapeos consistentes (alineados con LoanEnv / pipeline)
    # -----------------------------
    def _segment_id_map(self) -> Dict[str, int]:
        """
        Mapa estable alineado con pipeline:
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

        # Mapear también seg.value si existe (robustez)
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

        # aliases típicos
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
            "OTRO": m.get("OTHER", 9),
            "PROJECT FINANCE": m.get("OTHER", 9),
        })
        return m

    def _rating_num_map(self) -> Dict[str, int]:
        return {"AAA": 9, "AA": 8, "A": 7, "BBB": 6, "BB": 5, "B": 4, "CCC": 3}

    def _coerce_rw(self, rw: Any) -> float:
        try:
            x = float(rw)
        except Exception:
            return float("nan")
        if np.isnan(x) or np.isinf(x):
            return float("nan")
        if x > 10.0:
            x = x / 100.0
        return float(x)

    def _resolve_rw_default_discrete(self, st: Dict[str, Any]) -> float:
        seg = str(st.get("segment", "CORPORATE")).strip().upper()
        rating = str(st.get("rating", "BBB")).strip().upper()

        if "MORTGAGE" in seg or seg == "HIPOTECARIO":
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

        try:
            seg_enum = cfg.Segmento[seg] if seg in cfg.Segmento.__members__ else cfg.Segmento.CORPORATE
            rw_engine = self.CFG.basel_map.resolve_rw(
                segmento=seg_enum,
                rating=rating,
                estado=cfg.EstadoCredito.DEFAULT,
                secured_by_mortgage=(seg_enum == cfg.Segmento.MORTGAGE),
            )
            rw_engine = self._coerce_rw(rw_engine)
            if np.isfinite(rw_engine):
                rw = 1.00 if rw_engine < 1.25 else 1.50
        except Exception:
            pass

        return float(rw)

    def _ensure_rich_state(self, st: Dict[str, Any]) -> Dict[str, Any]:
        st = dict(st)

        rating = str(st.get("rating", "BBB")).strip().upper()
        segment = str(st.get("segment", "CORPORATE")).strip().upper()
        st["segment"] = segment
        st["rating"] = rating

        seg_map = self._segment_id_map()
        rat_map = self._rating_num_map()

        st["rating_num"] = float(rat_map.get(rating, 6))
        st["segmento_id"] = float(seg_map.get(segment, seg_map.get("CORPORATE", 1)))

        ead = _safe_float(st.get("EAD", 0.0), 0.0)
        pdv = _safe_float(st.get("PD", 0.50), 0.50)
        lgd = _safe_float(st.get("LGD", 0.60), 0.60)
        rate = _safe_float(st.get("rate", 0.06), 0.06)

        # RW: normalizar + discretizar (DEFAULT)
        rw_in = st.get("RW", None)
        rw = self._coerce_rw(rw_in) if rw_in is not None else float("nan")
        if not np.isfinite(rw):
            rw = self._resolve_rw_default_discrete(st)
        else:
            rw = 1.00 if rw < 1.25 else 1.50

        ead = float(max(ead, 0.0))
        pdv = float(np.clip(pdv, 0.01, 0.999))
        lgd = float(np.clip(lgd, 0.0, 1.0))
        rw = float(rw)

        rwa = float(ead * rw)

        # ✅ EL lifetime + annualization para NI (consistente con LoanEnv/generador)
        el_life = float(pdv * lgd * ead)
        el_ann = float(el_life / self.horizon_years)
        ni = float(ead * rate - el_ann - ead * COST_FUND)

        rorwa = self._rorwa(ni, rwa)
        eva = self._eva(rorwa, rwa)
        rona = 0.0 if ead <= 1e-12 else float(ni / ead)

        st["EAD"] = ead
        st["PD"] = pdv
        st["LGD"] = lgd
        st["RW"] = rw
        st["RWA"] = float(rwa)
        st["RORWA"] = float(np.clip(rorwa, -0.30, 0.30))
        st["EVA"] = float(eva)
        st["RONA"] = float(rona)
        st["rate"] = float(rate)

        st["EL"] = float(el_life)
        st["EL_annual"] = float(el_ann)
        st["NI"] = float(ni)

        dpd = _safe_float(st.get("DPD", 180.0), 180.0)
        dpd = float(max(dpd, 0.0))
        st["DPD"] = dpd
        st["meses_en_default"] = max(int(dpd // 30), 0)

        # secured: respeta input si viene; si no, deriva solo por mortgage
        if "secured" in st and st.get("secured") is not None:
            st["secured"] = bool(_safe_bool(st.get("secured"), default=False))
        else:
            st["secured"] = bool(segment in ("MORTGAGE", "HIPOTECARIO", "HIPOTECA"))

        ingreso = _safe_float(st.get("ingreso_mensual", None), max(1.0, ead / 24_000.0))
        cuota = _safe_float(st.get("cuota_mensual", None), (ead * rate / 12.0 if ead > 0 else 0.0))
        st["ingreso_mensual"] = float(max(ingreso, 1.0))
        st["cuota_mensual"] = float(max(cuota, 0.0))

        pti = float(st["cuota_mensual"] / max(st["ingreso_mensual"], 1.0))
        st["PTI"] = float(np.clip(pti, 0.0, 5.0))

        cf = st.get("cashflow_operativo_mensual", None)
        if cf is not None:
            dscr = _safe_float(cf, 0.0) / max(st["cuota_mensual"], 1e-6)
        else:
            dscr = 0.0 if st["PTI"] <= 1e-12 else float(1.0 / st["PTI"])
        st["DSCR"] = float(np.clip(dscr, 0.0, 10.0))

        st.setdefault("cured", False)
        st.setdefault("closed", False)
        st.setdefault("estado", "DEFAULT")

        st["loan_id"] = _as_str_id(st.get("loan_id"), "loan_unknown")
        return st

    # =====================================================================
    #      🔹 OBS MICRO (LoanEnv) + PREDICCIÓN PPO MICRO + RANKING
    # =====================================================================
    def _build_raw_micro_obs(self, row: Dict[str, Any]) -> np.ndarray:
        """
        Obs RAW (1x10) en el orden estándar:
        [EAD, PD, LGD, RW, EVA, RONA, RORWA, rating_num, segmento_id, DPD/30]

        Consistencia con LoanEnv:
          NI = EAD*rate - EL_annual - EAD*COST_FUND
          EL_annual = (PD*LGD*EAD)/horizon_years
          EVA = RWA*(RORWA - hurdle)
        """
        EAD = _safe_float(row.get("EAD", 0.0), 0.0)
        PD = _safe_float(row.get("PD", 0.0), 0.0)
        LGD = _safe_float(row.get("LGD", 0.0), 0.0)

        RW = self._coerce_rw(row.get("RW", 1.5))
        if not np.isfinite(RW):
            RW = 1.50
        RW = 1.00 if RW < 1.25 else 1.50

        rate = _safe_float(row.get("rate", 0.0), 0.0)

        el_life = float(PD * LGD * EAD)
        el_ann = float(el_life / max(self.horizon_years, 1e-6))
        NI = float(EAD * rate - el_ann - EAD * COST_FUND)

        RONA = 0.0 if EAD <= 1e-12 else float(NI / EAD)
        RWA = float(EAD * RW)
        RORWA = float(np.clip((NI / RWA) if RWA > 0 else 0.0, -0.30, 0.30))
        EVA = float(RWA * (RORWA - self.hurdle))

        rating = str(row.get("rating", "BBB")).strip().upper()
        rating_num = float(self._rating_num_map().get(rating, _safe_float(row.get("rating_num", 6.0), 6.0)))

        seg = str(row.get("segment", "CORPORATE")).strip().upper()
        segmento_id = float(self._segment_id_map().get(seg, _safe_float(row.get("segmento_id", 1.0), 1.0)))

        DPD_30 = _safe_float(row.get("DPD", 120.0), 120.0) / 30.0

        obs = np.array([EAD, PD, LGD, RW, EVA, RONA, RORWA, rating_num, segmento_id, DPD_30], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
        return obs

    def _predict_micro_action(self, row: Dict[str, Any]) -> Optional[int]:
        if self.ppo_micro is None:
            return None
        try:
            raw_obs = self._build_raw_micro_obs(row)

            # ✅ si es adapter, él decide cómo normalizar (no tocar)
            if self._micro_is_adapter():
                a_pred, _ = self.ppo_micro.predict(raw_obs, deterministic=self.micro_deterministic)
                return int(np.squeeze(a_pred))

            obs = raw_obs
            if self.micro_vn is not None and hasattr(self.micro_vn, "normalize_obs"):
                obs = self.micro_vn.normalize_obs(raw_obs)
            elif self.micro_vec_env is not None and hasattr(self.micro_vec_env, "normalize_obs"):
                obs = self.micro_vec_env.normalize_obs(raw_obs)

            a_pred, _ = self.ppo_micro.predict(obs, deterministic=self.micro_deterministic)
            return int(np.squeeze(a_pred))
        except Exception:
            return None

    def _rank_candidates_with_micro(
        self,
        indices: List[int],
        prefer: str,  # "sell" o "restruct"
        tie_key: str,
        tie_desc: bool,
    ) -> List[int]:
        if self.ppo_micro is None or not indices:
            return indices

        loans = self.portfolio
        want = 2 if prefer == "sell" else 1

        def tie_value(i: int) -> float:
            return _safe_float(loans[i].get(tie_key, 0.0), 0.0)

        scored: List[Tuple[int, int, float]] = []
        for i in indices:
            act = self._predict_micro_action(loans[i])
            hit = 1 if (act is not None and act == want) else 0
            scored.append((i, hit, tie_value(i)))

        if tie_desc:
            scored.sort(key=lambda t: (-t[1], -t[2]))
        else:
            scored.sort(key=lambda t: (-t[1], t[2]))

        return [i for (i, _, _) in scored]

    # =====================================================================
    #         SELECCIÓN TOP-K
    # =====================================================================
    def _active_indices(self) -> List[int]:
        return [i for i, l in enumerate(self.portfolio) if not bool(l.get("closed", False))]

    def _select_topk(
        self,
        key: str,
        k: int,
        reverse: bool = False,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[int]:
        idxs = self._active_indices()
        loans = self.portfolio

        def cond(i: int) -> bool:
            if condition is None:
                return True
            return bool(condition(loans[i]))

        filtered = [i for i in idxs if cond(i)]
        if not filtered:
            return []

        sorted_idx = sorted(
            filtered,
            key=lambda i: _safe_float(loans[i].get(key, 0.0), 0.0),
            reverse=reverse,
        )
        return sorted_idx[: max(1, int(k))]

    # =====================================================================
    #                     DINÁMICAS MICRO — NPL REAL
    # =====================================================================
    def _apply_maintain(self, loan: Dict[str, Any]) -> None:
        pd0 = float(loan["PD"])
        lgd0 = float(loan["LGD"])
        ead0 = float(loan["EAD"])
        rate0 = float(loan["rate"])
        rw0 = float(loan["RW"])
        cured0 = bool(loan.get("cured", False))

        if cured0:
            drift_pd = self.rng.uniform(0.90, 0.97)
            drift_lgd = self.rng.uniform(0.98, 1.01)
        else:
            drift_pd = self.rng.uniform(1.01, 1.05)
            drift_lgd = self.rng.uniform(1.00, 1.04)

        macro = {"baseline": 1.00, "adverse": 1.05, "severe": 1.15}.get(self.scenario, 1.00)

        pd1 = float(np.clip(pd0 * drift_pd * macro, 0.05, 1.00))
        lgd1 = float(np.clip(lgd0 * drift_lgd * macro, 0.20, 0.95))

        # ✅ cured no “vuelve” a delinquir por step tick
        dpd0 = float(loan.get("DPD", 0.0))
        dpd1 = 0.0 if cured0 else float(max(dpd0, 120.0) + 30.0)

        loan["PD"] = pd1
        loan["LGD"] = lgd1
        loan["DPD"] = dpd1
        loan["meses_en_default"] = int(dpd1 // 30) if dpd1 > 0 else 0

        rwa1 = float(ead0 * rw0)

        # ✅ EL lifetime + annualization para NI
        el_life = float(pd1 * lgd1 * ead0)
        el_ann = float(el_life / self.horizon_years)
        ni1 = float(ead0 * rate0 - el_ann - ead0 * COST_FUND)

        loan["EL"] = float(el_life)
        loan["EL_annual"] = float(el_ann)
        loan["NI"] = float(ni1)

        loan["RWA"] = float(rwa1)
        loan["RORWA"] = float(np.clip(self._rorwa(ni1, rwa1), -0.30, 0.30))
        loan["EVA"] = float(self._eva(loan["RORWA"], rwa1))
        loan["RONA"] = float(0.0 if ead0 <= 1e-12 else ni1 / ead0)

        ingreso = float(loan.get("ingreso_mensual", max(1.0, ead0 / 24_000)))
        cuota = float(loan.get("cuota_mensual", ead0 * rate0 / 12.0))
        loan["ingreso_mensual"] = ingreso
        loan["cuota_mensual"] = cuota
        pti = cuota / max(ingreso, 1.0)
        loan["PTI"] = float(np.clip(pti, 0.0, 5.0))
        dscr = float(1.0 / loan["PTI"]) if loan["PTI"] > 0 else 0.0
        loan["DSCR"] = float(np.clip(dscr, 0.0, 10.0))

    # =====================================================================
    #                 REESTRUCTURAR
    # =====================================================================
    def _apply_restructure(self, loan: Dict[str, Any]) -> Tuple[float, float]:
        eva0 = float(loan["EVA"])
        pd0 = float(loan["PD"])
        lgd0 = float(loan["LGD"])
        ead0 = float(loan["EAD"])
        ingreso = float(loan.get("ingreso_mensual", max(1.0, ead0 / 24_000)))
        cured0 = bool(loan.get("cured", False))

        best_gain = -1e18
        best_terms = None
        best_updates: Dict[str, Any] = {}

        esfuerzo_max = float(getattr(self.bank_strategy, "esfuerzo_alto", self.restruct_cfg.esfuerzo_umbral_alto))
        dscr_min = float(getattr(self.bank_strategy, "dscr_min", 1.10))

        pd_min = float(self.restruct_cfg.pd_min)
        pd_max = float(self.restruct_cfg.pd_max)

        # ✅ costes postura-aware (fallback reward_cfg)
        admin_cost_abs = self._p("restructure_admin_cost_abs", float(getattr(self.reward_cfg, "restructure_admin_cost_abs", 0.0)))
        quita_bps = self._p("restructure_cost_quita_bps", float(getattr(self.reward_cfg, "restructure_cost_quita_bps", 0.0)))

        cf = loan.get("cashflow_operativo_mensual", None)
        cf_val = _safe_float(cf, 0.0) if cf is not None else None

        for plazo in self.restruct_cfg.plazo_anios_grid:
            for tasa in self.restruct_cfg.tasa_anual_grid:
                for quita in self.restruct_cfg.quita_grid:
                    ead_c = ead0 * (1 - quita)

                    pd_c = pd0
                    pd_c *= (1 - self.sens_cfg.pd_reduction_per_year * plazo)
                    pd_c *= (1 - self.sens_cfg.pd_reduction_per_quita * quita)
                    pd_c = float(np.clip(pd_c, pd_min, pd_max))

                    lgd_c = lgd0 * (1 - self.sens_cfg.lgd_reduction_per_quita * quita)
                    lgd_c = float(np.clip(lgd_c, 0.0, 1.0))

                    cuota_c = ead_c * tasa / 12.0 if ead_c > 0 else 0.0
                    pti_c = cuota_c / max(ingreso, 1.0)
                    if pti_c > esfuerzo_max:
                        continue

                    # DSCR gate si hay CF disponible
                    if cf_val is not None:
                        dscr_c = cf_val / max(cuota_c, 1e-6)
                        if dscr_c < dscr_min:
                            continue

                    rwa_c = float(ead_c * float(loan["RW"]))

                    # ✅ EL lifetime + annualization para NI
                    el_life_c = float(pd_c * lgd_c * ead_c)
                    el_ann_c = float(el_life_c / self.horizon_years)
                    ni_c = float(ead_c * tasa - el_ann_c - ead_c * COST_FUND)

                    admin = float(admin_cost_abs)
                    quita_cost = (float(quita_bps) / 10_000.0) * (quita * ead0)
                    ni_c -= (admin + quita_cost)

                    rorwa_c = self._rorwa(ni_c, rwa_c)
                    eva_c = self._eva(rorwa_c, rwa_c)

                    gain = eva_c - eva0
                    if gain > best_gain:
                        best_gain = gain
                        best_terms = {"plazo": plazo, "tasa": tasa, "quita": quita}
                        best_updates = {
                            "EAD": float(ead_c),
                            "PD": float(pd_c),
                            "LGD": float(lgd_c),
                            "rate": float(tasa),
                            "RWA": float(rwa_c),
                            "EVA": float(eva_c),
                            "RORWA": float(np.clip(rorwa_c, -0.30, 0.30)),
                            "RONA": float(0.0 if ead_c <= 1e-12 else (ni_c / ead_c)),
                            "cuota_mensual": float(cuota_c),
                            "PTI": float(np.clip(pti_c, 0.0, 5.0)),
                            "DSCR": float(np.clip(
                                (cf_val / max(cuota_c, 1e-6)) if cf_val is not None else (1 / pti_c if pti_c > 0 else 0.0),
                                0.0, 10.0
                            )),
                            "DPD": 60.0,
                            "meses_en_default": 2,
                            "EL": float(el_life_c),
                            "EL_annual": float(el_ann_c),
                            "NI": float(ni_c),
                        }

        cure_bonus = 0.0
        restruct_cost = 0.0

        if best_terms is not None:
            loan.update(best_updates)
            restruct_cost = float(admin_cost_abs) + (float(quita_bps) / 10_000.0) * (float(best_terms["quita"]) * ead0)

            # audit trail
            loan["restruct_plazo"] = float(best_terms["plazo"])
            loan["restruct_tasa"] = float(best_terms["tasa"])
            loan["restruct_quita"] = float(best_terms["quita"])
            loan["restruct_cost"] = float(restruct_cost)
            loan["estado"] = "REESTRUCTURADO"

            if float(loan["PD"]) < float(self.sens_cfg.pd_cure_threshold):
                if not cured0:
                    base_eva = eva0 if eva0 != 0 else float(loan["EVA"])
                    cure_bonus = 0.2 * abs(base_eva)

                loan["cured"] = True
                loan["RW"] = float(self.sens_cfg.rw_perf_guess)  # performing proxy
                loan["DPD"] = 0.0
                loan["meses_en_default"] = 0
        else:
            restruct_cost = float(admin_cost_abs) * 0.25
            loan["restruct_cost"] = float(restruct_cost)

        # Recalcular métricas post (con annualization + coste reestructura)
        rwa1 = float(loan["EAD"]) * float(loan["RW"])
        loan["RWA"] = float(rwa1)

        el_life_1 = float(loan["PD"]) * float(loan["LGD"]) * float(loan["EAD"])
        el_ann_1 = float(el_life_1 / self.horizon_years)
        ni1 = float(loan["EAD"]) * float(loan["rate"]) - el_ann_1 - float(loan["EAD"]) * COST_FUND - float(restruct_cost)

        loan["EL"] = float(el_life_1)
        loan["EL_annual"] = float(el_ann_1)
        loan["NI"] = float(ni1)

        loan["RORWA"] = float(np.clip(self._rorwa(ni1, rwa1), -0.30, 0.30))
        loan["EVA"] = float(self._eva(float(loan["RORWA"]), rwa1))
        loan["RONA"] = float(0.0 if float(loan["EAD"]) <= 1e-12 else (ni1 / float(loan["EAD"])))

        return float(restruct_cost), float(cure_bonus)

    # =====================================================================
    #                        VENDER (con guardrail fire-sale vs book)
    # =====================================================================
    def _fire_sale_limits(self) -> Tuple[float, float]:
        """
        Returns: (max_loss_pct_of_book, min_price_to_book)

        max_loss_eur = -max_loss_pct_of_book * book_value
        """
        if self.bank_profile == cfg.BankProfile.PRUDENTE:
            return 0.15, 0.90
        if self.bank_profile == cfg.BankProfile.BALANCEADO:
            return 0.25, 0.85
        return 0.45, 0.70  # DESINVERSION

    def _apply_sell(self, loan: Dict[str, Any]) -> Tuple[float, float, float, bool]:
        """
        Returns: (capital_release, sell_cost, pnl, sold_executed)

        Fire-sale guardrail:
          - Se evalúa contra book_value: px_book = precio_optimo / book_value
          - y contra pnl contable (vs book_value) con umbral max_loss_eur

        Si fire-sale bloqueado (prudente/balanceado y allow_fire_sale=False):
          - NO cierra
          - registra sell_blocked + reason
          - aplica maintain (pasa el tiempo)
          - devuelve pnl negativo como penalización
        """
        ead0 = float(loan["EAD"])
        lgd0 = float(loan["LGD"])
        pd0 = float(loan["PD"])
        dpd0 = float(loan.get("DPD", 180))
        secured = bool(loan.get("secured", False))
        rwa0 = float(loan["RWA"])

        # ✅ book/coverage robustos
        cov = _safe_float(loan.get("coverage_rate", 0.0), 0.0)
        if cov > 1.0:
            cov = cov / 100.0
        cov = float(np.clip(cov, 0.0, 1.0))

        book_value = _safe_float(loan.get("book_value", ead0 * (1.0 - cov)), ead0 * (1.0 - cov))
        book_value = float(max(book_value, 1e-9))

        sim = simulate_npl_price(
            ead=ead0,
            lgd=lgd0,
            pd=pd0,
            dpd=dpd0,
            segment=str(loan.get("segment", "CORPORATE")).upper(),
            secured=secured,
            rating=str(loan.get("rating", "BBB")).upper(),
            book_value=book_value,
            coverage_rate=cov,
        )

        pnl = 0.0
        sell_cost = 0.0
        capital_release = rwa0 * self.cap_ratio
        precio_optimo = 0.0

        if sim is not None:
            pnl = float(sim.get("pnl", 0.0))
            sell_cost = float(sim.get("coste_tx", 0.0))
            capital_release = float(sim.get("capital_liberado", rwa0 * self.cap_ratio))
            precio_optimo = float(sim.get("precio_optimo", 0.0))

        px_ead = (precio_optimo / ead0) if ead0 > 1e-9 else 0.0
        px_book = (precio_optimo / book_value) if book_value > 1e-9 else 0.0

        max_loss_pct, min_px_book = self._fire_sale_limits()
        max_loss_eur = -float(max_loss_pct) * float(book_value)

        is_fire_sale = (px_book < float(min_px_book)) or (pnl <= float(max_loss_eur))

        # ✅ registrar SIEMPRE fire-sale (se venda o se bloquee)
        loan["Fire_Sale"] = bool(is_fire_sale)
        loan["sell_max_loss_pct_book"] = float(max_loss_pct)
        loan["sell_max_loss_eur"] = float(max_loss_eur)

        # Penalización fire-sale sobre P&L si vendes EVA positiva (guardrail “soft”)
        penalty_fire = self._p("penalty_fire_sale", 0.5)
        eva_pre = float(loan.get("EVA", 0.0))
        if eva_pre > 0:
            pnl -= abs(eva_pre) * float(penalty_fire)

        # Si fire-sale y no permitido en perfiles no-desinversión -> bloquear ejecución
        if (not self.allow_fire_sale) and is_fire_sale and (self.bank_profile != cfg.BankProfile.DESINVERSION):
            loan["sell_blocked"] = True
            loan["sell_block_reason"] = (
                f"fire_sale(px_book={px_book:.2f} < {min_px_book:.2f} OR pnl={pnl:,.0f} <= {max_loss_eur:,.0f})"
            )

            loan["sell_precio_optimo"] = float(precio_optimo)
            loan["sell_px_ead"] = float(px_ead)
            loan["sell_px_book"] = float(px_book)
            loan["sell_book_value"] = float(book_value)
            loan["sell_coverage_rate"] = float(cov)

            loan["sell_pnl"] = float(pnl)
            loan["sell_capital_liberado"] = 0.0
            loan["sell_coste_tx"] = float(sell_cost)

            # time passes: si no podemos vender, el loan sigue su dinámica
            self._apply_maintain(loan)

            # penalización explícita (en €) para desalentar intentos repetidos
            blocked_penalty = -abs(eva_pre) * max(float(penalty_fire), 0.25)
            return 0.0, 0.0, float(blocked_penalty), False

        # Venta ejecutada: guardar traza y cerrar
        loan["sell_blocked"] = False
        loan["sell_precio_optimo"] = float(precio_optimo)
        loan["sell_px_ead"] = float(px_ead)
        loan["sell_px_book"] = float(px_book)
        loan["sell_book_value"] = float(book_value)
        loan["sell_coverage_rate"] = float(cov)

        loan["sell_pnl"] = float(pnl)
        loan["sell_capital_liberado"] = float(capital_release)
        loan["sell_coste_tx"] = float(sell_cost)

        loan.update(
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
                "RONA": 0.0,
                "EVA": 0.0,
                "RORWA": 0.0,
                "DPD": 0.0,
                "meses_en_default": 0,
                "PTI": 0.0,
                "DSCR": 0.0,
                "cuota_mensual": 0.0,
                "closed": True,
                "estado": "VENDIDO",
            }
        )

        return float(capital_release), float(sell_cost), float(pnl), True

    # =====================================================================
    #                      CONSTRUCCIÓN DEL ESTADO MACRO
    # =====================================================================
    def _normalize_segment_bucket(self, seg: str) -> str:
        s = (seg or "OTHER").strip().upper()
        if s in ("LARGE CORPORATE", "CORP", "CORPORATE"):
            return "CORPORATE"
        if s in ("PYME", "SME", "MIDCAP"):
            return "SME"
        if s in ("RETAIL", "MINORISTA"):
            return "RETAIL"
        if s in ("MORTGAGE", "HIPOTECARIO", "HIPOTECA"):
            return "MORTGAGE"
        if s in ("CONSUMER", "CONSUMO"):
            return "CONSUMER"
        if s in ("BANK", "SOVEREIGN", "LEASING", "OTHER"):
            return s
        return "OTHER"

    def _build_obs(self) -> np.ndarray:
        loans = [l for l in self.portfolio if not l.get("closed", False)]
        if not loans:
            return np.zeros(self.state_dim, dtype=np.float32)

        eads = np.array([_safe_float(l.get("EAD", 0.0), 0.0) for l in loans], dtype=float)
        rwas = np.array([_safe_float(l.get("RWA", 0.0), 0.0) for l in loans], dtype=float)
        evas = np.array([_safe_float(l.get("EVA", 0.0), 0.0) for l in loans], dtype=float)
        pds = np.array([_safe_float(l.get("PD", 0.0), 0.0) for l in loans], dtype=float)
        lgds = np.array([_safe_float(l.get("LGD", 0.0), 0.0) for l in loans], dtype=float)
        rors = np.array([_safe_float(l.get("RORWA", 0.0), 0.0) for l in loans], dtype=float)

        total_ead = float(eads.sum())
        total_rwa = float(rwas.sum())
        total_eva = float(evas.sum())
        total_risk = float((pds * lgds * eads).sum())  # EL lifetime proxy
        num_loans = float(len(loans))

        avg_pd = float(pds.mean()) if num_loans > 0 else 0.0
        avg_lgd = float(lgds.mean()) if num_loans > 0 else 0.0
        avg_rorwa = float(rors.mean()) if num_loans > 0 else 0.0

        seg_keys = ["SOVEREIGN", "BANK", "CORPORATE", "SME", "RETAIL", "MORTGAGE", "CONSUMER", "LEASING", "OTHER"]
        seg_ead = {k: 0.0 for k in seg_keys}
        if total_ead > 0:
            for l in loans:
                seg = self._normalize_segment_bucket(str(l.get("segment", "OTHER")))
                seg_ead[seg] = seg_ead.get(seg, 0.0) + _safe_float(l.get("EAD", 0.0), 0.0)
        seg_share = [float(seg_ead[k] / total_ead) if total_ead > 0 else 0.0 for k in seg_keys]

        rating_keys = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        rat_ead = {k: 0.0 for k in rating_keys}
        if total_ead > 0:
            for l in loans:
                rat = str(l.get("rating", "BBB")).upper()
                if rat not in rat_ead:
                    rat = "BBB"
                rat_ead[rat] += _safe_float(l.get("EAD", 0.0), 0.0)
        rat_share = [float(rat_ead[k] / total_ead) if total_ead > 0 else 0.0 for k in rating_keys]

        features: List[float] = [
            total_ead, total_rwa, total_eva, total_risk, num_loans, avg_pd, avg_lgd, avg_rorwa
        ]
        features += seg_share
        features += rat_share

        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        else:
            features = features[: self.state_dim]

        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_obs:
            obs = np.clip(obs, -1e6, 1e6)

        return obs

    # =====================================================================
    #                             RESET
    # =====================================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # ✅ refrescar config/bank profile por si policy_inference ha hecho set_bank_profile sin reload
        self._refresh_cfg_aliases()

        # micro normalizer: solo re-inicializar si NO es adapter (y si el path pudiera cambiar)
        if not self._micro_is_adapter():
            self._init_micro_normalizer()

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.eva_history.clear()
        self._reset_portfolio_copy()

        obs = self._build_obs()
        loans = [l for l in self.portfolio if not l.get("closed", False)]
        eva0 = float(sum(_safe_float(l.get("EVA", 0.0), 0.0) for l in loans))
        self.eva_history.append(eva0)

        info = {
            "portfolio_metrics": {
                "EAD_total": float(sum(_safe_float(l.get("EAD", 0.0), 0.0) for l in loans)),
                "RWA_total": float(sum(_safe_float(l.get("RWA", 0.0), 0.0) for l in loans)),
                "EVA_total": eva0,
                "num_loans_active": int(len(loans)),
                "num_loans_closed": int(len(self.portfolio) - len(loans)),
                "eva_volatility": 0.0,
                "hhi_segment": 0.0,
                "hhi_rating": 0.0,
            },
            "action_summary": {
                "action": None,
                "sold_ids": [],
                "restructured_ids": [],
                "blocked_sell_ids": [],
            },
        }
        return obs, info

    def _concentration_metrics(self, loans: List[Dict[str, Any]]) -> Tuple[float, float]:
        activos = [l for l in loans if _safe_float(l.get("EAD", 0.0), 0.0) > 0.0 and not l.get("closed", False)]
        if len(activos) == 0:
            return 0.0, 0.0

        ead_total = sum(_safe_float(l.get("EAD", 0.0), 0.0) for l in activos)
        if ead_total <= 0:
            return 0.0, 0.0

        ead_por_segmento: Dict[str, float] = {}
        for l in activos:
            seg = self._normalize_segment_bucket(str(l.get("segment", "OTHER")))
            ead_por_segmento[seg] = ead_por_segmento.get(seg, 0.0) + _safe_float(l.get("EAD", 0.0), 0.0)

        hhi_seg = 0.0
        for ead_seg in ead_por_segmento.values():
            w = ead_seg / ead_total
            hhi_seg += w * w

        ead_por_rating: Dict[str, float] = {}
        for l in activos:
            r = str(l.get("rating", "BBB")).upper()
            if r not in ("AAA", "AA", "A", "BBB", "BB", "B", "CCC"):
                r = "BBB"
            ead_por_rating[r] = ead_por_rating.get(r, 0.0) + _safe_float(l.get("EAD", 0.0), 0.0)

        hhi_rat = 0.0
        for ead_rat in ead_por_rating.values():
            w = ead_rat / ead_total
            hhi_rat += w * w

        return float(np.clip(hhi_seg, 0.0, 1.0)), float(np.clip(hhi_rat, 0.0, 1.0))

    # =====================================================================
    #                             STEP
    # =====================================================================
    def step(self, action: int):
        self.steps += 1
        action = int(action)

        active_idxs = self._active_indices()
        loans_before = [self.portfolio[i] for i in active_idxs]

        eva0 = float(sum(_safe_float(l.get("EVA", 0.0), 0.0) for l in loans_before))
        rwa0 = float(sum(_safe_float(l.get("RWA", 0.0), 0.0) for l in loans_before))
        risk0 = float(sum(
            _safe_float(l.get("PD", 0.0), 0.0)
            * _safe_float(l.get("LGD", 0.0), 0.0)
            * _safe_float(l.get("EAD", 0.0), 0.0)
            for l in loans_before
        ))

        capital_release_total = 0.0
        cure_bonus_total = 0.0
        total_restruct_cost = 0.0
        total_sell_cost = 0.0
        total_pnl = 0.0

        sold_ids: List[str] = []
        blocked_sell_ids: List[str] = []
        restructured_ids: List[str] = []

        eva_sold_neg = 0.0
        eva_sold_pos = 0.0

        top_k_eff = self._effective_top_k()

        # ------------------------
        # APLICACIÓN ACCIONES
        # ------------------------
        if action == 0 or action == 11:
            for i in active_idxs:
                self._apply_maintain(self.portfolio[i])

        elif action == 1:
            idxs = self._select_topk("EVA", k=1, reverse=False)
            idxs = self._rank_candidates_with_micro(idxs, prefer="sell", tie_key="EVA", tie_desc=False)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                eva_loan = _safe_float(loan.get("EVA", 0.0), 0.0)

                cap, c, pnl, sold = self._apply_sell(loan)
                capital_release_total += cap
                total_sell_cost += c
                total_pnl += pnl

                if sold:
                    sold_ids.append(loan_id)
                    eva_sold_neg += (-eva_loan) if eva_loan < 0 else 0.0
                    eva_sold_pos += eva_loan if eva_loan > 0 else 0.0
                else:
                    blocked_sell_ids.append(loan_id)

        elif action == 2:
            idxs = self._select_topk("EVA", k=top_k_eff, reverse=False)
            idxs = self._rank_candidates_with_micro(idxs, prefer="sell", tie_key="EVA", tie_desc=False)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                eva_loan = _safe_float(loan.get("EVA", 0.0), 0.0)

                cap, c, pnl, sold = self._apply_sell(loan)
                capital_release_total += cap
                total_sell_cost += c
                total_pnl += pnl

                if sold:
                    sold_ids.append(loan_id)
                    eva_sold_neg += (-eva_loan) if eva_loan < 0 else 0.0
                    eva_sold_pos += eva_loan if eva_loan > 0 else 0.0
                else:
                    blocked_sell_ids.append(loan_id)

        elif action == 3:
            idxs = self._select_topk("EVA", k=1, reverse=False)
            idxs = self._rank_candidates_with_micro(idxs, prefer="restruct", tie_key="EVA", tie_desc=False)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                rwa_b = _safe_float(loan.get("RWA", 0.0), 0.0)
                rc, cb = self._apply_restructure(loan)
                rwa_a = _safe_float(loan.get("RWA", 0.0), 0.0)
                cap_rel = max(rwa_b - rwa_a, 0.0) * self.cap_ratio
                capital_release_total += cap_rel

                total_restruct_cost += rc
                cure_bonus_total += cb
                restructured_ids.append(loan_id)

        elif action == 4:
            idxs = self._select_topk("EVA", k=top_k_eff, reverse=False)
            idxs = self._rank_candidates_with_micro(idxs, prefer="restruct", tie_key="EVA", tie_desc=False)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                rwa_b = _safe_float(loan.get("RWA", 0.0), 0.0)
                rc, cb = self._apply_restructure(loan)
                rwa_a = _safe_float(loan.get("RWA", 0.0), 0.0)
                cap_rel = max(rwa_b - rwa_a, 0.0) * self.cap_ratio
                capital_release_total += cap_rel

                total_restruct_cost += rc
                cure_bonus_total += cb
                restructured_ids.append(loan_id)

        elif action == 5:
            idxs = self._select_topk("RORWA", k=1, reverse=False)
            idxs = self._rank_candidates_with_micro(idxs, prefer="sell", tie_key="EVA", tie_desc=False)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                eva_loan = _safe_float(loan.get("EVA", 0.0), 0.0)

                cap, c, pnl, sold = self._apply_sell(loan)
                capital_release_total += cap
                total_sell_cost += c
                total_pnl += pnl

                if sold:
                    sold_ids.append(loan_id)
                    eva_sold_neg += (-eva_loan) if eva_loan < 0 else 0.0
                    eva_sold_pos += eva_loan if eva_loan > 0 else 0.0
                else:
                    blocked_sell_ids.append(loan_id)

        elif action == 6:
            idxs = self._select_topk("RORWA", k=top_k_eff, reverse=False)
            idxs = self._rank_candidates_with_micro(idxs, prefer="sell", tie_key="EVA", tie_desc=False)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                eva_loan = _safe_float(loan.get("EVA", 0.0), 0.0)

                cap, c, pnl, sold = self._apply_sell(loan)
                capital_release_total += cap
                total_sell_cost += c
                total_pnl += pnl

                if sold:
                    sold_ids.append(loan_id)
                    eva_sold_neg += (-eva_loan) if eva_loan < 0 else 0.0
                    eva_sold_pos += eva_loan if eva_loan > 0 else 0.0
                else:
                    blocked_sell_ids.append(loan_id)

        elif action == 7:
            idxs = self._select_topk("PTI", k=1, reverse=True, condition=lambda l: not l.get("cured", False))
            idxs = self._rank_candidates_with_micro(idxs, prefer="restruct", tie_key="PTI", tie_desc=True)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                rwa_b = _safe_float(loan.get("RWA", 0.0), 0.0)
                rc, cb = self._apply_restructure(loan)
                rwa_a = _safe_float(loan.get("RWA", 0.0), 0.0)
                cap_rel = max(rwa_b - rwa_a, 0.0) * self.cap_ratio
                capital_release_total += cap_rel

                total_restruct_cost += rc
                cure_bonus_total += cb
                restructured_ids.append(loan_id)

        elif action == 8:
            idxs = self._select_topk("PTI", k=top_k_eff, reverse=True, condition=lambda l: not l.get("cured", False))
            idxs = self._rank_candidates_with_micro(idxs, prefer="restruct", tie_key="PTI", tie_desc=True)
            for i in idxs:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                rwa_b = _safe_float(loan.get("RWA", 0.0), 0.0)
                rc, cb = self._apply_restructure(loan)
                rwa_a = _safe_float(loan.get("RWA", 0.0), 0.0)
                cap_rel = max(rwa_b - rwa_a, 0.0) * self.cap_ratio
                capital_release_total += cap_rel

                total_restruct_cost += rc
                cure_bonus_total += cb
                restructured_ids.append(loan_id)

        elif action == 9:
            idx_sell = self._select_topk(
                "EVA", k=1, reverse=False,
                condition=lambda l: _safe_float(l.get("EVA", 0.0), 0.0) < 0
            )
            idx_sell = self._rank_candidates_with_micro(idx_sell, prefer="sell", tie_key="EVA", tie_desc=False)
            for i in idx_sell:
                loan = self.portfolio[i]
                loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                eva_loan = _safe_float(loan.get("EVA", 0.0), 0.0)

                cap, c, pnl, sold = self._apply_sell(loan)
                capital_release_total += cap
                total_sell_cost += c
                total_pnl += pnl

                if sold:
                    sold_ids.append(loan_id)
                    eva_sold_neg += (-eva_loan) if eva_loan < 0 else 0.0
                    eva_sold_pos += eva_loan if eva_loan > 0 else 0.0
                else:
                    blocked_sell_ids.append(loan_id)

            idx_restr = self._select_topk("PTI", k=1, reverse=True, condition=lambda l: not l.get("cured", False))
            idx_restr = self._rank_candidates_with_micro(idx_restr, prefer="restruct", tie_key="PTI", tie_desc=True)
            for i in idx_restr:
                if not self.portfolio[i].get("closed", False):
                    loan = self.portfolio[i]
                    loan_id = _as_str_id(loan.get("loan_id"), f"loan_{i}")
                    rwa_b = _safe_float(loan.get("RWA", 0.0), 0.0)
                    rc, cb = self._apply_restructure(loan)
                    rwa_a = _safe_float(loan.get("RWA", 0.0), 0.0)
                    cap_rel = max(rwa_b - rwa_a, 0.0) * self.cap_ratio
                    capital_release_total += cap_rel

                    total_restruct_cost += rc
                    cure_bonus_total += cb
                    restructured_ids.append(loan_id)

        elif action == 10:
            esfuerzo_bajo = float(getattr(self.bank_strategy, "esfuerzo_bajo", self.restruct_cfg.esfuerzo_umbral_bajo))

            neg_eva_idxs = self._select_topk(
                "EAD",
                k=self._effective_top_k(),
                reverse=True,
                condition=lambda l: _safe_float(l.get("EVA", 0.0), 0.0) < 0.0,
            )
            neg_eva_idxs = self._rank_candidates_with_micro(neg_eva_idxs, prefer="sell", tie_key="EVA", tie_desc=False)

            for i in neg_eva_idxs:
                l = self.portfolio[i]
                if l.get("closed", False):
                    continue
                loan_id = _as_str_id(l.get("loan_id"), f"loan_{i}")

                if _safe_float(l.get("PTI", 0.0), 0.0) <= esfuerzo_bajo:
                    eva_loan = _safe_float(l.get("EVA", 0.0), 0.0)

                    cap, c, pnl, sold = self._apply_sell(l)
                    capital_release_total += cap
                    total_sell_cost += c
                    total_pnl += pnl

                    if sold:
                        sold_ids.append(loan_id)
                        eva_sold_neg += (-eva_loan) if eva_loan < 0 else 0.0
                        eva_sold_pos += eva_loan if eva_loan > 0 else 0.0
                    else:
                        blocked_sell_ids.append(loan_id)
                else:
                    rwa_b = _safe_float(l.get("RWA", 0.0), 0.0)
                    rc, cb = self._apply_restructure(l)
                    rwa_a = _safe_float(l.get("RWA", 0.0), 0.0)
                    cap_rel = max(rwa_b - rwa_a, 0.0) * self.cap_ratio
                    capital_release_total += cap_rel

                    total_restruct_cost += rc
                    cure_bonus_total += cb
                    restructured_ids.append(loan_id)

            for i in self._active_indices():
                if i in neg_eva_idxs:
                    continue
                self._apply_maintain(self.portfolio[i])

        else:
            # fallback defensivo: NO-OP
            for i in active_idxs:
                self._apply_maintain(self.portfolio[i])

        # ------------------------
        # POST-ACCIÓN
        # ------------------------
        active_idxs_after = self._active_indices()
        loans_after = [self.portfolio[i] for i in active_idxs_after]

        eva1 = float(sum(_safe_float(l.get("EVA", 0.0), 0.0) for l in loans_after))
        rwa1 = float(sum(_safe_float(l.get("RWA", 0.0), 0.0) for l in loans_after))
        risk1 = float(sum(
            _safe_float(l.get("PD", 0.0), 0.0)
            * _safe_float(l.get("LGD", 0.0), 0.0)
            * _safe_float(l.get("EAD", 0.0), 0.0)
            for l in loans_after
        ))

        self.eva_history.append(eva1)
        eva_gain = eva1 - eva0

        esfuerzo_bajo = float(getattr(self.bank_strategy, "esfuerzo_bajo", self.restruct_cfg.esfuerzo_umbral_bajo))
        esfuerzo_alto = float(getattr(self.bank_strategy, "esfuerzo_alto", self.restruct_cfg.esfuerzo_umbral_alto))

        pti_penalty = 0.0
        for l in loans_after:
            excess = max(0.0, _safe_float(l.get("PTI", 0.0), 0.0) - esfuerzo_bajo) / max(esfuerzo_alto, 1e-6)
            pti_penalty += excess * abs(_safe_float(l.get("EVA", 0.0), 0.0)) / 1e6

        hhi_seg, hhi_rat = self._concentration_metrics(loans_after)
        eva_vol = self._eva_volatility()

        # ✅ capital carry (no hurdle): cap_blocked * cost_of_capital * dt
        cap_blocked = rwa1 * self.cap_ratio
        capital_carry_cost = cap_blocked * float(self.cost_of_capital) * float(self.dt_years)

        # ✅ pesos/bonos por postura (fallback reward_cfg)
        w_pnl = self._p("w_pnl", float(getattr(self.reward_cfg, "w_pnl", 0.0)))
        w_eva = self._p("w_eva", float(getattr(self.reward_cfg, "w_eva", 1.0)))
        w_cap = self._p("w_capital", float(getattr(self.reward_cfg, "w_capital", 0.0)))
        w_stab = self._p("w_stab", float(getattr(self.reward_cfg, "w_stab", 0.0)))

        sell_bonus_eva_neg = self._p("sell_bonus_eva_neg", float(getattr(self.reward_cfg, "sell_bonus_eva_neg", 0.0)))
        maintain_bonus_eva_pos = self._p("maintain_bonus_eva_pos", float(getattr(self.reward_cfg, "maintain_bonus_eva_pos", 0.0)))

        clip_low, clip_high = getattr(self.reward_cfg, "clip", (-10.0, 10.0))

        eva_term = float(w_eva) * (eva_gain / 1e6)
        cap_term = float(w_cap) * (capital_release_total / 1e6)
        risk_term = -float(w_stab) * (risk1 / 1e6)
        pti_term = -0.1 * pti_penalty
        pnl_term = float(w_pnl) * (total_pnl / 1e6)
        cure_term = (cure_bonus_total / 1e6)

        vol_term = -float(self.w_vol) * (eva_vol / 1e6)
        conc_term = -float(self.w_conc) * (hhi_seg + hhi_rat)
        carry_term = -float(self.w_cap_carry) * (capital_carry_cost / 1e6)

        r = (
            eva_term
            + cap_term
            + risk_term
            + pti_term
            + pnl_term
            + cure_term
            + vol_term
            + conc_term
            + carry_term
        )

        if eva_sold_neg > 0.0:
            r += float(sell_bonus_eva_neg) * (eva_sold_neg / 1e6)

        if eva0 > 0 and action == 0:
            r += float(maintain_bonus_eva_pos) * (eva0 / 1e6)

        r = float(np.clip(r, clip_low, clip_high))

        # ✅ Gymnasium semantics
        terminated = (len(active_idxs_after) == 0)
        truncated = (not terminated) and (self.steps >= self.max_steps)

        obs = self._build_obs()

        info = {
            "portfolio_metrics": {
                "EVA_before": eva0,
                "EVA_after": eva1,
                "EVA_gain": eva_gain,
                "RWA_before": rwa0,
                "RWA_after": rwa1,
                "capital_liberado": capital_release_total,
                "risk_before": risk0,
                "risk_after": risk1,
                "num_loans_active": len(active_idxs_after),
                "num_loans_closed": len(self.portfolio) - len(active_idxs_after),
                "eva_volatility": eva_vol,
                "hhi_segment": hhi_seg,
                "hhi_rating": hhi_rat,
                "capital_carry_cost": capital_carry_cost,
                "pnl_total": total_pnl,
                "eva_sold_neg": eva_sold_neg,
                "eva_sold_pos": eva_sold_pos,
            },
            "costs": {
                "restruct_cost_total": total_restruct_cost,
                "sell_cost_total": total_sell_cost,
            },
            "bonuses": {
                "cure_bonus_total": cure_bonus_total,
            },
            "reward_terms": {
                "eva_term": eva_term,
                "cap_term": cap_term,
                "risk_term": risk_term,
                "pti_term": pti_term,
                "pnl_term": pnl_term,
                "cure_term": cure_term,
                "vol_term": vol_term,
                "conc_term": conc_term,
                "carry_term": carry_term,
                "reward_total": r,
            },
            "action_summary": {
                "action": action,
                "top_k_eff": int(top_k_eff),
                "sold_ids": sold_ids,
                "blocked_sell_ids": blocked_sell_ids,
                "restructured_ids": restructured_ids,
                "n_sold": int(len(sold_ids)),
                "n_sell_blocked": int(len(blocked_sell_ids)),
                "n_restructured": int(len(restructured_ids)),
            },
        }

        return obs, r, terminated, truncated, info

    # =====================================================================
    #                               RENDER
    # =====================================================================
    def render(self):
        loans = [l for l in self.portfolio if not l.get("closed", False)]
        total_ead = sum(_safe_float(l.get("EAD", 0.0), 0.0) for l in loans)
        total_eva = sum(_safe_float(l.get("EVA", 0.0), 0.0) for l in loans)
        total_rwa = sum(_safe_float(l.get("RWA", 0.0), 0.0) for l in loans)
        hhi_seg, hhi_rat = self._concentration_metrics(loans)
        eva_vol = self._eva_volatility()

        prof = self.bank_profile
        prof_str = getattr(prof, "value", None) or str(prof)

        print(
            f"[PortfolioEnv] step={self.steps} | active={len(loans)} "
            f"EAD={total_ead:,.0f} RWA={total_rwa:,.0f} EVA={total_eva:,.0f} | "
            f"HHI_seg={hhi_seg:.3f} HHI_rat={hhi_rat:.3f} | "
            f"EVA_vol={eva_vol:,.0f} | perfil={prof_str}"
        )
