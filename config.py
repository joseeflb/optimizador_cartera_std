# ============================================
# config.py — versión Banco L1.5 (CORREGIDO y armonizado)
# ============================================
# OPTIMIZADOR DE CARTERAS EN DEFAULT (Método Estándar · Basilea III)
#
# Coherente con:
#   ✔ LoanEnv (micro)
#   ✔ PortfolioEnv (macro)
#   ✔ restructure_optimizer
#   ✔ price_simulator
#   ✔ policy_inference y train_agent
#   ✔ arquitectura multi-agente
#
# Nota crítica (NPL):
#   - DEFAULT es un ESTADO (trigger regulatorio y operativo).
#   - PD NO es 100% por estar en default. PD se interpreta como PD forward 12–24m.
#   - El severidad de default se captura con DPD, LGD, colateral, pricing y RW (STD).
# ============================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json, os, sys, random, logging
from pathlib import Path

# ================================================================
# 🔢 Semillas
# ================================================================
GLOBAL_SEED: int = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ================================================================
# 📁 Directorios
# ================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR, TEMP_DIR = [
    ROOT_DIR / p for p in ("data", "reports", "models", "logs", "_tmp")
]
for d in (DATA_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)

MODEL_FILE = MODELS_DIR / "best_model.zip"
PIPELINE_LOG = LOGS_DIR / "pipeline.log"

# ================================================================
# 🧱 Tipos y esquema
# ================================================================
class Segmento(Enum):
    CORPORATE = "corporate"
    SME = "sme"
    RETAIL = "retail"
    MORTGAGE = "mortgage"
    SOVEREIGN = "sovereign"
    BANK = "bank"
    LEASING = "leasing"
    CONSUMER = "consumer"
    OTHER = "other"

class EstadoCredito(Enum):
    PERFORMING = "performing"
    DEFAULT = "default"

VALID_RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

ID_COL = "loan_id"

# Importante: estas 10 columnas reflejan la OBSERVACIÓN efectiva del LoanEnv.
# El DPD se ingiere en días (DPD) y en el entorno se normaliza como DPD/30.
FEATURE_COLUMNS = [
    "EAD", "PD", "LGD", "RW", "EVA", "RONA", "RORWA", "rating_num", "segmento_id", "DPD"
]

ACTION_ENUM = {"MANTENER": 0, "REESTRUCTURAR": 1, "VENDER": 2}

# ================================================================
# ✅ Convención NPL (ESTÁNDAR ÚNICO PARA TODO EL REPO)
# ================================================================
@dataclass
class NPLConventions:
    """
    Convenciones comunes para interpretar variables en NPL:

    - DEFAULT es un estado regulatorio/operativo.
    - PD representa PD forward (12–24m) condicionada a estar en default.
      No se fija a 100%: el evento "default" ya ocurrió; el riesgo relevante es
      cure vs remain-default / further loss / timeline (capturado además por DPD).
    - LGD en NPL suele estar elevada y se acota por floors/caps.
    - DPD se usa como proxy de severidad y urgencia.
    """
    # PD forward para NPL (rango típico para decisiones, calibrable por banco)
    pd_default_floor: float = 0.35
    pd_default_cap: float = 0.85

    # LGD en NPL (floors/caps)
    lgd_default_floor: float = 0.40
    lgd_default_cap: float = 0.90

    # DPD normalización / límites (para robustez en features)
    dpd_floor: float = 90.0
    dpd_cap: float = 1080.0  # 3 años

    # Semántica explícita (evita “PD=100%” en toda la base de código)
    default_is_state_not_pd: bool = True

    # Horizonte conceptual (documentación + futuras extensiones)
    pd_horizon_months: int = 12


def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except Exception:
        xf = lo
    return float(max(lo, min(hi, xf)))


def clamp_npl_pd(pd: float, npl: NPLConventions) -> float:
    """PD forward NPL acotada (evita PD=1.0 por construcción)."""
    return clamp(pd, npl.pd_default_floor, npl.pd_default_cap)


def clamp_npl_lgd(lgd: float, npl: NPLConventions) -> float:
    """LGD NPL acotada."""
    return clamp(lgd, npl.lgd_default_floor, npl.lgd_default_cap)


def clamp_dpd(dpd: float, npl: NPLConventions) -> float:
    """DPD acotado para estabilidad numérica y coherencia de features."""
    return clamp(dpd, npl.dpd_floor, npl.dpd_cap)


# ================================================================
# Basilea III STD (RW)
# ================================================================
@dataclass
class BaselSTDMapping:
    sovereign: Dict[str, float] = field(default_factory=lambda: {
        "AAA": 0.00, "AA": 0.00, "A": 0.20, "BBB": 0.50, "BB": 1.00, "B": 1.50, "CCC": 1.50})
    bank: Dict[str, float] = field(default_factory=lambda: {
        "AAA": 0.20, "AA": 0.20, "A": 0.50, "BBB": 0.50, "BB": 1.00, "B": 1.50, "CCC": 1.50})
    corporate: Dict[str, float] = field(default_factory=lambda: {
        "AAA": 0.20, "AA": 0.20, "A": 0.50, "BBB": 1.00, "BB": 1.50, "B": 1.50, "CCC": 1.50})

    retail_performing: float = 0.75
    mortgage_performing: float = 0.35
    default_unsecured: float = 1.50
    default_mortgage_residential: float = 1.00

    def resolve_rw(
        self,
        segmento: Segmento,
        rating: Optional[str],
        estado: EstadoCredito,
        secured_by_mortgage: bool = False
    ) -> float:

        # Default primero (STD): RW depende del estado, NO de forzar PD=1.0
        if estado == EstadoCredito.DEFAULT:
            if secured_by_mortgage and segmento == Segmento.MORTGAGE:
                return self.default_mortgage_residential
            return self.default_unsecured

        r = (rating or "BBB").upper()
        if r not in VALID_RATINGS:
            r = "BBB"

        if segmento == Segmento.SOVEREIGN:
            return self.sovereign.get(r, 1.0)
        if segmento == Segmento.BANK:
            return self.bank.get(r, 1.0)
        if segmento in (Segmento.CORPORATE, Segmento.SME):
            return self.corporate.get(r, 1.0)
        if segmento == Segmento.MORTGAGE:
            return self.mortgage_performing
        if segmento in (Segmento.RETAIL, Segmento.CONSUMER):
            return self.retail_performing
        return 1.0

# ================================================================
# 🧱 Regulación
# ================================================================
@dataclass
class BuffersRegulatorios:
    conservation: float = 0.025
    countercyclical: float = 0.0
    systemic: float = 0.0

    def total_buffer(self) -> float:
        return self.conservation + max(self.countercyclical, self.systemic)

@dataclass
class Regulacion:
    cet1_min: float = 0.045
    tier1_min: float = 0.06
    total_capital_min: float = 0.08
    leverage_ratio_min: float = 0.03
    hurdle_rate: float = 0.10
    buffers: BuffersRegulatorios = field(default_factory=BuffersRegulatorios)

    def required_total_capital_ratio(self) -> float:
        return self.total_capital_min + self.buffers.total_buffer()

# ================================================================
# 📊 Simulación sintética
# ================================================================
@dataclass
class SimulacionCartera:
    n_prestamos: int = 1000
    pct_segments: Dict[Segmento, float] = field(default_factory=lambda: {
        Segmento.CORPORATE: 0.2, Segmento.SME: 0.2,
        Segmento.RETAIL: 0.2, Segmento.MORTGAGE: 0.2,
        Segmento.CONSUMER: 0.2
    })
    ead_range: Tuple[float, float] = (5000, 1_000_000)

    # Performing PD/LGD (para generación base, no para NPL)
    pd_by_rating: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "AAA": (0.0005, 0.005), "AA": (0.0005, 0.008), "A": (0.001, 0.02),
        "BBB": (0.005, 0.02), "BB": (0.02, 0.06), "B": (0.06, 0.15), "CCC": (0.15, 0.30)
    })
    lgd_by_rating: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "AAA": (0.20, 0.25), "AA": (0.20, 0.28), "A": (0.22, 0.32),
        "BBB": (0.25, 0.35), "BB": (0.35, 0.45), "B": (0.45, 0.60), "CCC": (0.55, 0.70)
    })

    # NPL generation guidance (para carteras en default sintéticas coherentes)
    # Nota: NO es PD=100%. Es PD forward en default.
    npl_pd_range: Tuple[float, float] = (0.35, 0.85)
    npl_lgd_range: Tuple[float, float] = (0.40, 0.90)
    npl_dpd_range: Tuple[float, float] = (120.0, 900.0)

# ================================================================
# 🔧 Reestructuración
# ================================================================
@dataclass
class ReestructuraParams:
    plazo_anios_grid: List[int] = field(default_factory=lambda: [3, 5, 7, 10, 15, 20])
    tasa_anual_grid: List[float] = field(default_factory=lambda: [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19])
    quita_grid: List[float] = field(default_factory=lambda: [0.00, 0.05, 0.10, 0.15])
    esfuerzo_umbral_bajo: float = 0.30
    esfuerzo_umbral_alto: float = 0.50
    pd_min: float = 0.20
    pd_max: float = 0.80

@dataclass
class SensibilidadReestructura:
    pd_reduction_per_year: float = 0.08
    pd_reduction_per_quita: float = 0.35
    lgd_reduction_per_quita: float = 0.50

    cure_window_months: int = 3
    pd_cure_threshold: float = 0.20
    rw_perf_guess: float = 0.75
    horizon_months: int = 24

# ================================================================
# 🏷️ Precio de venta NPL
# ================================================================
@dataclass
class PrecioVentaParams:
    alpha_by_segment: Dict[Segmento, Tuple[float, float]] = field(default_factory=lambda: {
        Segmento.CONSUMER: (0.30, 0.55), Segmento.RETAIL: (0.40, 0.65),
        Segmento.MORTGAGE: (0.60, 0.80), Segmento.CORPORATE: (0.40, 0.60),
        Segmento.SME: (0.35, 0.60)
    })
    coste_legal_estimado_abs: Tuple[float, float] = (500, 5000)
    buyer_discount_rate: float = 0.10
    buyer_years_to_recover: Tuple[int, int] = (1, 3)

# ================================================================
# 🧠 Perfiles de banco (estrategia global)
# ================================================================
class BankProfile(Enum):
    PRUDENTE = "prudente"
    BALANCEADO = "balanceado"
    DESINVERSION = "desinversion"

@dataclass
class BankStrategy:
    """
    Estrategia de alto nivel del banco:
    - Pesa la recompensa (EVA, capital, P&L, estabilidad, etc.).
    - Define umbrales prudenciales para heurísticos / LoanEnv.
    """
    name: str

    # Pesos de reward (micro + macro)
    w_eva: float
    w_capital: float
    w_stab: float
    w_diversif: float

    w_pnl: float
    pnl_penalty_scale: float
    pnl_weight: float

    w_vol: float
    w_concentration: float
    w_cap_carry: float

    penalty_fire_sale: float
    penalty_mode_collapse: float
    sell_bonus_eva_neg: float
    maintain_bonus_eva_pos: float

    restructure_admin_cost_abs: float
    restructure_cost_quita_bps: int
    sell_tx_cost_bps: int

    # Umbrales prudenciales para heurístico / LoanEnv
    eva_min_improvement: float
    eva_strongly_neg: float
    esfuerzo_bajo: float
    esfuerzo_alto: float
    dscr_min: float

BANK_STRATEGIES: Dict[BankProfile, BankStrategy] = {
    BankProfile.PRUDENTE: BankStrategy(
        name="Banco prudente",
        w_eva=0.25,
        w_capital=0.30,
        w_stab=0.15,
        w_diversif=0.05,
        w_pnl=0.25,
        pnl_penalty_scale=1.5e5,
        pnl_weight=1.0,
        w_vol=0.04,
        w_concentration=0.08,
        w_cap_carry=0.03,
        penalty_fire_sale=0.85,
        penalty_mode_collapse=0.10,
        sell_bonus_eva_neg=0.10,
        maintain_bonus_eva_pos=0.35,
        restructure_admin_cost_abs=250.0,
        restructure_cost_quita_bps=40,
        sell_tx_cost_bps=100,
        eva_min_improvement=10_000.0,
        eva_strongly_neg=-50_000.0,
        esfuerzo_bajo=0.30,
        esfuerzo_alto=0.50,
        dscr_min=1.05,
    ),
    BankProfile.BALANCEADO: BankStrategy(
        name="Banco balanceado",
        w_eva=0.30,
        w_capital=0.45,
        w_stab=0.12,
        w_diversif=0.05,
        w_pnl=0.30,
        pnl_penalty_scale=9e4,
        pnl_weight=1.0,
        w_vol=0.03,
        w_concentration=0.06,
        w_cap_carry=0.07,
        penalty_fire_sale=0.55,
        penalty_mode_collapse=0.08,
        sell_bonus_eva_neg=0.45,
        maintain_bonus_eva_pos=0.20,
        restructure_admin_cost_abs=250.0,
        restructure_cost_quita_bps=40,
        sell_tx_cost_bps=80,
        eva_min_improvement=5_000.0,
        eva_strongly_neg=-30_000.0,
        esfuerzo_bajo=0.35,
        esfuerzo_alto=0.50,
        dscr_min=1.1,
    ),
    BankProfile.DESINVERSION: BankStrategy(
        name="Plan de desinversión NPL",
        w_eva=0.25,
        w_capital=0.60,
        w_stab=0.10,
        w_diversif=0.05,
        w_pnl=0.35,
        pnl_penalty_scale=4e4,
        pnl_weight=1.0,
        w_vol=0.02,
        w_concentration=0.05,
        w_cap_carry=0.12,
        penalty_fire_sale=0.2,
        penalty_mode_collapse=0.06,
        sell_bonus_eva_neg=0.85,
        maintain_bonus_eva_pos=0.5,
        restructure_admin_cost_abs=250.0,
        restructure_cost_quita_bps=40,
        sell_tx_cost_bps=70,
        eva_min_improvement=2_500.0,
        eva_strongly_neg=-20_000.0,
        esfuerzo_bajo=0.40,
        esfuerzo_alto=0.55,
        dscr_min=1.15,
    ),
}

# ================================================================
# 🧠 Recompensa RL (micro + macro)
# ================================================================
@dataclass
class RewardParams:
    """
    Parámetros de recompensa compartidos micro + macro.

    Se inicializa a partir del perfil de banco activo.
    """
    # Pesos base (LoanEnv + PortfolioEnv)
    w_eva: float = 0.25
    w_capital: float = 0.30
    w_stab: float = 0.15
    w_diversif: float = 0.05

    # P&L de ventas NPL (PortfolioEnv.step)
    w_pnl: float = 0.25
    pnl_penalty_scale: float = 1e5

    # P&L a nivel micro (LoanEnv.step)
    pnl_weight: float = 1.0

    # Pesos macro adicionales (PortfolioEnv)
    w_vol: float = 0.04
    w_concentration: float = 0.08
    w_cap_carry: float = 0.03

    # Penalizaciones / bonus prudenciales
    penalty_fire_sale: float = 0.50
    penalty_mode_collapse: float = 0.10
    sell_bonus_eva_neg: float = 0.20
    maintain_bonus_eva_pos: float = 0.25

    # Costes explícitos
    restructure_admin_cost_abs: float = 250.0
    restructure_cost_quita_bps: int = 40
    sell_tx_cost_bps: int = 100

    # Clipping global
    clip: Tuple[float, float] = (-1e6, 1e6)

    @classmethod
    def from_strategy(cls, strat: BankStrategy) -> "RewardParams":
        """Crea RewardParams a partir de un BankStrategy."""
        return cls(
            w_eva=strat.w_eva,
            w_capital=strat.w_capital,
            w_stab=strat.w_stab,
            w_diversif=strat.w_diversif,
            w_pnl=strat.w_pnl,
            pnl_penalty_scale=strat.pnl_penalty_scale,
            pnl_weight=strat.pnl_weight,
            w_vol=strat.w_vol,
            w_concentration=strat.w_concentration,
            w_cap_carry=strat.w_cap_carry,
            penalty_fire_sale=strat.penalty_fire_sale,
            penalty_mode_collapse=strat.penalty_mode_collapse,
            sell_bonus_eva_neg=strat.sell_bonus_eva_neg,
            maintain_bonus_eva_pos=strat.maintain_bonus_eva_pos,
            restructure_admin_cost_abs=strat.restructure_admin_cost_abs,
            restructure_cost_quita_bps=strat.restructure_cost_quita_bps,
            sell_tx_cost_bps=strat.sell_tx_cost_bps,
        )

# ================================================================
# 🤖 PPO
# ================================================================
@dataclass
class PPOParams:
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 500_000
    seed: int = GLOBAL_SEED
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {"net_arch": [128, 128]})
    tensorboard_log: str = str((LOGS_DIR / "tb").as_posix())

# ================================================================
# 📝 Logging
# ================================================================
@dataclass
class LoggingParams:
    level: int = logging.INFO
    fmt: str = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_file: Path = PIPELINE_LOG

    def configure(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        logging.basicConfig(
            level=self.level,
            format=self.fmt,
            datefmt=self.datefmt,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file, encoding="utf-8"),
            ],
        )
        logging.getLogger(__name__).info("Logging configurado.")

# ================================================================
# 🌐 CONFIG principal (micro + macro)
# ================================================================
@dataclass
class ProjectConfig:
    basel_map: BaselSTDMapping = field(default_factory=BaselSTDMapping)
    regulacion: Regulacion = field(default_factory=Regulacion)
    simulacion: SimulacionCartera = field(default_factory=SimulacionCartera)
    reestructura: ReestructuraParams = field(default_factory=ReestructuraParams)
    sensibilidad_reestructura: SensibilidadReestructura = field(default_factory=SensibilidadReestructura)
    precio_venta: PrecioVentaParams = field(default_factory=PrecioVentaParams)

    # ✅ Convención NPL centralizada
    npl: NPLConventions = field(default_factory=NPLConventions)

    # Estrategia global del banco (palanca principal)
    bank_profile: BankProfile = BankProfile.PRUDENTE
    bank_strategy: BankStrategy = field(default_factory=lambda: BANK_STRATEGIES[BankProfile.PRUDENTE])

    # Reward se deriva de la estrategia activa
    reward: RewardParams = field(default_factory=lambda: RewardParams.from_strategy(BANK_STRATEGIES[BankProfile.PRUDENTE]))

    ppo: PPOParams = field(default_factory=PPOParams)
    logging: LoggingParams = field(default_factory=LoggingParams)
    fast_debug: bool = False

    env: Dict[str, Any] = field(default_factory=lambda: {
        "n_features": 10,
        "n_actions": 3,
        "reward_weights": {"eva_delta": 1.5, "capital_lib": 0.7, "stability": 0.15, "prudence": 0.8},
        "max_steps": 50,
        "discount_gamma": 0.99,
        "normalize_obs": True,
        "normalize_rew": True,
        "portfolio_state_dim": 308,
        "portfolio_n_actions": 12,
        "normalize_obs_portfolio": True,
        "max_steps_portfolio": 30,
        "portfolio_eva_vol_window": 8,
    })

    def validate(self):
        tot = sum(self.simulacion.pct_segments.values())
        assert 0.99 <= tot <= 1.01, "pct_segments debe sumar 1.0"
        assert 0.02 <= self.regulacion.hurdle_rate <= 0.25, "hurdle fuera de rango"
        assert self.ppo.total_timesteps > 10_000, "timesteps insuficientes"

        # Checks NPL
        assert 0.10 <= self.npl.pd_default_floor <= self.npl.pd_default_cap <= 0.99, "rango PD NPL inválido"
        assert 0.10 <= self.npl.lgd_default_floor <= self.npl.lgd_default_cap <= 0.99, "rango LGD NPL inválido"
        assert self.npl.dpd_floor >= 0 and self.npl.dpd_cap > self.npl.dpd_floor, "rango DPD inválido"

        logging.getLogger(__name__).info("Validación de configuración OK.")

# ================================================================
# ⚡ Perfiles FAST / NORMAL (para pipelines)
# ================================================================
FAST_CFG = dict(ppo_steps=50_000, mc_sims=300, restre_grid_factor=0.33, horizon_months=6)
NORMAL_CFG = dict(ppo_steps=1_000_000, mc_sims=3000, restre_grid_factor=1.0, horizon_months=24)

def speed_profile(fast: bool) -> dict:
    return FAST_CFG if fast else NORMAL_CFG

# ================================================================
# 💾 Utilidades comunes (feature order)
# ================================================================
def save_feature_order(order: List[str], path=MODELS_DIR / "feature_order.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(order, f, ensure_ascii=False, indent=2)

def load_feature_order(path=MODELS_DIR / "feature_order.json") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ================================================================
# Bootstrapping
# ================================================================
def _bootstrap() -> ProjectConfig:
    cfg_obj = ProjectConfig()

    env_prof = os.environ.get("BANK_PROFILE", "").strip().upper()
    if env_prof:
        mapping = {"PRUDENTE": BankProfile.PRUDENTE, "BALANCEADO": BankProfile.BALANCEADO, "DESINVERSION": BankProfile.DESINVERSION}
        prof = mapping.get(env_prof)
        if prof is not None:
            cfg_obj.bank_profile = prof
        else:
            logging.getLogger(__name__).warning(
                "[config] BANK_PROFILE=%r no reconocido; se mantiene perfil por defecto %s",
                env_prof,
                cfg_obj.bank_profile.name,
            )

    # Resolver estrategia y reward de forma coherente
    strat = BANK_STRATEGIES[cfg_obj.bank_profile]
    cfg_obj.bank_strategy = strat
    cfg_obj.reward = RewardParams.from_strategy(strat)

    cfg_obj.logging.configure()
    set_all_seeds(GLOBAL_SEED)
    cfg_obj.validate()
    return cfg_obj

CONFIG = _bootstrap()

# ================================================================
# Debug
# ================================================================
if __name__ == "__main__":
    print(json.dumps({
        "hurdle": CONFIG.regulacion.hurdle_rate,
        "capital_ratio": CONFIG.regulacion.required_total_capital_ratio(),
        "portfolio_state_dim": CONFIG.env["portfolio_state_dim"],
        "w_pnl": CONFIG.reward.w_pnl,
        "pnl_weight": CONFIG.reward.pnl_weight,
        "bank_profile": CONFIG.bank_profile.value,
        "bank_strategy": CONFIG.bank_strategy.name,
        "npl_pd_range": [CONFIG.npl.pd_default_floor, CONFIG.npl.pd_default_cap],
        "npl_lgd_range": [CONFIG.npl.lgd_default_floor, CONFIG.npl.lgd_default_cap],
        "npl_dpd_range": [CONFIG.npl.dpd_floor, CONFIG.npl.dpd_cap],
        "default_is_state_not_pd": CONFIG.npl.default_is_state_not_pd,
    }, indent=2))
