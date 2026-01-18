# -*- coding: utf-8 -*-
# ============================================
# agent/train_subagents.py — Entrenamiento de subagentes (LoanEnv + PortfolioEnv)
# Versión Banco L1.5 · STD · PPO — v2.0 (VN split micro/macro, audit-ready)
# ============================================
"""
POC — OPTIMIZADOR DE CARTERAS EN DEFAULT (Método Estándar · Basilea III)

Este script entrena **por separado** los dos niveles de agente:

1️⃣ Subagente MICRO (LoanEnv)
    - Observación: 10 features
    - Objetivo: política óptima MANTENER / REESTRUCTURAR / VENDER a nivel préstamo
    - Guarda (NUEVO, sin pisar):
        • models/best_model_loan.zip
        • models/vecnormalize_loan.pkl
      Compatibilidad (LEGACY, opcional):
        • models/best_model.zip
        • models/vecnormalize_final.pkl

2️⃣ Subagente MACRO (PortfolioEnv)
    - Objetivo: acciones macro sobre cartera
    - Guarda (NUEVO, con normalización propia):
        • models/best_model_portfolio.zip
        • models/vecnormalize_portfolio.pkl

Notas críticas:
- Separa VecNormalize MICRO vs MACRO para evitar mismatch de shapes (10) vs (308, ...) en inferencia coordinada.
- PortfolioEnv actual (v3.5+) usa constructor con loan_dicts (lista de dicts) y seed/top_k/scenario.

CLI:
    python -m agent.train_subagents --agent loan --portfolio data/portfolio_synth.xlsx
    python -m agent.train_subagents --agent portfolio --portfolio data/portfolio_synth.xlsx
    python -m agent.train_subagents --agent both --portfolio data/portfolio_synth.xlsx
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import shutil
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 🔧 Rutas de proyecto
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 📣 Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "train_subagents.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_subagents")

# ---------------------------------------------------------------------
# 📦 Configuración y entornos
# ---------------------------------------------------------------------
import config as cfg
from env.loan_env import LoanEnv
from env.portfolio_env import PortfolioEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise SystemExit(
        "❌ Faltan dependencias RL (stable_baselines3, gymnasium, torch). "
        "Ejecuta primero install_requirements_smart.py."
    ) from e


# ---------------------------------------------------------------------
# 🧩 Utilidades
# ---------------------------------------------------------------------
def _load_portfolio(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Carga una cartera desde CSV/Excel si se proporciona ruta."""
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ No se encuentra la cartera: {path}")
    ext = os.path.splitext(path.lower())[1]
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    logger.info(f"📥 Cartera cargada ({len(df):,} préstamos) desde {path}")
    return df


def _loan_pool_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convierte un DataFrame de cartera en lista de dicts compatible con LoanEnv/PortfolioEnv."""
    return df.to_dict(orient="records")


def _device_auto(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _safe_copy(src: str, dst: str) -> None:
    """Copia robusta para compatibilidad legacy."""
    try:
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo copiar {src} -> {dst}: {e}")


# ---------------------------------------------------------------------
# 🧠 Entrenamiento subagente MICRO (LoanEnv)
# ---------------------------------------------------------------------
def train_loan_agent(
    portfolio_path: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    legacy_artifacts: bool = True,
) -> str:
    """
    Entrena el subagente micro (LoanEnv) y guarda:
        - models/best_model_loan.zip
        - models/vecnormalize_loan.pkl
      (opcional legacy):
        - models/best_model.zip
        - models/vecnormalize_final.pkl
    """
    logger.info("🏦 [MICRO] Entrenamiento subagente LoanEnv (nivel préstamo)…")

    ppo_cfg = cfg.CONFIG.ppo
    total_ts = int(total_timesteps or ppo_cfg.total_timesteps)
    seed_final = int(seed if seed is not None else ppo_cfg.seed)

    # 1) Cargar cartera si se pasa (para crear loan_pool), si no, LoanEnv sintético
    df: Optional[pd.DataFrame] = _load_portfolio(portfolio_path) if portfolio_path else None
    loan_pool: Optional[List[Dict[str, Any]]] = _loan_pool_from_df(df) if df is not None else None

    def _make_env():
        # LoanEnv v6.x: (seed, loan_pool) — si tu clase no admite seed, no rompe por **kwargs si lo ignoras
        return LoanEnv(seed=seed_final, loan_pool=loan_pool)

    vec_env = DummyVecEnv([_make_env])

    # VecNormalize MICRO (10-dim) — coherente con inferencia micro
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    device_final = _device_auto(device)
    logger.info(f"⚙️ Dispositivo para LoanEnv: {device_final} | seed={seed_final}")

    model = PPO(
        policy=ppo_cfg.policy,
        env=vec_env,
        learning_rate=ppo_cfg.learning_rate,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        tensorboard_log=os.path.join(ppo_cfg.tensorboard_log, "loan"),
        policy_kwargs=ppo_cfg.policy_kwargs,
        device=device_final,
        seed=seed_final,
        verbose=1,
    )

    logger.info(f"🚀 Iniciando entrenamiento LoanEnv por {total_ts:,} pasos…")
    model.learn(total_timesteps=total_ts, progress_bar=True)

    # Desactivar modo entrenamiento antes de guardar normalizador
    vec_env.training = False
    vec_env.norm_reward = False

    # Rutas NUEVAS
    model_path = os.path.join(MODELS_DIR, "best_model_loan.zip")
    vn_path = os.path.join(MODELS_DIR, "vecnormalize_loan.pkl")

    model.save(model_path)
    vec_env.save(vn_path)

    logger.info(f"✅ [MICRO] Modelo LoanEnv guardado en: {model_path}")
    logger.info(f"✅ [MICRO] VecNormalize MICRO guardado en: {vn_path}")

    # Compatibilidad LEGACY (para scripts antiguos). Importante:
    # - best_model.zip y vecnormalize_final.pkl se consideran “micro”.
    if legacy_artifacts:
        legacy_model = os.path.join(MODELS_DIR, "best_model.zip")
        legacy_vn = os.path.join(MODELS_DIR, "vecnormalize_final.pkl")
        _safe_copy(model_path, legacy_model)
        _safe_copy(vn_path, legacy_vn)
        logger.info(f"↩️ [LEGACY] Copias micro: {legacy_model} | {legacy_vn}")

    return model_path


# ---------------------------------------------------------------------
# 🧠 Entrenamiento subagente MACRO (PortfolioEnv)
# ---------------------------------------------------------------------
def train_portfolio_agent(
    portfolio_path: Optional[str],
    total_timesteps: Optional[int] = None,
    device: str = "auto",
    top_k: int = 5,
    scenario: str = "baseline",
    seed: Optional[int] = None,
) -> str:
    """
    Entrena el subagente macro (PortfolioEnv) y guarda:
        - models/best_model_portfolio.zip
        - models/vecnormalize_portfolio.pkl
    """
    logger.info("📈 [MACRO] Entrenamiento subagente PortfolioEnv (nivel cartera)…")

    if portfolio_path is None:
        raise ValueError("❌ Para entrenar el subagente de cartera es obligatorio pasar --portfolio.")

    df = _load_portfolio(portfolio_path)
    assert df is not None

    ppo_cfg = cfg.CONFIG.ppo
    total_ts = int(total_timesteps or ppo_cfg.total_timesteps)
    seed_final = int(seed if seed is not None else ppo_cfg.seed)

    loan_pool = _loan_pool_from_df(df)

    def _make_env():
        # PortfolioEnv v3.5+: usa loan_dicts
        return PortfolioEnv(
            loan_dicts=loan_pool,
            seed=seed_final,
            top_k=int(top_k),
            scenario=str(scenario),
            ppo_micro=None,
            micro_vecnormalize_path=None,
        )

    vec_env = DummyVecEnv([_make_env])

    # VecNormalize MACRO (dim alto, p.ej. 308). Recomendado:
    # - norm_obs=True
    # - norm_reward=False (macro reward suele ser estable / no conviene normalizar si ya está escalada)
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    device_final = _device_auto(device)
    logger.info(f"⚙️ Dispositivo para PortfolioEnv: {device_final} | seed={seed_final} | top_k={top_k} | scenario={scenario}")

    model = PPO(
        policy=ppo_cfg.policy,
        env=vec_env,
        learning_rate=ppo_cfg.learning_rate,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        tensorboard_log=os.path.join(ppo_cfg.tensorboard_log, "portfolio"),
        policy_kwargs=ppo_cfg.policy_kwargs,
        device=device_final,
        seed=seed_final,
        verbose=1,
    )

    logger.info(f"🚀 Iniciando entrenamiento PortfolioEnv por {total_ts:,} pasos…")
    model.learn(total_timesteps=total_ts, progress_bar=True)

    # Desactivar modo entrenamiento antes de guardar normalizador
    vec_env.training = False
    vec_env.norm_reward = False

    model_path = os.path.join(MODELS_DIR, "best_model_portfolio.zip")
    vn_path = os.path.join(MODELS_DIR, "vecnormalize_portfolio.pkl")

    model.save(model_path)
    vec_env.save(vn_path)

    logger.info(f"✅ [MACRO] Modelo PortfolioEnv guardado en: {model_path}")
    logger.info(f"✅ [MACRO] VecNormalize MACRO guardado en: {vn_path}")

    return model_path


# ---------------------------------------------------------------------
# 🧭 CLI principal
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de subagentes RL (LoanEnv + PortfolioEnv) · Banco L1.5"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["loan", "portfolio", "both"],
        default="loan",
        help="Qué subagente entrenar: loan, portfolio o both.",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Ruta a cartera (Excel/CSV). Recomendada para ambos agentes.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total de timesteps para PPO (si se omite usa config.CONFIG.ppo.total_timesteps).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo para SB3 (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Valor de top_k a usar en PortfolioEnv (coherente con portfolio_env.py).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["baseline", "adverse", "severe"],
        default="baseline",
        help="Escenario macro de PortfolioEnv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opcional (si se omite usa cfg.CONFIG.ppo.seed).",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Si se activa, NO genera copias legacy best_model.zip / vecnormalize_final.pkl (micro).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("===================================================")
    logger.info("🧠 Entrenamiento de subagentes RL (Banco L1.5)")
    logger.info(f"   Agent mode   : {args.agent}")
    logger.info(f"   Portfolio    : {args.portfolio if args.portfolio else 'synthetic / default'}")
    logger.info(f"   Total steps  : {args.total_steps or cfg.CONFIG.ppo.total_timesteps:,}")
    logger.info(f"   Device       : {args.device}")
    logger.info(f"   Seed         : {args.seed if args.seed is not None else cfg.CONFIG.ppo.seed}")
    logger.info("===================================================")

    if args.agent in ("loan", "both"):
        train_loan_agent(
            portfolio_path=args.portfolio,
            total_timesteps=args.total_steps,
            device=args.device,
            seed=args.seed,
            legacy_artifacts=(not args.no_legacy),
        )

    if args.agent in ("portfolio", "both"):
        train_portfolio_agent(
            portfolio_path=args.portfolio,
            total_timesteps=args.total_steps,
            device=args.device,
            top_k=args.top_k,
            scenario=args.scenario,
            seed=args.seed,
        )

    logger.info("🏁 Entrenamiento de subagentes completado correctamente.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Entrenamiento interrumpido manualmente.")
    except Exception as e:
        logger.error(f"❌ Error crítico en train_subagents: {e}")
        raise
