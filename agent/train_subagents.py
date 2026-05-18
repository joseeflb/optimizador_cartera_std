# -*- coding: utf-8 -*-
# ============================================================
# agent/train_subagents.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Entrena por separado el subagente MICRO (LoanEnv, 10 obs / 3 acciones) y el subagente MACRO (PortfolioEnv, 308 obs / 12 acciones) con eval callbacks y checkpointing audit-ready.
# ============================================================

"""
POC — OPTIMIZADOR DE CARTERAS EN DEFAULT (Metodo Estandar · Basilea III)

Este script entrena **por separado** los dos niveles de agente:

[1] Subagente MICRO (LoanEnv)
    - Observacion: 10 features
    - Acciones: 3 (MANTENER / REESTRUCTURAR / VENDER)
    - Objetivo: politica optima a nivel prestamo individual
    - Guarda:
        * models/best_model_loan.zip (mejor modelo por eval callback)
        * models/vecnormalize_loan.pkl
      Legacy:
        * models/best_model.zip / vecnormalize_final.pkl

[2] Subagente MACRO (PortfolioEnv)
    - Observacion: 308 features (pre-escaladas a MM€ + ratios + concentracion)
    - Acciones: 12 (sell/restructure por EVA/RORWA/PTI/mix/regla/hold)
    - Objetivo: acciones macro sobre cartera completa
    - Guarda:
        * models/best_model_portfolio.zip
        * models/vecnormalize_portfolio.pkl

Mejoras v3.0:
- Eval callback con early stopping basado en reward medio
- Checkpoints periodicos durante entrenamiento
- Monitorizacion de entropia, KL y explained variance
- Entorno de evaluacion separado con VecNormalize sincronizado
- Soporte para timesteps independientes por agente (--total-steps-loan / --total-steps-portfolio)

CLI:
    python -m agent.train_subagents --agent loan --portfolio data/portfolio_synth.xlsx
    python -m agent.train_subagents --agent portfolio --portfolio data/portfolio_synth.xlsx
    python -m agent.train_subagents --agent both --portfolio data/portfolio_synth.xlsx
    python -m agent.train_subagents --agent both --portfolio data/portfolio_synth.xlsx --total-steps-loan 500000 --total-steps-portfolio 1000000
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import shutil
import time
import json
from typing import Any, Dict, Optional, List
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Rutas de proyecto
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
# Logging
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
# Configuracion y entornos
# ---------------------------------------------------------------------
import config as cfg
from env.loan_env import LoanEnv
from env.portfolio_env import PortfolioEnv


# ---------------------------------------------------------------------
# [POSTURE] Helpers de postura
# ---------------------------------------------------------------------
_VALID_POSTURES = ("prudente", "balanceado", "desinversion")


def _normalize_posture(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    s = str(p).strip().lower()
    s = (
        s.replace("\u00e1", "a").replace("\u00e9", "e").replace("\u00ed", "i")
         .replace("\u00f3", "o").replace("\u00fa", "u").replace("\u00f1", "n")
    )
    alias = {
        "prudencial": "prudente", "prudent": "prudente",
        "balanced": "balanceado", "neutral": "balanceado",
        "divestment": "desinversion", "desinvertir": "desinversion",
    }
    s = alias.get(s, s)
    if s not in _VALID_POSTURES:
        return None
    return s


def _suffix_for_posture(posture: Optional[str]) -> str:
    p = _normalize_posture(posture)
    return f"_{p}" if p else ""

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
except ImportError as e:
    raise SystemExit(
        "[ERR] Faltan dependencias RL (stable_baselines3, gymnasium, torch). "
        "Ejecuta primero install_requirements_smart.py."
    ) from e


# =====================================================================
#  Eval Callback con metricas de negocio y early stopping
# =====================================================================
class SubagentEvalCallback(BaseCallback):
    """
    Callback de evaluacion periodica con:
    - Evaluacion en entorno separado (VecNormalize sincronizado)
    - Early stopping por paciencia en reward medio
    - Logging de metricas clave (entropia, diversidad de acciones, reward)
    - Guardado del mejor modelo
    """

    def __init__(
        self,
        eval_env: VecNormalize,
        agent_name: str = "agent",
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        patience: int = 10,
        best_model_path: str = "",
        best_vn_path: str = "",
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.agent_name = agent_name
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self.patience = max(1, patience)
        self.best_model_path = best_model_path
        self.best_vn_path = best_vn_path
        self.deterministic = deterministic

        self.best_mean_reward = -np.inf
        self.no_improve_count = 0
        self.eval_history: List[Dict[str, float]] = []
        self._last_eval_step = 0

    def _sync_normalizer(self):
        """Sincroniza estadisticas del VecNormalize de entrenamiento al de eval."""
        try:
            train_vn = self.model.get_vec_normalize_env()
            if train_vn is not None and isinstance(train_vn, VecNormalize):
                self.eval_env.obs_rms = train_vn.obs_rms
                self.eval_env.ret_rms = train_vn.ret_rms
                self.eval_env.training = False
                self.eval_env.norm_reward = False
        except Exception as e:
            logger.warning(f"[{self.agent_name}] Error sync VN: {e}")

    def _evaluate(self) -> Dict[str, float]:
        """Ejecuta n_eval_episodes y retorna metricas agregadas."""
        self._sync_normalizer()

        rewards_ep = []
        lengths_ep = []
        action_counts = defaultdict(int)
        total_actions = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            done = False
            ep_reward = 0.0
            ep_len = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                out = self.eval_env.step(action)

                if len(out) == 5:
                    obs, reward, terminated, truncated, info = out
                    d = np.logical_or(terminated, truncated)
                else:
                    obs, reward, d, info = out

                d0 = d[0] if isinstance(d, (list, np.ndarray)) else d
                r0 = reward[0] if isinstance(reward, (list, np.ndarray)) else reward

                ep_reward += float(r0)
                ep_len += 1

                a0 = action[0] if isinstance(action, (list, np.ndarray)) else action
                action_counts[int(a0)] += 1
                total_actions += 1

                done = bool(d0)

            rewards_ep.append(ep_reward)
            lengths_ep.append(ep_len)

        mean_reward = float(np.mean(rewards_ep)) if rewards_ep else 0.0
        std_reward = float(np.std(rewards_ep)) if rewards_ep else 0.0
        mean_length = float(np.mean(lengths_ep)) if lengths_ep else 0.0

        # Diversidad de acciones (entropia normalizada)
        n_actions_seen = max(len(action_counts), 1)
        total = max(total_actions, 1)
        max_action_id = max(action_counts.keys()) + 1 if action_counts else 1
        probs = np.array([action_counts.get(a, 0) / total for a in range(max_action_id)])
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(np.log(max_action_id + 1e-10))
        entropy_ratio = entropy / max(max_entropy, 1e-10)

        # Accion dominante
        dominant_action = max(action_counts, key=action_counts.get) if action_counts else 0
        dominant_pct = action_counts.get(dominant_action, 0) / max(total_actions, 1) * 100

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "entropy_ratio": entropy_ratio,
            "dominant_action": dominant_action,
            "dominant_pct": dominant_pct,
            "n_unique_actions": n_actions_seen,
        }

    def _save_best(self):
        """Guarda mejor modelo y VecNormalize."""
        if self.best_model_path:
            self.model.save(self.best_model_path)
        if self.best_vn_path:
            try:
                vn = self.model.get_vec_normalize_env()
                if vn is not None:
                    vn.save(self.best_vn_path)
            except Exception as e:
                logger.warning(f"[{self.agent_name}] No se pudo guardar VN: {e}")

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        if (t - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = t

        metrics = self._evaluate()
        metrics["timestep"] = t
        self.eval_history.append(metrics)

        mr = metrics["mean_reward"]
        sr = metrics["std_reward"]
        er = metrics["entropy_ratio"]
        da = metrics["dominant_action"]
        dp = metrics["dominant_pct"]
        nu = metrics["n_unique_actions"]

        logger.info(
            f"[{self.agent_name}] EVAL @ {t:,} | reward={mr:.4f} +/- {sr:.4f} | "
            f"entropy_ratio={er:.3f} | dominant=a{da}({dp:.0f}%) | "
            f"unique_actions={nu} | best={self.best_mean_reward:.4f}"
        )

        # Alerta de policy collapse
        if dp > 85.0:
            logger.warning(
                f"[{self.agent_name}] ALERTA: posible policy collapse - "
                f"accion {da} domina al {dp:.0f}% (entropy_ratio={er:.3f})"
            )

        if mr > self.best_mean_reward:
            self.best_mean_reward = mr
            self.no_improve_count = 0
            self._save_best()
            logger.info(f"[{self.agent_name}] NUEVO MEJOR modelo guardado (reward={mr:.4f})")
        else:
            self.no_improve_count += 1

        if self.no_improve_count >= self.patience:
            logger.info(
                f"[{self.agent_name}] EARLY STOPPING: sin mejora en {self.patience} "
                f"evaluaciones ({self.patience * self.eval_freq:,} pasos)"
            )
            return False

        return True

    def get_summary(self) -> Dict[str, Any]:
        """Resumen final del entrenamiento para logging."""
        return {
            "agent": self.agent_name,
            "best_mean_reward": self.best_mean_reward,
            "total_evals": len(self.eval_history),
            "early_stopped": self.no_improve_count >= self.patience,
            "eval_history": self.eval_history,
        }


# =====================================================================
#  Utilidades
# =====================================================================
def _load_portfolio(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Carga una cartera desde CSV/Excel si se proporciona ruta."""
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERR] No se encuentra la cartera: {path}")
    ext = os.path.splitext(path.lower())[1]
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    logger.info(f"Cartera cargada ({len(df):,} prestamos) desde {path}")
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
        logger.warning(f"[WARN] No se pudo copiar {src} -> {dst}: {e}")


def _save_training_metadata(
    agent_name: str,
    model_path: str,
    total_timesteps: int,
    ppo_cfg: Any,
    eval_summary: Optional[Dict] = None,
    elapsed_sec: float = 0.0,
) -> None:
    """Guarda metadatos del entrenamiento para auditoria."""
    meta = {
        "agent": agent_name,
        "model_path": model_path,
        "total_timesteps": total_timesteps,
        "elapsed_seconds": round(elapsed_sec, 1),
        "ppo_config": {
            "learning_rate": ppo_cfg.learning_rate,
            "n_steps": ppo_cfg.n_steps,
            "batch_size": ppo_cfg.batch_size,
            "n_epochs": ppo_cfg.n_epochs,
            "gamma": ppo_cfg.gamma,
            "ent_coef": ppo_cfg.ent_coef,
            "clip_range": ppo_cfg.clip_range,
            "net_arch": ppo_cfg.policy_kwargs.get("net_arch", []),
        },
    }
    if eval_summary:
        meta["eval_summary"] = {
            "best_mean_reward": eval_summary.get("best_mean_reward", None),
            "total_evals": eval_summary.get("total_evals", 0),
            "early_stopped": eval_summary.get("early_stopped", False),
        }

    meta_path = model_path.replace(".zip", "_training_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"[{agent_name}] Metadatos guardados: {meta_path}")
    except Exception as e:
        logger.warning(f"[{agent_name}] Error guardando metadatos: {e}")


# =====================================================================
#  Entrenamiento subagente MICRO (LoanEnv)
# =====================================================================
def train_loan_agent(
    portfolio_path: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    legacy_artifacts: bool = True,
    eval_freq: int = 10_000,
    patience: int = 15,
    posture: Optional[str] = None,
) -> str:
    """
    Entrena el subagente micro (LoanEnv) con eval callback y early stopping.
    """
    posture = _normalize_posture(posture)
    suffix = _suffix_for_posture(posture)
    logger.info("=" * 60)
    logger.info(f"[MICRO] Entrenamiento subagente LoanEnv (nivel prestamo) | posture={posture or 'default'}")
    logger.info("=" * 60)

    ppo_cfg = cfg.CONFIG.ppo
    total_ts = int(total_timesteps or ppo_cfg.total_timesteps)
    seed_final = int(seed if seed is not None else ppo_cfg.seed)

    # 1) Cargar cartera
    df: Optional[pd.DataFrame] = _load_portfolio(portfolio_path) if portfolio_path else None
    loan_pool: Optional[List[Dict[str, Any]]] = _loan_pool_from_df(df) if df is not None else None

    # 2) Entorno de entrenamiento
    def _make_train_env():
        return LoanEnv(seed=seed_final, loan_pool=loan_pool, bank_profile=posture)

    vec_env = DummyVecEnv([_make_train_env])
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # 3) Entorno de evaluacion (separado, misma config)
    def _make_eval_env():
        return LoanEnv(seed=seed_final + 1000, loan_pool=loan_pool, bank_profile=posture)

    eval_vec = DummyVecEnv([_make_eval_env])
    eval_vec = VecNormalize(
        eval_vec,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    device_final = _device_auto(device)
    logger.info(f"[CONFIG] device={device_final} | seed={seed_final} | timesteps={total_ts:,}")
    logger.info(f"[CONFIG] lr={ppo_cfg.learning_rate} | ent_coef={ppo_cfg.ent_coef} | "
                f"net_arch={ppo_cfg.policy_kwargs.get('net_arch', '?')} | "
                f"n_steps={ppo_cfg.n_steps} | batch={ppo_cfg.batch_size}")

    # 4) Modelo PPO
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

    # 5) Callbacks
    model_path = os.path.join(MODELS_DIR, f"best_model_loan{suffix}.zip")
    vn_path = os.path.join(MODELS_DIR, f"vecnormalize_loan{suffix}.pkl")

    eval_cb = SubagentEvalCallback(
        eval_env=eval_vec,
        agent_name="MICRO/loan",
        eval_freq=eval_freq,
        n_eval_episodes=10,
        patience=patience,
        best_model_path=model_path,
        best_vn_path=vn_path,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_ts // 10, 10_000),
        save_path=os.path.join(MODELS_DIR, f"checkpoints_loan{suffix}"),
        name_prefix=f"loan{suffix}",
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_cb, checkpoint_cb])

    # 6) Entrenar
    logger.info(f"Iniciando entrenamiento LoanEnv por {total_ts:,} pasos...")
    t0 = time.time()
    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    # 7) Guardar modelo final (si no hay mejor por eval, guardar ultimo)
    vec_env.training = False
    vec_env.norm_reward = False

    if eval_cb.best_mean_reward == -np.inf:
        # Eval nunca mejoro: guardar ultimo modelo
        model.save(model_path)
        vec_env.save(vn_path)
        logger.warning("[MICRO] Eval callback nunca encontro mejora; guardando modelo final.")
    else:
        # Asegurar que VN final este guardado
        vec_env.save(vn_path)

    # 8) Metadatos
    eval_summary = eval_cb.get_summary()
    if posture:
        eval_summary = dict(eval_summary or {})
        eval_summary["posture"] = posture
    _save_training_metadata(f"loan{suffix}", model_path, total_ts, ppo_cfg, eval_summary, elapsed)

    logger.info(f"[MICRO] Entrenamiento completado en {elapsed:.0f}s")
    logger.info(f"[MICRO] Mejor reward: {eval_cb.best_mean_reward:.4f}")
    logger.info(f"[MICRO] Modelo: {model_path}")

    # 9) Legacy
    # Solo replicamos a paths legacy (sin sufijo) cuando NO hay postura,
    # para no pisar artefactos de otras posturas.
    if legacy_artifacts and not posture:
        legacy_model = os.path.join(MODELS_DIR, "best_model.zip")
        legacy_vn = os.path.join(MODELS_DIR, "vecnormalize_final.pkl")
        _safe_copy(model_path, legacy_model)
        _safe_copy(vn_path, legacy_vn)
        logger.info(f"[LEGACY] Copias micro: {legacy_model} | {legacy_vn}")

    return model_path


# =====================================================================
#  Entrenamiento subagente MACRO (PortfolioEnv)
# =====================================================================
def train_portfolio_agent(
    portfolio_path: Optional[str],
    total_timesteps: Optional[int] = None,
    device: str = "auto",
    top_k: int = 5,
    scenario: str = "baseline",
    seed: Optional[int] = None,
    eval_freq: int = 5_000,
    patience: int = 20,
    posture: Optional[str] = None,
) -> str:
    """
    Entrena el subagente macro (PortfolioEnv) con eval callback y early stopping.

    Portfolio necesita mas paciencia y evaluaciones menos frecuentes ya que:
    - 12 acciones (vs 3 del loan) -> mas exploracion necesaria
    - Episodios de 30 pasos -> cada rollout cubre menos episodios
    """
    posture = _normalize_posture(posture)
    suffix = _suffix_for_posture(posture)
    logger.info("=" * 60)
    logger.info(f"[MACRO] Entrenamiento subagente PortfolioEnv (nivel cartera) | posture={posture or 'default'}")
    logger.info("=" * 60)

    if portfolio_path is None:
        raise ValueError("[ERR] Para entrenar el subagente de cartera es obligatorio pasar --portfolio.")

    df = _load_portfolio(portfolio_path)
    assert df is not None

    ppo_cfg = cfg.CONFIG.ppo
    total_ts = int(total_timesteps or ppo_cfg.total_timesteps)
    seed_final = int(seed if seed is not None else ppo_cfg.seed)
    loan_pool = _loan_pool_from_df(df)

    # 0) Suprimir logging verbose de todos los modulos del env durante entrenamiento
    for _logger_name in ("portfolio_env", "price_simulator", "guardrails", "restructure_optimizer"):
        _lg = logging.getLogger(_logger_name)
        _lg.setLevel(logging.WARNING)
    logger.info("[MACRO] Loggers env/optimizer -> WARNING (reducir I/O durante training)")

    # Portfolio necesita n_steps mas pequeno para rollouts rapidos con 500 loans
    portfolio_n_steps = min(ppo_cfg.n_steps, 2048)
    logger.info(f"[MACRO] n_steps ajustado: {portfolio_n_steps} (base={ppo_cfg.n_steps})")

    # 1) Entorno de entrenamiento
    def _make_train_env():
        return PortfolioEnv(
            loan_dicts=loan_pool,
            seed=seed_final,
            top_k=int(top_k),
            scenario=str(scenario),
            ppo_micro=None,
            micro_vecnormalize_path=None,
            bank_profile=posture,
        )

    vec_env = DummyVecEnv([_make_train_env])
    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,   # Normalizamos reward tambien para estabilidad
        clip_obs=10.0,
    )

    # 2) Entorno de evaluacion (separado)
    def _make_eval_env():
        return PortfolioEnv(
            loan_dicts=loan_pool,
            seed=seed_final + 2000,
            top_k=int(top_k),
            scenario=str(scenario),
            ppo_micro=None,
            micro_vecnormalize_path=None,
            bank_profile=posture,
        )

    eval_vec = DummyVecEnv([_make_eval_env])
    eval_vec = VecNormalize(
        eval_vec,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    device_final = _device_auto(device)
    logger.info(f"[CONFIG] device={device_final} | seed={seed_final} | timesteps={total_ts:,}")
    logger.info(f"[CONFIG] top_k={top_k} | scenario={scenario}")
    logger.info(f"[CONFIG] lr={ppo_cfg.learning_rate} | ent_coef={ppo_cfg.ent_coef} | "
                f"net_arch={ppo_cfg.policy_kwargs.get('net_arch', '?')} | "
                f"n_steps={ppo_cfg.n_steps} | batch={ppo_cfg.batch_size}")

    # 3) Modelo PPO — Portfolio usa learning rate mas bajo para estabilidad con 12 acciones
    portfolio_lr = ppo_cfg.learning_rate * 0.5  # Mitad del LR para portfolio
    portfolio_ent = max(ppo_cfg.ent_coef * 5.0, 0.10)  # Mas entropia para 12 acciones (prevenir colapso)
    # [OK] Boost extra de entropia para PRUDENTE/BALANCEADO (familias macro mas dificiles de descubrir)
    if posture in ("prudente", "balanceado"):
        portfolio_ent = max(portfolio_ent, 0.20)
    portfolio_epochs = min(ppo_cfg.n_epochs, 3)  # Fewer epochs to avoid overfitting early data

    model = PPO(
        policy=ppo_cfg.policy,
        env=vec_env,
        learning_rate=portfolio_lr,
        n_steps=portfolio_n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=portfolio_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=ppo_cfg.clip_range,
        ent_coef=portfolio_ent,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        tensorboard_log=os.path.join(ppo_cfg.tensorboard_log, "portfolio"),
        policy_kwargs=ppo_cfg.policy_kwargs,
        device=device_final,
        seed=seed_final,
        verbose=1,
    )

    logger.info(f"[MACRO] LR ajustado: {portfolio_lr} (base * 0.5)")
    logger.info(f"[MACRO] ent_coef ajustado: {portfolio_ent} (base * 5.0, min 0.10)")
    logger.info(f"[MACRO] n_epochs ajustado: {portfolio_epochs} (max {ppo_cfg.n_epochs}, 3)")

    # 4) Callbacks
    model_path = os.path.join(MODELS_DIR, f"best_model_portfolio{suffix}.zip")
    vn_path = os.path.join(MODELS_DIR, f"vecnormalize_portfolio{suffix}.pkl")

    eval_cb = SubagentEvalCallback(
        eval_env=eval_vec,
        agent_name="MACRO/portfolio",
        eval_freq=eval_freq,
        n_eval_episodes=5,   # Portfolio episodes son mas largos (30 steps)
        patience=patience,
        best_model_path=model_path,
        best_vn_path=vn_path,
        deterministic=False,  # Stochastic eval for diverse portfolio actions
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_ts // 10, 5_000),
        save_path=os.path.join(MODELS_DIR, f"checkpoints_portfolio{suffix}"),
        name_prefix=f"portfolio{suffix}",
        save_vecnormalize=True,
    )

    callbacks = CallbackList([eval_cb, checkpoint_cb])

    # 5) Entrenar
    logger.info(f"Iniciando entrenamiento PortfolioEnv por {total_ts:,} pasos...")
    t0 = time.time()
    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    # 6) Guardar
    vec_env.training = False
    vec_env.norm_reward = False

    if eval_cb.best_mean_reward == -np.inf:
        model.save(model_path)
        vec_env.save(vn_path)
        logger.warning("[MACRO] Eval callback nunca encontro mejora; guardando modelo final.")
    else:
        vec_env.save(vn_path)

    # 7) Metadatos
    eval_summary = eval_cb.get_summary()
    if posture:
        eval_summary = dict(eval_summary or {})
        eval_summary["posture"] = posture
    _save_training_metadata(f"portfolio{suffix}", model_path, total_ts, ppo_cfg, eval_summary, elapsed)

    logger.info(f"[MACRO] Entrenamiento completado en {elapsed:.0f}s")
    logger.info(f"[MACRO] Mejor reward: {eval_cb.best_mean_reward:.4f}")
    logger.info(f"[MACRO] Modelo: {model_path}")

    return model_path


# =====================================================================
#  CLI principal
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de subagentes RL (LoanEnv + PortfolioEnv) v3.0"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["loan", "portfolio", "both"],
        default="loan",
        help="Que subagente entrenar: loan, portfolio o both.",
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
        help="Total de timesteps para ambos agentes (si se omite usa config).",
    )
    parser.add_argument(
        "--total-steps-loan",
        type=int,
        default=None,
        help="Timesteps especificos para loan agent (override --total-steps).",
    )
    parser.add_argument(
        "--total-steps-portfolio",
        type=int,
        default=None,
        help="Timesteps especificos para portfolio agent (override --total-steps).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo para SB3.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Valor de top_k para PortfolioEnv.",
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
        help="Seed opcional.",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="No generar copias legacy.",
    )
    parser.add_argument(
        "--eval-freq-loan",
        type=int,
        default=10_000,
        help="Frecuencia de evaluacion para loan (en timesteps).",
    )
    parser.add_argument(
        "--eval-freq-portfolio",
        type=int,
        default=5_000,
        help="Frecuencia de evaluacion para portfolio.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Paciencia para early stopping (numero de evals sin mejora).",
    )
    parser.add_argument(
        "--posture",
        type=str,
        choices=list(_VALID_POSTURES),
        default=None,
        help="Postura del banco (PRUDENTE/BALANCEADO/DESINVERSION). Si se omite, usa la postura por defecto de config y NO sufija los artefactos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determinar timesteps por agente
    ts_loan = args.total_steps_loan or args.total_steps or cfg.CONFIG.ppo.total_timesteps
    ts_portfolio = args.total_steps_portfolio or args.total_steps or cfg.CONFIG.ppo.total_timesteps

    logger.info("=" * 60)
    logger.info("  Entrenamiento de subagentes RL (Banco L1.5) v3.0")
    logger.info("=" * 60)
    logger.info(f"  Agent mode      : {args.agent}")
    logger.info(f"  Portfolio       : {args.portfolio or 'synthetic'}")
    logger.info(f"  Timesteps loan  : {ts_loan:,}")
    logger.info(f"  Timesteps portf : {ts_portfolio:,}")
    logger.info(f"  Device          : {args.device}")
    logger.info(f"  Seed            : {args.seed or cfg.CONFIG.ppo.seed}")
    logger.info(f"  Patience        : {args.patience}")
    logger.info(f"  Posture         : {args.posture or '(default config)'}")
    logger.info("=" * 60)

    t_global = time.time()

    if args.agent in ("loan", "both"):
        train_loan_agent(
            portfolio_path=args.portfolio,
            total_timesteps=ts_loan,
            device=args.device,
            seed=args.seed,
            legacy_artifacts=(not args.no_legacy),
            eval_freq=args.eval_freq_loan,
            patience=args.patience,
            posture=args.posture,
        )

    if args.agent in ("portfolio", "both"):
        train_portfolio_agent(
            portfolio_path=args.portfolio,
            total_timesteps=ts_portfolio,
            device=args.device,
            top_k=args.top_k,
            scenario=args.scenario,
            seed=args.seed,
            eval_freq=args.eval_freq_portfolio,
            patience=args.patience,
            posture=args.posture,
        )

    elapsed_total = time.time() - t_global
    logger.info(f"Entrenamiento completado en {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("[WARN] Entrenamiento interrumpido manualmente.")
    except Exception as e:
        logger.error(f"[ERR] Error critico en train_subagents: {e}")
        raise
