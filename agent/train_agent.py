# -*- coding: utf-8 -*-
# ============================================
# agent/train_agent.py — Entrenamiento PPO (LoanEnv + PortfolioEnv)
# (AUDIT-READY · STD · NPL) — v1.2 hardened (VN split micro/macro)
# ============================================
"""
Entrenamiento del agente PPO sobre LoanEnv o PortfolioEnv.

Incluye:
- Early stopping basado en EVA_final + Capital_liberado_total (A+B) + penalties macro (audit-ready)
- TensorBoard + logs
- Checkpoints periódicos + VecNormalize por checkpoint
- VecNormalize (obs + reward) con sincronización en eval
- FAST-DEBUG coherente con config.py
- Soporte entornos: loan / portfolio
- (Opcional) PortfolioEnv micro→macro re-ranking con PPO micro (LoanEnv) + VecNormalize micro por ruta

FIX v1.2 (crítico):
- Artefactos separados por env_type para evitar pisar VecNormalize:
    * best_model_loan.zip / best_model_portfolio.zip
    * last_model_loan.zip / last_model_portfolio.zip
    * vecnormalize_loan.pkl / vecnormalize_portfolio.pkl
    * checkpoints/loan/* y checkpoints/portfolio/*
    * training_metadata_loan.json y training_metadata_portfolio.json
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# -----------------------------------------------------------
# 🔧 Rutas
# -----------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config as cfg
from env.loan_env import LoanEnv
from env.portfolio_env import PortfolioEnv

LOG_DIR = os.path.join(ROOT_DIR, "logs")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
for d in (LOG_DIR, REPORTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------
# 📦 Imports RL
# -----------------------------------------------------------
try:
    import torch
    import gymnasium as gym  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    raise SystemExit("❌ Faltan dependencias RL. Ejecuta install_requirements_smart.py.")

# -----------------------------------------------------------
# 🧠 Logging
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train_agent.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_agent")

# -----------------------------------------------------------
# 🥪 Semillas globales
# -----------------------------------------------------------
def set_global_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    cfg.set_all_seeds(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------
# ⚙️ Configuración de entrenamiento
# -----------------------------------------------------------
@dataclass
class TrainConfig:
    seed: int = cfg.CONFIG.ppo.seed
    total_timesteps: int = cfg.CONFIG.ppo.total_timesteps
    n_envs: int = 4
    n_steps: int = cfg.CONFIG.ppo.n_steps
    batch_size: int = cfg.CONFIG.ppo.batch_size
    gamma: float = cfg.CONFIG.ppo.gamma
    gae_lambda: float = cfg.CONFIG.ppo.gae_lambda
    ent_coef: float = cfg.CONFIG.ppo.ent_coef
    vf_coef: float = cfg.CONFIG.ppo.vf_coef
    clip_range: float = cfg.CONFIG.ppo.clip_range
    learning_rate: float = cfg.CONFIG.ppo.learning_rate
    max_grad_norm: float = cfg.CONFIG.ppo.max_grad_norm
    policy_hidden: tuple[int, int] = tuple(cfg.CONFIG.ppo.policy_kwargs.get("net_arch", [128, 128]))  # type: ignore
    activation_fn: str = "tanh"
    tensorboard: bool = True

    # Evaluación financiera base (A+B) — escalada a MM€ para estabilidad
    eval_episodes: int = 8
    eval_every: int = 10_000
    patience_evals: int = 3
    cap_weight: float = 0.5
    scale_mm: float = 1e6

    # Penalizaciones macro (solo PortfolioEnv; audit-friendly)
    blocked_weight: float = 0.10   # penaliza intentos de venta bloqueados
    conc_weight: float = 0.05      # penaliza concentración (HHI_seg + HHI_rat)
    vol_weight: float = 0.05       # penaliza volatilidad EVA (en MM€)

    # Normalización
    norm_obs: bool = True
    norm_reward: Optional[bool] = None  # None => auto: loan=True, portfolio=False

    # Otros
    device: str = "auto"
    fast_debug: bool = False
    portfolio_path: str = ""
    save_scaler: bool = True

    # Entorno
    env_type: str = "loan"     # "loan" | "portfolio"
    top_k: int = 5
    scenario: str = "baseline"

    # (Opcional) PPO micro para PortfolioEnv (re-ranking)
    micro_model_path: str = ""
    micro_vecnorm_path: str = ""


# -----------------------------------------------------------
# 📈 Métricas financieras (por episodio)
# -----------------------------------------------------------
MICRO_OBS_ORDER = [
    "EAD", "PD", "LGD", "RW", "EVA", "RONA", "RORWA", "rating_num", "segmento_id", "DPD/30"
]


def extract_episode_business_metrics(info_list: List[Any]) -> Dict[str, float]:
    """
    Métricas por episodio (robusto para LoanEnv/PortfolioEnv):
      - eva_final: EVA en el último step
      - cap_total: suma de capital_liberado en el episodio
      - n_sell_blocked: suma/último (si existe)
      - hhi_seg, hhi_rat, eva_volatility: último (si existe)
    """
    eva_final = 0.0
    cap_total = 0.0

    n_sell_blocked = 0.0
    hhi_seg = 0.0
    hhi_rat = 0.0
    eva_vol = 0.0

    last_info: Optional[Dict[str, Any]] = None

    for info in info_list:
        if not isinstance(info, dict):
            continue

        last_info = info
        m = info.get("metrics") or info.get("portfolio_metrics") or {}
        if "capital_liberado" in m:
            try:
                cap_total += float(m["capital_liberado"])
            except Exception:
                pass

        if "n_sell_blocked" in (info.get("action_summary") or {}):
            try:
                n_sell_blocked += float(info["action_summary"]["n_sell_blocked"])
            except Exception:
                pass

    if isinstance(last_info, dict):
        m_last = last_info.get("metrics") or last_info.get("portfolio_metrics") or {}
        if "EVA" in m_last:
            eva_final = float(m_last.get("EVA", 0.0) or 0.0)
        elif "EVA_after" in m_last:
            eva_final = float(m_last.get("EVA_after", 0.0) or 0.0)

        hhi_seg = float(m_last.get("hhi_segment", 0.0) or 0.0)
        hhi_rat = float(m_last.get("hhi_rating", 0.0) or 0.0)
        eva_vol = float(m_last.get("eva_volatility", 0.0) or 0.0)

        a_sum = last_info.get("action_summary") or {}
        if "n_sell_blocked" in a_sum:
            try:
                n_sell_blocked = float(a_sum["n_sell_blocked"])
            except Exception:
                pass

    return {
        "eva_final": float(eva_final),
        "cap_total": float(cap_total),
        "n_sell_blocked": float(n_sell_blocked),
        "hhi_seg": float(hhi_seg),
        "hhi_rat": float(hhi_rat),
        "eva_vol": float(eva_vol),
    }


# -----------------------------------------------------------
# 📊 Callback EVA + Capital + penalties (early stopping)
# -----------------------------------------------------------
class BusinessEvalCallback(BaseCallback):
    """
    Early stopping financiero:
      base = EVA_final/MM€ + cap_weight * capital_total/MM€

    En PortfolioEnv añade (si está disponible):
      - blocked_weight * n_sell_blocked
      - conc_weight * (hhi_segment + hhi_rating)
      - vol_weight * (eva_volatility/MM€)
    """
    def __init__(
        self,
        eval_env,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        patience_evals: int = 3,
        best_model_path: str | None = None,
        cap_weight: float = 0.5,
        scale_mm: float = 1e6,
        blocked_weight: float = 0.10,
        conc_weight: float = 0.05,
        vol_weight: float = 0.05,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(max(1, eval_freq))
        self.n_eval_episodes = int(max(1, n_eval_episodes))
        self.patience_evals = int(max(1, patience_evals))
        self.best_model_path = best_model_path

        self.cap_weight = float(cap_weight)
        self.scale_mm = float(scale_mm)

        self.blocked_weight = float(blocked_weight)
        self.conc_weight = float(conc_weight)
        self.vol_weight = float(vol_weight)

        self.best_metric = -np.inf
        self.no_improve = 0
        self.stopped_early = False

        self._last_eval_timestep = 0

    def _sync_eval_normalizer(self):
        try:
            from stable_baselines3.common.vec_env import VecNormalize as VN
            train_norm = self.model.get_vec_normalize_env()
            if isinstance(train_norm, VN) and isinstance(self.eval_env, VN):
                self.eval_env.obs_rms = train_norm.obs_rms
                self.eval_env.ret_rms = train_norm.ret_rms
                self.eval_env.training = False
                self.eval_env.norm_reward = False
        except Exception as e:
            logger.warning(f"⚠️ Error sincronizando VecNormalize: {e}")

    def _save_best_bundle(self):
        if not self.best_model_path:
            return
        self.model.save(self.best_model_path)
        try:
            vn = self.model.get_vec_normalize_env()
            if vn is not None and hasattr(vn, "save"):
                vn_path = self.best_model_path.replace(".zip", "_vecnormalize.pkl")
                vn.save(vn_path)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar VecNormalize del best model: {e}")

    def _score_from_episode_metrics(self, m: Dict[str, float]) -> float:
        base = (m["eva_final"] / self.scale_mm) + self.cap_weight * (m["cap_total"] / self.scale_mm)

        penalty_blocked = self.blocked_weight * float(m.get("n_sell_blocked", 0.0))
        penalty_conc = self.conc_weight * (float(m.get("hhi_seg", 0.0)) + float(m.get("hhi_rat", 0.0)))
        penalty_vol = self.vol_weight * (float(m.get("eva_vol", 0.0)) / self.scale_mm)

        return float(base - penalty_blocked - penalty_conc - penalty_vol)

    def _eval_once(self) -> float:
        self._sync_eval_normalizer()

        scores: List[float] = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            done = False
            info_list: List[Any] = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                out = self.eval_env.step(action)

                if len(out) == 4:
                    obs, rewards, dones, infos = out
                else:
                    obs, rewards, terminated, truncated, infos = out
                    dones = np.logical_or(terminated, truncated)

                d0 = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
                done = bool(d0)

                i0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                info_list.append(i0)

            m = extract_episode_business_metrics(info_list)
            scores.append(self._score_from_episode_metrics(m))

        return float(np.mean(scores)) if scores else 0.0

    def _on_step(self) -> bool:
        t = int(getattr(self.model, "num_timesteps", self.num_timesteps))
        if (t - self._last_eval_timestep) < self.eval_freq:
            return True
        self._last_eval_timestep = t

        metric = self._eval_once()
        logger.info(f"📊 Eval @ {t:,}: score={metric:.6f} (best={self.best_metric:.6f})")

        if metric > self.best_metric:
            self.best_metric = metric
            self.no_improve = 0
            if self.best_model_path:
                self._save_best_bundle()
                logger.info(f"💾 Mejor modelo actualizado: {self.best_model_path}")
        else:
            self.no_improve += 1

        if self.no_improve >= self.patience_evals:
            self.stopped_early = True
            logger.info("🛑 Early stopping: no mejora en evaluaciones sucesivas.")
            return False

        return True


# -----------------------------------------------------------
# 🏠 Fábrica de entornos
# -----------------------------------------------------------
def make_env(
    env_type: str,
    seed: int,
    loan_pool=None,
    top_k: int = 5,
    scenario: str = "baseline",
    ppo_micro: Optional[Any] = None,
    micro_vecnorm_path: str = "",
):
    if env_type == "loan":
        env = LoanEnv(seed=seed, loan_pool=loan_pool)
    elif env_type == "portfolio":
        env = PortfolioEnv(
            loan_dicts=loan_pool,
            seed=seed,
            top_k=top_k,
            scenario=scenario,
            ppo_micro=ppo_micro,
            micro_vecnormalize_path=(micro_vecnorm_path or None),
        )
    else:
        raise ValueError(f"Tipo de entorno desconocido: {env_type}")
    return Monitor(env)


def make_vec_envs(
    env_type: str,
    n_envs: int,
    seed: int,
    loan_pool,
    top_k: int = 5,
    scenario: str = "baseline",
    ppo_micro: Optional[Any] = None,
    micro_vecnorm_path: str = "",
):
    """
    Nota crítica:
      - Si pasas ppo_micro (PortfolioEnv micro re-ranking), NO puedes usar SubprocVecEnv
        porque el modelo no es picklable. Forzamos DummyVecEnv.
    """
    def thunk(rank: int):
        return lambda: make_env(
            env_type=env_type,
            seed=seed + 1000 * rank,
            loan_pool=loan_pool,
            top_k=top_k,
            scenario=scenario,
            ppo_micro=ppo_micro,
            micro_vecnorm_path=micro_vecnorm_path,
        )

    if ppo_micro is not None:
        return DummyVecEnv([thunk(0)])

    if n_envs <= 1:
        return DummyVecEnv([thunk(0)])

    return SubprocVecEnv([thunk(i) for i in range(n_envs)], start_method="spawn")


# -----------------------------------------------------------
# ⚡ PPO model
# -----------------------------------------------------------
def _auto_device(device: str) -> str:
    return "cuda" if device == "auto" and torch.cuda.is_available() else device


def build_model(cfg_train: TrainConfig, vec_env) -> PPO:
    from torch import nn

    act = dict(tanh=nn.Tanh, relu=nn.ReLU).get(cfg_train.activation_fn.lower(), nn.Tanh)
    net_arch = list(cfg_train.policy_hidden) if isinstance(cfg_train.policy_hidden, (list, tuple)) else [128, 128]
    policy_kwargs = dict(net_arch=net_arch, activation_fn=act, ortho_init=False)

    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps=cfg_train.n_steps,
        batch_size=cfg_train.batch_size,
        gamma=cfg_train.gamma,
        gae_lambda=cfg_train.gae_lambda,
        ent_coef=cfg_train.ent_coef,
        vf_coef=cfg_train.vf_coef,
        clip_range=cfg_train.clip_range,
        learning_rate=cfg_train.learning_rate,
        max_grad_norm=cfg_train.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(LOG_DIR, "tb") if cfg_train.tensorboard else None,
        device=_auto_device(cfg_train.device),
    )


# -----------------------------------------------------------
# 🧩 Carga PPO micro (opcional) para PortfolioEnv
# -----------------------------------------------------------
def load_micro_policy(micro_model_path: str, micro_vecnorm_path: str) -> Optional[Any]:
    if not micro_model_path:
        return None
    if not os.path.exists(micro_model_path):
        raise FileNotFoundError(f"❌ micro_model no existe: {micro_model_path}")

    if micro_vecnorm_path:
        if not os.path.exists(micro_vecnorm_path):
            raise FileNotFoundError(f"❌ micro_vecnorm no existe: {micro_vecnorm_path}")
        logger.info(f"🔎 Micro VecNormalize provisto: {micro_vecnorm_path}")
    else:
        logger.warning("⚠️ micro_vecnorm_path vacío: el re-ranking micro podría ir sin normalización.")

    ppo = PPO.load(micro_model_path, device="cpu")
    return ppo


def _file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# -----------------------------------------------------------
# 🚀 Entrenamiento
# -----------------------------------------------------------
def train(cfg_train: TrainConfig) -> str:
    logger.info(f"============== ENTRENAMIENTO PPO — {cfg_train.env_type.upper()} ENV ==============")

    if isinstance(cfg_train, dict):
        cfg_train = TrainConfig(**cfg_train)

    # FAST DEBUG
    if cfg_train.fast_debug or cfg.CONFIG.fast_debug:
        logger.info("⚙️ FAST-DEBUG activado")
        cfg_train.total_timesteps = 25_000
        cfg_train.n_envs = 1
        cfg_train.n_steps = 256
        cfg_train.batch_size = 128

    set_global_seeds(cfg_train.seed)

    # -------------------------------------------------------
    # 🧾 Naming por entorno (evita pisar VN/modelos)
    # -------------------------------------------------------
    env_tag = "loan" if cfg_train.env_type == "loan" else "portfolio"
    best_model_path = os.path.join(MODELS_DIR, f"best_model_{env_tag}.zip")
    last_model_path = os.path.join(MODELS_DIR, f"last_model_{env_tag}.zip")
    vn_path_final = os.path.join(MODELS_DIR, f"vecnormalize_{env_tag}.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"training_metadata_{env_tag}.json")
    ckpt_dir = os.path.join(MODELS_DIR, "checkpoints", env_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Cargar cartera
    import pandas as pd

    if not cfg_train.portfolio_path:
        raise FileNotFoundError("❌ Debes indicar --portfolio con un CSV/XLSX de préstamos")
    if not os.path.exists(cfg_train.portfolio_path):
        raise FileNotFoundError(f"❌ No existe {cfg_train.portfolio_path}")

    df = (
        pd.read_excel(cfg_train.portfolio_path)
        if cfg_train.portfolio_path.lower().endswith(".xlsx")
        else pd.read_csv(cfg_train.portfolio_path)
    )
    loan_pool = df.to_dict("records")

    # (Opcional) cargar PPO micro si entrenas PortfolioEnv con re-ranking
    ppo_micro = None
    if cfg_train.env_type == "portfolio" and cfg_train.micro_model_path:
        logger.info("🔗 Cargando PPO micro para re-ranking (PortfolioEnv micro→macro).")
        ppo_micro = load_micro_policy(cfg_train.micro_model_path, cfg_train.micro_vecnorm_path)
        if cfg_train.n_envs != 1:
            logger.warning("⚠️ Con PPO micro activo se fuerza n_envs=1 (DummyVecEnv).")
            cfg_train.n_envs = 1

    # Frecuencia/Chunks para checkpoints
    CHUNK_SIZE = 50_000

    # Normalización de reward: auto si no se especifica
    if cfg_train.norm_reward is None:
        norm_reward_train = bool(cfg_train.env_type == "loan")
    else:
        norm_reward_train = bool(cfg_train.norm_reward)

    # Vec envs + normalizador (train)
    vec_env = make_vec_envs(
        cfg_train.env_type,
        cfg_train.n_envs,
        cfg_train.seed,
        loan_pool,
        cfg_train.top_k,
        scenario=cfg_train.scenario,
        ppo_micro=ppo_micro,
        micro_vecnorm_path=cfg_train.micro_vecnorm_path,
    )
    vec_env = VecNormalize(vec_env, training=True, norm_obs=cfg_train.norm_obs, norm_reward=norm_reward_train)

    model = build_model(cfg_train, vec_env)

    # Eval env (mismo pipeline)
    eval_env = make_vec_envs(
        cfg_train.env_type,
        1,
        cfg_train.seed + 999,
        loan_pool,
        cfg_train.top_k,
        scenario=cfg_train.scenario,
        ppo_micro=ppo_micro,
        micro_vecnorm_path=cfg_train.micro_vecnorm_path,
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    eval_cb = BusinessEvalCallback(
        eval_env,
        eval_freq=cfg_train.eval_every,
        n_eval_episodes=cfg_train.eval_episodes,
        patience_evals=cfg_train.patience_evals,
        best_model_path=best_model_path,
        cap_weight=cfg_train.cap_weight,
        scale_mm=cfg_train.scale_mm,
        blocked_weight=cfg_train.blocked_weight,
        conc_weight=cfg_train.conc_weight,
        vol_weight=cfg_train.vol_weight,
    )

    # Metadata (reproducible)
    try:
        portfolio_sha = _file_sha256(cfg_train.portfolio_path)
    except Exception:
        portfolio_sha = ""

    # obs space (audit)
    try:
        obs_shape = str(getattr(vec_env.observation_space, "shape", None))
    except Exception:
        obs_shape = "unknown"

    metadata = {
        "pipeline": "PPO-NPL-STD",
        "env_type": cfg_train.env_type,
        "env_tag": env_tag,
        "scenario": cfg_train.scenario,
        "seed": cfg_train.seed,
        "timesteps_target": cfg_train.total_timesteps,
        "n_envs": cfg_train.n_envs,
        "n_steps": cfg_train.n_steps,
        "batch_size": cfg_train.batch_size,
        "gamma": cfg_train.gamma,
        "gae_lambda": cfg_train.gae_lambda,
        "ent_coef": cfg_train.ent_coef,
        "vf_coef": cfg_train.vf_coef,
        "clip_range": cfg_train.clip_range,
        "learning_rate": cfg_train.learning_rate,
        "max_grad_norm": cfg_train.max_grad_norm,
        "policy_hidden": list(cfg_train.policy_hidden),
        "activation_fn": cfg_train.activation_fn,
        "obs_space_shape": obs_shape,
        "normalization": {
            "norm_obs": cfg_train.norm_obs,
            "norm_reward_train": norm_reward_train,
        },
        "early_stop": {
            "eval_every": cfg_train.eval_every,
            "eval_episodes": cfg_train.eval_episodes,
            "patience_evals": cfg_train.patience_evals,
            "cap_weight": cfg_train.cap_weight,
            "scale_mm": cfg_train.scale_mm,
            "blocked_weight": cfg_train.blocked_weight,
            "conc_weight": cfg_train.conc_weight,
            "vol_weight": cfg_train.vol_weight,
        },
        "micro_obs_order": MICRO_OBS_ORDER,
        "portfolio_top_k": cfg_train.top_k,
        "micro_reranking": bool(cfg_train.micro_model_path),
        "micro_model_path": cfg_train.micro_model_path,
        "micro_vecnorm_path": cfg_train.micro_vecnorm_path,
        "portfolio_input": {
            "path": cfg_train.portfolio_path,
            "sha256": portfolio_sha,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
        },
        "artifacts": {
            "best_model_path": best_model_path,
            "last_model_path": last_model_path,
            "vecnormalize_final_path": vn_path_final,
            "checkpoints_dir": ckpt_dir,
            "metadata_path": metadata_path,
            "best_vecnormalize_path": best_model_path.replace(".zip", "_vecnormalize.pkl"),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Loop entrenamiento
    steps_done = 0
    start = time.time()

    while steps_done < cfg_train.total_timesteps:
        chunk = min(CHUNK_SIZE, cfg_train.total_timesteps - steps_done)
        model.learn(total_timesteps=chunk, callback=eval_cb, reset_num_timesteps=False)

        steps_done = int(getattr(model, "num_timesteps", steps_done + chunk))

        ckpt_path = os.path.join(ckpt_dir, f"ppo_{steps_done:07d}.zip")
        model.save(ckpt_path)

        # Guardar VecNormalize del checkpoint (clave para reproducibilidad)
        try:
            vn = model.get_vec_normalize_env()
            if vn is not None and hasattr(vn, "save"):
                vn_ckpt = os.path.join(ckpt_dir, f"ppo_{steps_done:07d}_vecnormalize.pkl")
                vn.save(vn_ckpt)
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar VN checkpoint: {e}")

        logger.info(f"📂 Guardado checkpoint: {ckpt_path}")

        if eval_cb.stopped_early:
            logger.info("🛑 Entrenamiento detenido por early stopping.")
            break

    # Guardar modelo final (por entorno)
    model.save(last_model_path)

    # Guardar normalizador final (por entorno)
    if hasattr(vec_env, "save") and cfg_train.save_scaler:
        vec_env.save(vn_path_final)

    duration = (time.time() - start) / 60.0
    logger.info(f"⏱️ Duración total: {duration:.2f} min")
    logger.info(f"🏆 Mejor modelo: {best_model_path}")
    logger.info(f"📦 Último modelo: {last_model_path}")
    logger.info(f"🧠 VecNormalize final: {vn_path_final}")
    logger.info(f"📁 Checkpoints: {ckpt_dir}")
    logger.info(f"🧾 Metadata: {metadata_path}")

    return best_model_path


# -----------------------------------------------------------
# ▶️ CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entrenar PPO sobre LoanEnv o PortfolioEnv")
    p.add_argument("--portfolio", required=True, help="Ruta a CSV/XLSX con la cartera")
    p.add_argument("--env-type", choices=["loan", "portfolio"], default="loan")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--scenario", type=str, default="baseline")
    p.add_argument("--total-steps", type=int, default=cfg.CONFIG.ppo.total_timesteps)
    p.add_argument("--fast-debug", action="store_true")
    p.add_argument("--save-scaler", type=int, default=1)

    # opcional para PortfolioEnv micro re-ranking
    p.add_argument("--micro-model", type=str, default="", help="Ruta a modelo PPO micro (LoanEnv) .zip")
    p.add_argument("--micro-vecnorm", type=str, default="", help="Ruta a VecNormalize micro .pkl (opcional)")

    args = p.parse_args()

    cfg_train = TrainConfig(
        env_type=args.env_type,
        portfolio_path=args.portfolio,
        top_k=int(args.top_k),
        scenario=str(args.scenario),
        fast_debug=bool(args.fast_debug),
        total_timesteps=int(args.total_steps),
        save_scaler=bool(args.save_scaler),
        micro_model_path=str(args.micro_model),
        micro_vecnorm_path=str(args.micro_vecnorm),
    )

    best = train(cfg_train)
    logger.info(f"🏁 Entrenamiento completado — Best model: {best}")