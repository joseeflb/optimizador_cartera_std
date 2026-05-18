# -*- coding: utf-8 -*-
# ============================================================
# _tmp/validate_agents.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Script standalone de validación final de ambos agentes (loan + portfolio): métricas de rendimiento, distribución de acciones, KPIs de negocio.
# ============================================================
"""
Validacion final de ambos agentes entrenados (Loan + Portfolio).
Genera un informe completo con:
  1. Metricas de rendimiento por agente
  2. Distribucion de acciones
  3. Comparacion con baselines (random, hold)
  4. Metricas de negocio (EVA, capital, riesgo)
"""
import sys, os, json, logging
from collections import Counter, defaultdict
from typing import Dict, List, Any
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Suprimir logs verbosos
for n in ("portfolio_env", "price_simulator", "guardrails", "restructure_optimizer"):
    logging.getLogger(n).setLevel(logging.WARNING)

import pandas as pd
import config as cfg
from env.loan_env import LoanEnv
from env.portfolio_env import PortfolioEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    raise SystemExit("SB3 not available")

MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# =====================================================================
#  Utilidades
# =====================================================================
def _load_portfolio():
    df = pd.read_excel("data/portfolio_synth.xlsx")
    return df.to_dict(orient="records")


def _action_entropy(counts: Dict[int, int], n_actions: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([counts.get(a, 0) / total for a in range(n_actions)])
    probs = probs[probs > 0]
    h = float(-np.sum(probs * np.log(probs + 1e-10)))
    h_max = float(np.log(n_actions + 1e-10))
    return h / max(h_max, 1e-10)


# =====================================================================
#  1. Validacion agente LOAN
# =====================================================================
def validate_loan_agent(n_episodes: int = 50) -> Dict[str, Any]:
    print("=" * 60)
    print("  VALIDACION AGENTE LOAN (micro)")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "best_model_loan.zip")
    vn_path = os.path.join(MODELS_DIR, "vecnormalize_loan.pkl")

    if not os.path.exists(model_path):
        print(f"[SKIP] Modelo no encontrado: {model_path}")
        return {}

    loans = _load_portfolio()

    # Env con VecNormalize
    def _make_env():
        return LoanEnv(seed=999, loan_pool=loans)

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(vn_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    rewards = []
    lengths = []
    action_counts = Counter()
    total_steps = 0

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_r = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, d, info = vec_env.step(action)
            d0 = d[0] if isinstance(d, (list, np.ndarray)) else d
            r0 = r[0] if isinstance(r, (list, np.ndarray)) else r
            a0 = action[0] if isinstance(action, (list, np.ndarray)) else action
            ep_r += float(r0)
            ep_len += 1
            action_counts[int(a0)] += 1
            total_steps += 1
            done = bool(d0)
        rewards.append(ep_r)
        lengths.append(ep_len)

    # Random baseline
    rand_rewards = []
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = [vec_env.action_space.sample()]
            obs, r, d, _ = vec_env.step(action)
            d0 = d[0] if isinstance(d, (list, np.ndarray)) else d
            r0 = r[0] if isinstance(r, (list, np.ndarray)) else r
            ep_r += float(r0)
            done = bool(d0)
        rand_rewards.append(ep_r)

    results = {
        "agent": "loan",
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "random_baseline": float(np.mean(rand_rewards)),
        "improvement_vs_random": float(np.mean(rewards) - np.mean(rand_rewards)),
        "improvement_pct": float((np.mean(rewards) - np.mean(rand_rewards)) / abs(np.mean(rand_rewards)) * 100),
        "action_distribution": {str(k): v for k, v in sorted(action_counts.items())},
        "entropy_ratio": _action_entropy(action_counts, 3),
        "dominant_action": action_counts.most_common(1)[0][0] if action_counts else -1,
        "dominant_pct": action_counts.most_common(1)[0][1] / max(total_steps, 1) * 100 if action_counts else 0,
    }

    ACTION_NAMES = {0: "MANTENER", 1: "REESTRUCTURAR", 2: "VENDER"}
    print(f"\n  Reward medio: {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")
    print(f"  Random baseline: {results['random_baseline']:.4f}")
    print(f"  Mejora vs random: {results['improvement_pct']:.1f}%")
    print(f"  Entropy ratio: {results['entropy_ratio']:.3f}")
    print(f"  Distribucion acciones:")
    for a, cnt in sorted(action_counts.items()):
        pct = cnt / total_steps * 100
        name = ACTION_NAMES.get(a, f"a{a}")
        print(f"    {name:15s} (a{a}): {cnt:6d} ({pct:5.1f}%)")

    return results


# =====================================================================
#  2. Validacion agente PORTFOLIO
# =====================================================================
def validate_portfolio_agent(n_episodes: int = 20) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("  VALIDACION AGENTE PORTFOLIO (macro)")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "best_model_portfolio.zip")
    vn_path = os.path.join(MODELS_DIR, "vecnormalize_portfolio.pkl")

    if not os.path.exists(model_path):
        print(f"[SKIP] Modelo no encontrado: {model_path}")
        return {}

    loans = _load_portfolio()

    def _make_env():
        return PortfolioEnv(loan_dicts=loans, seed=888, top_k=5, scenario="baseline")

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecNormalize.load(vn_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    ACTION_NAMES = {
        0: "HOLD", 1: "SELL_1_EVA", 2: "SELL_TOPK_EVA", 3: "RESTRUCT_1_EVA",
        4: "RESTRUCT_TOPK_EVA", 5: "SELL_1_RORWA", 6: "SELL_TOPK_RORWA",
        7: "RESTRUCT_1_PTI", 8: "RESTRUCT_TOPK_PTI", 9: "MIX_SELL_RESTRUCT",
        10: "RULE_NEG_EVA", 11: "HOLD_PASSIVE"
    }

    rewards = []
    action_counts = Counter()
    total_steps = 0
    biz_metrics = defaultdict(list)

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            out = vec_env.step(action)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                d = np.logical_or(term, trunc)
            else:
                obs, r, d, info = out
            d0 = d[0] if isinstance(d, (list, np.ndarray)) else d
            r0 = r[0] if isinstance(r, (list, np.ndarray)) else r
            a0 = action[0] if isinstance(action, (list, np.ndarray)) else action
            ep_r += float(r0)
            action_counts[int(a0)] += 1
            total_steps += 1
            done = bool(d0)
        rewards.append(ep_r)

    # Random baseline
    rand_rewards = []
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = [vec_env.action_space.sample()]
            out = vec_env.step(action)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                d = np.logical_or(term, trunc)
            else:
                obs, r, d, info = out
            d0 = d[0] if isinstance(d, (list, np.ndarray)) else d
            r0 = r[0] if isinstance(r, (list, np.ndarray)) else r
            ep_r += float(r0)
            done = bool(d0)
        rand_rewards.append(ep_r)

    # Hold baseline
    hold_rewards = []
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = np.array([0])  # HOLD
            out = vec_env.step(action)
            if len(out) == 5:
                obs, r, term, trunc, info = out
                d = np.logical_or(term, trunc)
            else:
                obs, r, d, info = out
            d0 = d[0] if isinstance(d, (list, np.ndarray)) else d
            r0 = r[0] if isinstance(r, (list, np.ndarray)) else r
            ep_r += float(r0)
            done = bool(d0)
        hold_rewards.append(ep_r)

    results = {
        "agent": "portfolio",
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "random_baseline": float(np.mean(rand_rewards)),
        "hold_baseline": float(np.mean(hold_rewards)),
        "improvement_vs_random": float(np.mean(rewards) - np.mean(rand_rewards)),
        "improvement_vs_random_pct": float((np.mean(rewards) - np.mean(rand_rewards)) / abs(np.mean(rand_rewards)) * 100),
        "improvement_vs_hold": float(np.mean(rewards) - np.mean(hold_rewards)),
        "improvement_vs_hold_pct": float((np.mean(rewards) - np.mean(hold_rewards)) / abs(np.mean(hold_rewards)) * 100),
        "action_distribution": {str(k): v for k, v in sorted(action_counts.items())},
        "entropy_ratio": _action_entropy(action_counts, 12),
        "dominant_action": action_counts.most_common(1)[0][0] if action_counts else -1,
        "dominant_pct": action_counts.most_common(1)[0][1] / max(total_steps, 1) * 100 if action_counts else 0,
    }

    print(f"\n  Reward medio (agent):  {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")
    print(f"  Random baseline:       {results['random_baseline']:.4f}")
    print(f"  Hold baseline:         {results['hold_baseline']:.4f}")
    print(f"  Mejora vs random:      {results['improvement_vs_random_pct']:.1f}%")
    print(f"  Mejora vs hold:        {results['improvement_vs_hold_pct']:.1f}%")
    print(f"  Entropy ratio:         {results['entropy_ratio']:.3f}")
    print(f"\n  Distribucion acciones:")
    for a, cnt in sorted(action_counts.items()):
        pct = cnt / total_steps * 100
        name = ACTION_NAMES.get(a, f"a{a}")
        print(f"    {name:22s} (a{a:2d}): {cnt:5d} ({pct:5.1f}%)")

    return results


# =====================================================================
#  Main
# =====================================================================
def main():
    print("\n" + "#" * 60)
    print("#  VALIDACION FINAL — AMBOS AGENTES RL")
    print("#  Optimizador Cartera NPL · Basel III STD")
    print("#" * 60)

    loan_results = validate_loan_agent(n_episodes=50)
    portfolio_results = validate_portfolio_agent(n_episodes=20)

    # Guardar resultados
    all_results = {
        "loan": loan_results,
        "portfolio": portfolio_results,
        "summary": {
            "loan_reward": loan_results.get("mean_reward", None),
            "loan_vs_random_pct": loan_results.get("improvement_pct", None),
            "portfolio_reward": portfolio_results.get("mean_reward", None),
            "portfolio_vs_random_pct": portfolio_results.get("improvement_vs_random_pct", None),
            "portfolio_vs_hold_pct": portfolio_results.get("improvement_vs_hold_pct", None),
        }
    }

    report_path = os.path.join(REPORTS_DIR, "validation_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Resultados guardados: {report_path}")

    # Resumen final
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    if loan_results:
        print(f"  LOAN:      reward={loan_results['mean_reward']:.4f}  vs_random={loan_results.get('improvement_pct',0):.1f}%  entropy={loan_results['entropy_ratio']:.3f}")
    if portfolio_results:
        print(f"  PORTFOLIO: reward={portfolio_results['mean_reward']:.4f}  vs_random={portfolio_results.get('improvement_vs_random_pct',0):.1f}%  vs_hold={portfolio_results.get('improvement_vs_hold_pct',0):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
