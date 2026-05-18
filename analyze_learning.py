# -*- coding: utf-8 -*-
# ============================================================
# analyze_learning.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Análisis completo del aprendizaje PPO (TensorBoard + checkpoints + policy distribution + diagnósticos). Genera artefactos en reports/learning_analysis/.
# ============================================================
"""
============================================================
analyze_learning.py — Análisis completo del aprendizaje PPO
(TensorBoard + Checkpoints + Policy Distribution + Diagnóstico)
============================================================

Genera:
  reports/learning_analysis/
    ├── 01_loss_curves.png
    ├── 02_policy_gradient_and_kl.png
    ├── 03_entropy_and_clip.png
    ├── 04_explained_variance.png
    ├── 05_fps_throughput.png
    ├── 06_checkpoint_evaluation.png
    ├── 07_policy_distribution_evolution.png
    ├── 08_reward_curve.png            (si hay rollout data)
    ├── 09_portfolio_comparison.png
    ├── 10_summary_dashboard.png
    └── learning_report.txt

Uso:
    python analyze_learning.py
"""
from __future__ import annotations

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ─── Rutas ───────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

LOGS_DIR = os.path.join(ROOT_DIR, "logs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "learning_analysis")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─── Imports ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titleweight"] = "bold"

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════
#  1. LECTURA DE TENSORBOARD
# ═══════════════════════════════════════════════════════════════
def load_tb_scalars(logdir: str) -> dict:
    """Lee todos los scalars de un directorio TensorBoard."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {"steps": np.array(steps), "values": np.array(values)}
    return data


def _parse_reward_from_logs():
    """Parse reward data from BusinessEvalCallback training logs.
    Returns dict: {"loan": {"steps": np.array, "values": np.array},
                   "portfolio": {"steps": ..., "values": ...}}
    """
    import re
    reward_data = {}
    eval_re = re.compile(
        r'\[(MICRO/loan|MACRO/portfolio)\]\s*EVAL\s*@\s*([\d,]+)\s*\|\s*reward=([-\d.]+)'
    )

    log_files = [
        os.path.join(LOGS_DIR, "train_both_run.log"),
        os.path.join(LOGS_DIR, "portfolio_train_v4.log"),
        os.path.join(LOGS_DIR, "train_subagents.log"),
    ]

    for log_path in log_files:
        if not os.path.isfile(log_path):
            continue
        try:
            # Detect encoding: check for UTF-16 BOM
            enc = "utf-8"
            with open(log_path, "rb") as fb:
                bom = fb.read(2)
                if bom == b'\xff\xfe':
                    enc = "utf-16-le"
                elif bom == b'\xfe\xff':
                    enc = "utf-16-be"
            with open(log_path, "r", encoding=enc, errors="replace") as f:
                for line in f:
                    m = eval_re.search(line)
                    if m:
                        agent_tag = m.group(1)
                        step = int(m.group(2).replace(",", ""))
                        reward = float(m.group(3))
                        key = "loan" if "loan" in agent_tag else "portfolio"
                        if key not in reward_data:
                            reward_data[key] = {"steps": [], "values": []}
                        reward_data[key]["steps"].append(step)
                        reward_data[key]["values"].append(reward)
        except Exception:
            pass

    # Convert to numpy, keep only the last run for each agent
    result = {}
    for key, rd in reward_data.items():
        steps = np.array(rd["steps"])
        values = np.array(rd["values"])
        if len(steps) == 0:
            continue
        # If there are multiple runs, the steps will reset — keep the last run
        resets = np.where(np.diff(steps) < 0)[0]
        if len(resets) > 0:
            start_idx = resets[-1] + 1
            steps = steps[start_idx:]
            values = values[start_idx:]
        result[key] = {"steps": steps, "values": values}

    for key in result:
        print(f"  [REWARD LOG] {key}: {len(result[key]['steps'])} evals, "
              f"range [{result[key]['values'][0]:.1f} .. {result[key]['values'][-1]:.1f}]")
    return result


def find_tb_runs():
    """Encuentra automáticamente los runs de TensorBoard."""
    tb_base = os.path.join(LOGS_DIR, "tb")
    runs = {}

    # Buscar runs por tipo
    for sub in ["loan", "portfolio"]:
        sub_dir = os.path.join(tb_base, sub)
        if os.path.isdir(sub_dir):
            # Tomar el run con más datos (por tamaño de archivo)
            best_run = None
            best_size = 0
            for d in os.listdir(sub_dir):
                full = os.path.join(sub_dir, d)
                if os.path.isdir(full):
                    for f in os.listdir(full):
                        if f.startswith("events.out"):
                            sz = os.path.getsize(os.path.join(full, f))
                            if sz > best_size:
                                best_size = sz
                                best_run = full
            if best_run:
                runs[sub] = best_run

    # PPO_0 (legacy / general)
    ppo0 = os.path.join(tb_base, "PPO_0")
    if os.path.isdir(ppo0):
        runs["general"] = ppo0

    return runs


# ═══════════════════════════════════════════════════════════════
#  2. FUNCIONES DE PLOTTING
# ═══════════════════════════════════════════════════════════════
def _smooth(values, weight=0.85):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)


def _k_formatter(x, p):
    """Format axis to show K for thousands."""
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def plot_loss_curves(data: dict, label: str, save_path: str):
    """Plot total loss, value loss, and policy gradient loss."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Curvas de Pérdida — {label}", fontsize=14, fontweight="bold")

    metrics = [
        ("train/loss", "Loss Total", "tab:red"),
        ("train/value_loss", "Value Loss", "tab:blue"),
        ("train/policy_gradient_loss", "Policy Gradient Loss", "tab:green"),
    ]

    for ax, (tag, title, color) in zip(axes, metrics):
        if tag in data:
            steps = data[tag]["steps"]
            vals = data[tag]["values"]
            ax.plot(steps, vals, alpha=0.3, color=color, linewidth=0.8)
            ax.plot(steps, _smooth(vals), color=color, linewidth=2, label="Suavizado")
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Valor")
            ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No disponible", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


def plot_pg_and_kl(data: dict, label: str, save_path: str):
    """Policy gradient loss y KL divergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Policy Gradient & KL — {label}", fontsize=14, fontweight="bold")

    # Policy gradient loss
    if "train/policy_gradient_loss" in data:
        d = data["train/policy_gradient_loss"]
        axes[0].plot(d["steps"], d["values"], alpha=0.3, color="tab:green", linewidth=0.8)
        axes[0].plot(d["steps"], _smooth(d["values"]), color="tab:green", linewidth=2)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("Policy Gradient Loss")
        axes[0].set_xlabel("Timesteps")
        axes[0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))

        # Annotate final value
        final_val = d["values"][-1]
        axes[0].annotate(f"Final: {final_val:.6f}", xy=(d["steps"][-1], final_val),
                         fontsize=9, ha="right", color="tab:green")

    # Approx KL
    if "train/approx_kl" in data:
        d = data["train/approx_kl"]
        axes[1].plot(d["steps"], d["values"], alpha=0.3, color="tab:orange", linewidth=0.8)
        axes[1].plot(d["steps"], _smooth(d["values"]), color="tab:orange", linewidth=2)
        axes[1].axhline(y=0.01, color="red", linestyle="--", alpha=0.5, label="Target KL (~0.01)")
        axes[1].axhline(y=0.03, color="darkred", linestyle=":", alpha=0.5, label="Alerta KL (>0.03)")
        axes[1].set_title("Approx KL Divergence")
        axes[1].set_xlabel("Timesteps")
        axes[1].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        axes[1].legend(fontsize=9)

        # KL health diagnostic
        mean_kl = np.mean(d["values"])
        final_kl = d["values"][-1]
        axes[1].annotate(f"Mean: {mean_kl:.5f}\nFinal: {final_kl:.5f}",
                         xy=(d["steps"][-1], final_kl), fontsize=9, ha="right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


def plot_entropy_and_clip(data: dict, label: str, save_path: str):
    """Entropy loss y clip fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Entropía & Clipping — {label}", fontsize=14, fontweight="bold")

    # Entropy
    if "train/entropy_loss" in data:
        d = data["train/entropy_loss"]
        axes[0].plot(d["steps"], d["values"], alpha=0.3, color="tab:purple", linewidth=0.8)
        axes[0].plot(d["steps"], _smooth(d["values"]), color="tab:purple", linewidth=2)
        axes[0].set_title("Entropy Loss (exploración)")
        axes[0].set_xlabel("Timesteps")
        axes[0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))

        # Max entropy for 3 actions = -ln(1/3) ≈ -1.0986
        axes[0].axhline(y=-1.0986, color="gray", linestyle="--", alpha=0.5, label="Max entropy (3 acciones)")
        axes[0].legend(fontsize=9)

        # Anotación de diagnóstico
        final_ent = d["values"][-1]
        initial_ent = d["values"][0]
        axes[0].annotate(f"Inicio: {initial_ent:.4f}\nFinal: {final_ent:.4f}",
                         xy=(d["steps"][-1], final_ent), fontsize=9, ha="right")

    # Clip fraction
    if "train/clip_fraction" in data:
        d = data["train/clip_fraction"]
        axes[1].plot(d["steps"], d["values"], alpha=0.3, color="tab:red", linewidth=0.8)
        axes[1].plot(d["steps"], _smooth(d["values"]), color="tab:red", linewidth=2)
        axes[1].axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="Rango saludable (~0.1)")
        axes[1].axhline(y=0.3, color="red", linestyle=":", alpha=0.5, label="Alta clipping (>0.3)")
        axes[1].set_title("Clip Fraction (estabilidad de updates)")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


def plot_explained_variance(data: dict, label: str, save_path: str):
    """Explained variance del value function."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Explained Variance — {label}", fontsize=14, fontweight="bold")

    if "train/explained_variance" in data:
        d = data["train/explained_variance"]
        ax.plot(d["steps"], d["values"], alpha=0.3, color="tab:cyan", linewidth=0.8)
        ax.plot(d["steps"], _smooth(d["values"]), color="tab:cyan", linewidth=2, label="EV suavizado")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="EV=0 (no learning)")
        ax.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="EV=1 (perfect predictor)")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Explained Variance")
        ax.set_ylim(-0.5, 1.1)
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax.legend(fontsize=10)

        # Diagnosis zones
        final_ev = d["values"][-1]
        mean_ev = np.mean(d["values"][-50:]) if len(d["values"]) > 50 else np.mean(d["values"])
        ax.fill_between(d["steps"], 0, 1, alpha=0.05, color="green")
        ax.fill_between(d["steps"], -0.5, 0, alpha=0.05, color="red")

        status = "EXCELENTE" if mean_ev > 0.8 else "BUENO" if mean_ev > 0.5 else "ACEPTABLE" if mean_ev > 0 else "POBRE"
        ax.annotate(f"EV medio (últimos 50): {mean_ev:.4f} → {status}",
                   xy=(d["steps"][-1], mean_ev), fontsize=10, ha="right",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


def plot_fps(data: dict, label: str, save_path: str):
    """FPS throughput."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"Throughput (FPS) — {label}", fontsize=14, fontweight="bold")

    if "time/fps" in data:
        d = data["time/fps"]
        ax.plot(d["steps"], d["values"], color="tab:gray", linewidth=1.5, alpha=0.7)
        ax.fill_between(d["steps"], 0, d["values"], alpha=0.15, color="tab:gray")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Frames/segundo")
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))

        mean_fps = np.mean(d["values"])
        ax.axhline(y=mean_fps, color="orange", linestyle="--", alpha=0.5, label=f"Media: {mean_fps:.0f} FPS")
        ax.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


def plot_reward_curve(data: dict, label: str, save_path: str, reward_log: dict = None):
    """Reward medio por episodio from BusinessEvalCallback logs + explained variance from TB."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Métricas de Entrenamiento — {label}", fontsize=14, fontweight="bold")

    # Determine which agent this is
    agent_key = None
    if reward_log:
        lbl_lower = label.lower()
        if "loan" in lbl_lower:
            agent_key = "loan"
        elif "portfolio" in lbl_lower:
            agent_key = "portfolio"

    # Left panel: Reward curve from training logs
    has_reward = False
    if agent_key and reward_log and agent_key in reward_log:
        rd = reward_log[agent_key]
        axes[0].plot(rd["steps"], rd["values"], alpha=0.4, color="tab:blue", linewidth=1, marker=".", markersize=3)
        axes[0].plot(rd["steps"], _smooth(rd["values"], 0.8), color="tab:blue", linewidth=2.5, label="Suavizado (EMA)")
        axes[0].set_title("Reward medio por episodio (BusinessEvalCallback)")
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Reward")
        axes[0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        axes[0].legend(fontsize=9)

        # Annotate best
        best_idx = np.argmax(rd["values"])
        best_step = rd["steps"][best_idx]
        best_val = rd["values"][best_idx]
        axes[0].annotate(f"Best: {best_val:.1f}\n@ {best_step:,} steps",
                         xy=(best_step, best_val), fontsize=9,
                         arrowprops=dict(arrowstyle="->", color="green"),
                         xytext=(best_step * 0.6, best_val * 0.85),
                         bbox=dict(facecolor="lightyellow", alpha=0.8))

        # Trend
        if len(rd["values"]) > 5:
            z = np.polyfit(range(len(rd["values"])), rd["values"], 1)
            trend_dir = "MEJORANDO ↑" if z[0] > 0 else "EMPEORANDO ↓" if z[0] < 0 else "ESTABLE →"
            axes[0].annotate(f"Tendencia: {trend_dir}", xy=(0.02, 0.95), xycoords="axes fraction",
                           fontsize=10, bbox=dict(facecolor="lightgreen" if z[0] > 0 else "lightyellow", alpha=0.8))
        has_reward = True

    if not has_reward and "rollout/ep_rew_mean" in data:
        d = data["rollout/ep_rew_mean"]
        axes[0].plot(d["steps"], d["values"], alpha=0.4, color="tab:blue", linewidth=1)
        axes[0].plot(d["steps"], _smooth(d["values"], 0.8), color="tab:blue", linewidth=2)
        axes[0].set_title("Reward medio por episodio")
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Reward")
        axes[0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        has_reward = True

    if not has_reward:
        axes[0].text(0.5, 0.5, "Reward no disponible", ha="center", va="center",
                    transform=axes[0].transAxes, fontsize=12, color="gray")
        axes[0].set_title("Reward medio por episodio")

    # Right panel: Explained Variance (more useful than ep_len_mean which is often unavailable)
    if "train/explained_variance" in data:
        d = data["train/explained_variance"]
        axes[1].plot(d["steps"], d["values"], alpha=0.3, color="tab:purple", linewidth=0.8)
        axes[1].plot(d["steps"], _smooth(d["values"]), color="tab:purple", linewidth=2, label="Suavizado")
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].axhline(y=1, color="green", linestyle="--", alpha=0.3, label="Ideal (1.0)")
        axes[1].set_title("Explained Variance")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("EV")
        axes[1].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        axes[1].legend(fontsize=9)

        final_ev = d["values"][-1]
        axes[1].annotate(f"Final: {final_ev:.3f}", xy=(d["steps"][-1], final_ev),
                        fontsize=9, ha="right", color="tab:purple",
                        bbox=dict(facecolor="white", alpha=0.7))
    elif "rollout/ep_len_mean" in data:
        d = data["rollout/ep_len_mean"]
        axes[1].plot(d["steps"], d["values"], alpha=0.4, color="tab:orange", linewidth=1)
        axes[1].plot(d["steps"], _smooth(d["values"], 0.8), color="tab:orange", linewidth=2)
        axes[1].set_title("Longitud media de episodio")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Steps/episodio")
        axes[1].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    else:
        axes[1].text(0.5, 0.5, "Métricas adicionales\nno disponibles", ha="center", va="center",
                    transform=axes[1].transAxes, fontsize=12, color="gray")
        axes[1].set_title("Métricas adicionales")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════
#  3. EVALUACIÓN DE CHECKPOINTS
# ═══════════════════════════════════════════════════════════════
def evaluate_checkpoints():
    """Carga cada checkpoint y extrae métricas del modelo (weights stats, policy distribution)."""
    ckpt_dir = os.path.join(MODELS_DIR, "checkpoints")
    results = []

    if not HAS_SB3 or not HAS_TORCH:
        print("  [WARN] SB3/torch no disponible, saltando evaluación de checkpoints")
        return results

    # Buscar checkpoints .zip (excluir subdirectorio loan/)
    ckpts = []
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith(".zip") and f.startswith("ppo_"):
            ckpts.append(os.path.join(ckpt_dir, f))

    # También buscar en loan/ subdirectory
    loan_ckpt_dir = os.path.join(ckpt_dir, "loan")
    if os.path.isdir(loan_ckpt_dir):
        for f in sorted(os.listdir(loan_ckpt_dir)):
            if f.endswith(".zip") and f.startswith("ppo_"):
                ckpts.append(os.path.join(loan_ckpt_dir, f))

    if not ckpts:
        print("  [WARN] No se encontraron checkpoints")
        return results

    print(f"  Evaluando {len(ckpts)} checkpoints...")

    for ckpt_path in ckpts:
        try:
            model = PPO.load(ckpt_path, device="cpu")
            fname = os.path.basename(ckpt_path)

            # Extraer timestep del nombre
            step = int(fname.replace("ppo_", "").replace(".zip", ""))

            # Estadísticas de pesos de la policy network
            policy_params = model.policy.state_dict()
            weight_stats = {}
            total_params = 0
            for name, param in policy_params.items():
                total_params += param.numel()
                weight_stats[name] = {
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "min": float(param.min()),
                    "max": float(param.max()),
                    "norm": float(param.norm()),
                }

            # Extraer action logits distribution en una observación dummy
            obs_dim = model.observation_space.shape[0]
            dummy_obs = torch.zeros(1, obs_dim)
            with torch.no_grad():
                dist = model.policy.get_distribution(dummy_obs)
                action_probs = dist.distribution.probs.numpy().flatten()

            results.append({
                "step": step,
                "path": ckpt_path,
                "total_params": total_params,
                "weight_mean": float(np.mean([ws["mean"] for ws in weight_stats.values()])),
                "weight_std": float(np.mean([ws["std"] for ws in weight_stats.values()])),
                "weight_norm": float(np.mean([ws["norm"] for ws in weight_stats.values()])),
                "action_probs": action_probs.tolist(),
                "entropy": float(-np.sum(action_probs * np.log(action_probs + 1e-8))),
            })
        except Exception as e:
            print(f"    [WARN] Error en {ckpt_path}: {e}")

    results = sorted(results, key=lambda x: x["step"])
    return results


def plot_checkpoint_evolution(ckpt_results: list, save_path: str):
    """Evolución de métricas por checkpoint."""
    if not ckpt_results:
        return

    steps = [r["step"] for r in ckpt_results]
    weight_means = [r["weight_mean"] for r in ckpt_results]
    weight_stds = [r["weight_std"] for r in ckpt_results]
    weight_norms = [r["weight_norm"] for r in ckpt_results]
    entropies = [r["entropy"] for r in ckpt_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evolución de Checkpoints PPO", fontsize=14, fontweight="bold")

    # Weight mean
    axes[0, 0].plot(steps, weight_means, "o-", color="tab:blue", markersize=4)
    axes[0, 0].set_title("Media de pesos (policy)")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))

    # Weight std
    axes[0, 1].plot(steps, weight_stds, "o-", color="tab:orange", markersize=4)
    axes[0, 1].set_title("Desviación estándar de pesos")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].xaxis.set_major_formatter(FuncFormatter(_k_formatter))

    # Weight norm
    axes[1, 0].plot(steps, weight_norms, "o-", color="tab:green", markersize=4)
    axes[1, 0].set_title("Norma media de pesos")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))

    # Entropy
    axes[1, 1].plot(steps, entropies, "o-", color="tab:purple", markersize=4)
    axes[1, 1].axhline(y=np.log(3), color="gray", linestyle="--", alpha=0.5, label="Max entropy (uniform)")
    axes[1, 1].set_title("Entropía de la política (en obs 0)")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Entropía (nats)")
    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


def plot_policy_distribution_evolution(ckpt_results: list, save_path: str):
    """Cómo cambia la distribución de acciones con el entrenamiento."""
    if not ckpt_results:
        return

    action_names = ["MANTENER", "REESTRUCTURAR", "VENDER"]
    steps = [r["step"] for r in ckpt_results]
    probs = np.array([r["action_probs"] for r in ckpt_results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Evolución de la Distribución de Política", fontsize=14, fontweight="bold")

    # Stacked area plot
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    axes[0].stackplot(steps, probs.T, labels=action_names, colors=colors, alpha=0.8)
    axes[0].set_title("Probabilidades de acción (obs=0, evolución)")
    axes[0].set_xlabel("Timesteps")
    axes[0].set_ylabel("Probabilidad")
    axes[0].set_ylim(0, 1)
    axes[0].xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    axes[0].legend(loc="center right")

    # Bar chart: first vs mid vs last
    indices = [0, len(ckpt_results) // 2, -1]
    labels_bar = ["Inicio", "Mitad", "Final"]
    x = np.arange(len(action_names))
    width = 0.25

    for i, (idx, lbl) in enumerate(zip(indices, labels_bar)):
        c = ckpt_results[idx]
        axes[1].bar(x + i * width, c["action_probs"], width, label=f"{lbl} ({c['step']:,})")

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(action_names)
    axes[1].set_title("Comparación inicio / mitad / final")
    axes[1].set_ylabel("Probabilidad")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════
#  4. COMPARACIÓN LOAN vs PORTFOLIO
# ═══════════════════════════════════════════════════════════════
def plot_comparison(all_data: dict, save_path: str):
    """Compara métricas clave loan vs portfolio."""
    if "loan" not in all_data or "portfolio" not in all_data:
        print("  [SKIP] No se puede comparar loan vs portfolio (faltan datos)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Comparación Loan (micro) vs Portfolio (macro)", fontsize=14, fontweight="bold")

    metrics = [
        ("train/loss", "Loss Total", axes[0, 0]),
        ("train/value_loss", "Value Loss", axes[0, 1]),
        ("train/entropy_loss", "Entropy Loss", axes[1, 0]),
        ("train/explained_variance", "Explained Variance", axes[1, 1]),
    ]

    for tag, title, ax in metrics:
        for name, data in [("Loan (micro)", all_data["loan"]), ("Portfolio (macro)", all_data["portfolio"])]:
            if tag in data:
                d = data[tag]
                ax.plot(d["steps"], _smooth(d["values"]), linewidth=2, label=name)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════
#  5. DASHBOARD RESUMEN
# ═══════════════════════════════════════════════════════════════
def plot_summary_dashboard(all_data: dict, ckpt_results: list, save_path: str):
    """Panel resumen con métricas clave."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("DASHBOARD — Análisis de Aprendizaje PPO\nOptimizador de Carteras NPL (Basilea III · STD)",
                 fontsize=16, fontweight="bold")

    # 1. Loss loan
    ax1 = fig.add_subplot(gs[0, 0:2])
    if "loan" in all_data and "train/loss" in all_data["loan"]:
        d = all_data["loan"]["train/loss"]
        ax1.plot(d["steps"], _smooth(d["values"]), color="tab:red", linewidth=2)
        ax1.set_title("Loss — Loan (micro)")
        ax1.xaxis.set_major_formatter(FuncFormatter(_k_formatter))

    # 2. Loss portfolio
    ax2 = fig.add_subplot(gs[0, 2:4])
    if "portfolio" in all_data and "train/loss" in all_data["portfolio"]:
        d = all_data["portfolio"]["train/loss"]
        ax2.plot(d["steps"], _smooth(d["values"]), color="tab:blue", linewidth=2)
        ax2.set_title("Loss — Portfolio (macro)")
        ax2.xaxis.set_major_formatter(FuncFormatter(_k_formatter))

    # 3. Entropy comparison
    ax3 = fig.add_subplot(gs[1, 0:2])
    for name, key, color in [("Loan", "loan", "tab:red"), ("Portfolio", "portfolio", "tab:blue")]:
        if key in all_data and "train/entropy_loss" in all_data[key]:
            d = all_data[key]["train/entropy_loss"]
            ax3.plot(d["steps"], _smooth(d["values"]), color=color, linewidth=2, label=name)
    ax3.set_title("Entropy Loss (exploración)")
    ax3.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    ax3.legend()

    # 4. Explained Variance
    ax4 = fig.add_subplot(gs[1, 2:4])
    for name, key, color in [("Loan", "loan", "tab:red"), ("Portfolio", "portfolio", "tab:blue")]:
        if key in all_data and "train/explained_variance" in all_data[key]:
            d = all_data[key]["train/explained_variance"]
            ax4.plot(d["steps"], _smooth(d["values"]), color=color, linewidth=2, label=name)
    ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax4.set_title("Explained Variance")
    ax4.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    ax4.legend()

    # 5. Reward from training logs (BusinessEvalCallback)
    ax5 = fig.add_subplot(gs[2, 0:2])
    reward_log = _parse_reward_from_logs()
    has_reward = False
    colors_rl = {"loan": "tab:red", "portfolio": "tab:blue"}
    for key in ["loan", "portfolio"]:
        if key in reward_log:
            rd = reward_log[key]
            ax5.plot(rd["steps"], _smooth(rd["values"], 0.7), linewidth=2,
                     label=f"Reward ({key})", color=colors_rl.get(key, "tab:gray"))
            has_reward = True
    if not has_reward:
        for key in ["general", "loan", "portfolio"]:
            if key in all_data and "rollout/ep_rew_mean" in all_data[key]:
                d = all_data[key]["rollout/ep_rew_mean"]
                ax5.plot(d["steps"], _smooth(d["values"], 0.7), linewidth=2, label=f"Reward ({key})")
                has_reward = True
    if has_reward:
        ax5.set_title("Reward medio por episodio (BusinessEvalCallback)")
        ax5.set_xlabel("Timesteps")
        ax5.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, "Reward no disponible",
                ha="center", va="center", transform=ax5.transAxes, fontsize=11, color="gray")
        ax5.set_title("Reward medio por episodio")

    # 6. Policy evolution (if checkpoints available)
    ax6 = fig.add_subplot(gs[2, 2:4])
    if ckpt_results:
        action_names = ["MANTENER", "REESTR.", "VENDER"]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        steps = [r["step"] for r in ckpt_results]
        probs = np.array([r["action_probs"] for r in ckpt_results])
        ax6.stackplot(steps, probs.T, labels=action_names, colors=colors, alpha=0.7)
        ax6.set_title("Evolución política (checkpoints)")
        ax6.set_ylim(0, 1)
        ax6.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax6.legend(loc="upper right", fontsize=8)
    else:
        ax6.text(0.5, 0.5, "Checkpoints no evaluados", ha="center", va="center",
                transform=ax6.transAxes, fontsize=11, color="gray")
        ax6.set_title("Evolución política")

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  [OK] {os.path.basename(save_path)}")


# ═══════════════════════════════════════════════════════════════
#  6. MODELO FINAL — INSPECCIÓN DETALLADA
# ═══════════════════════════════════════════════════════════════
def inspect_best_models():
    """Inspección de los modelos best (loan + portfolio)."""
    report_lines = []

    if not HAS_SB3:
        return ["SB3 no disponible — no se puede inspeccionar modelos."]

    for env_tag in ["loan", "portfolio"]:
        model_path = os.path.join(MODELS_DIR, f"best_model_{env_tag}.zip")
        if not os.path.exists(model_path):
            report_lines.append(f"\n{'='*60}")
            report_lines.append(f"  {env_tag.upper()} — NO ENCONTRADO: {model_path}")
            continue

        try:
            model = PPO.load(model_path, device="cpu")
            report_lines.append(f"\n{'='*60}")
            report_lines.append(f"  MODELO BEST — {env_tag.upper()}")
            report_lines.append(f"{'='*60}")
            report_lines.append(f"  Archivo:         {model_path}")
            report_lines.append(f"  Obs space:       {model.observation_space}")
            report_lines.append(f"  Action space:    {model.action_space}")
            report_lines.append(f"  Learning rate:   {model.learning_rate}")
            report_lines.append(f"  Gamma:           {model.gamma}")
            report_lines.append(f"  Batch size:      {model.batch_size}")
            report_lines.append(f"  N steps:         {model.n_steps}")
            report_lines.append(f"  N epochs:        {model.n_epochs}")
            report_lines.append(f"  Clip range:      {model.clip_range}")
            report_lines.append(f"  Ent coef:        {model.ent_coef}")
            report_lines.append(f"  VF coef:         {model.vf_coef}")
            report_lines.append(f"  Max grad norm:   {model.max_grad_norm}")

            # Network architecture
            report_lines.append(f"\n  Arquitectura de red:")
            total_params = 0
            for name, param in model.policy.state_dict().items():
                total_params += param.numel()
                shape_str = str(list(param.shape)).ljust(20)
                report_lines.append(f"    {name:40s} shape={shape_str} "
                                   f"mean={param.mean():.6f} std={param.std():.6f}")
            report_lines.append(f"  Total parámetros: {total_params:,}")

            # Policy output on zero observation
            obs_dim = model.observation_space.shape[0]
            dummy_obs = torch.zeros(1, obs_dim)
            with torch.no_grad():
                dist = model.policy.get_distribution(dummy_obs)
                probs = dist.distribution.probs.numpy().flatten()

            action_names = ["MANTENER", "REESTRUCTURAR", "VENDER"]
            report_lines.append(f"\n  Distribución de acciones (obs=0):")
            for i, (name, p) in enumerate(zip(action_names, probs)):
                bar = "█" * int(p * 40)
                report_lines.append(f"    {name:18s}: {p:.4f} {bar}")

            report_lines.append(f"  Entropía: {float(-np.sum(probs * np.log(probs + 1e-8))):.4f} nats "
                               f"(max={np.log(len(probs)):.4f})")

        except Exception as e:
            report_lines.append(f"\n  [ERROR] {env_tag}: {e}")

    return report_lines


# ═══════════════════════════════════════════════════════════════
#  7. DIAGNÓSTICO TEXTUAL
# ═══════════════════════════════════════════════════════════════
def generate_diagnostic(all_data: dict, ckpt_results: list) -> list:
    """Genera diagnóstico textual del aprendizaje."""
    lines = []
    lines.append("=" * 70)
    lines.append("  DIAGNÓSTICO DE APRENDIZAJE — PPO Optimizador Carteras NPL")
    lines.append("=" * 70)

    # Metadata
    meta_path = os.path.join(MODELS_DIR, "training_metadata_loan.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        lines.append(f"\n  📋 CONFIGURACIÓN DE ENTRENAMIENTO:")
        lines.append(f"     Pipeline:        {meta.get('pipeline', 'N/A')}")
        lines.append(f"     Timesteps:       {meta.get('timesteps_target', 'N/A'):,}")
        lines.append(f"     Seed:            {meta.get('seed', 'N/A')}")
        lines.append(f"     N envs:          {meta.get('n_envs', 'N/A')}")
        lines.append(f"     Batch size:      {meta.get('batch_size', 'N/A')}")
        lines.append(f"     Learning rate:   {meta.get('learning_rate', 'N/A')}")
        lines.append(f"     Gamma:           {meta.get('gamma', 'N/A')}")
        lines.append(f"     Clip range:      {meta.get('clip_range', 'N/A')}")
        lines.append(f"     Red:             {meta.get('policy_hidden', 'N/A')}")
        lines.append(f"     Activación:      {meta.get('activation_fn', 'N/A')}")
        lines.append(f"     Obs space:       {meta.get('obs_space_shape', 'N/A')}")
        lines.append(f"     Cartera:         {meta.get('portfolio_input', {}).get('n_rows', 'N/A')} préstamos")

    for run_name, data in all_data.items():
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  📊 RUN: {run_name.upper()}")
        lines.append(f"  {'─'*60}")

        # Loss analysis
        if "train/loss" in data:
            d = data["train/loss"]
            lines.append(f"\n  [LOSS]")
            lines.append(f"    Primera loss:      {d['values'][0]:.6f}")
            lines.append(f"    Última loss:       {d['values'][-1]:.6f}")
            lines.append(f"    Media:             {np.mean(d['values']):.6f}")
            lines.append(f"    Min:               {np.min(d['values']):.6f}")
            lines.append(f"    Max:               {np.max(d['values']):.6f}")
            change = (d["values"][-1] - d["values"][0]) / (abs(d["values"][0]) + 1e-8) * 100
            lines.append(f"    Cambio total:      {change:+.2f}%")
            if d["values"][-1] < d["values"][0]:
                lines.append(f"    ✅ LOSS decreció → aprendizaje positivo")
            else:
                lines.append(f"    ⚠️  LOSS no decreció → revisar hiperparámetros")

        # Value loss
        if "train/value_loss" in data:
            d = data["train/value_loss"]
            lines.append(f"\n  [VALUE FUNCTION]")
            lines.append(f"    Primera:           {d['values'][0]:.6f}")
            lines.append(f"    Última:            {d['values'][-1]:.6f}")
            if d["values"][-1] < d["values"][0]:
                lines.append(f"    ✅ Value function mejoró")
            else:
                lines.append(f"    ⚠️  Value function no mejoró nítidamente")

        # Explained variance
        if "train/explained_variance" in data:
            d = data["train/explained_variance"]
            ev_final = np.mean(d["values"][-20:]) if len(d["values"]) > 20 else np.mean(d["values"])
            lines.append(f"\n  [EXPLAINED VARIANCE]")
            lines.append(f"    Media últimos 20:  {ev_final:.4f}")
            if ev_final > 0.8:
                lines.append(f"    ✅ EXCELENTE — el value function predice bien los returns")
            elif ev_final > 0.5:
                lines.append(f"    ✅ BUENO — predicción de returns razonable")
            elif ev_final > 0:
                lines.append(f"    ⚠️  ACEPTABLE — el value function aporta algo")
            else:
                lines.append(f"    ❌ POBRE — el value function no mejora predicciones aleatorias")

        # KL divergence
        if "train/approx_kl" in data:
            d = data["train/approx_kl"]
            mean_kl = float(np.mean(d["values"]))
            max_kl = float(np.max(d["values"]))
            lines.append(f"\n  [KL DIVERGENCE]")
            lines.append(f"    Media:             {mean_kl:.6f}")
            lines.append(f"    Max:               {max_kl:.6f}")
            if max_kl < 0.02:
                lines.append(f"    ✅ Updates estables (KL < 0.02)")
            elif max_kl < 0.05:
                lines.append(f"    ⚠️  Algunos updates agresivos (KL pico {max_kl:.4f})")
            else:
                lines.append(f"    ❌ Updates muy agresivos — considerar reducir learning_rate")

        # Entropy
        if "train/entropy_loss" in data:
            d = data["train/entropy_loss"]
            initial_ent = d["values"][0]
            final_ent = d["values"][-1]
            lines.append(f"\n  [ENTROPÍA]")
            lines.append(f"    Inicial:           {initial_ent:.4f}")
            lines.append(f"    Final:             {final_ent:.4f}")
            # Max entropy for 3 discrete actions = -ln(1/3) ≈ -1.0986
            ratio = abs(final_ent) / 1.0986  # Proportion of max
            lines.append(f"    Ratio vs max:      {ratio:.2%}")
            if abs(final_ent) < 0.05:
                lines.append(f"    ⚠️  Entropía muy baja — política determinista ({run_name})")
                lines.append(f"       Esto puede ser convergencia o colapso — validar con eval score")
            elif abs(final_ent) > 0.8:
                lines.append(f"    ⚠️  Entropía alta — política aún exploratoria")
            else:
                lines.append(f"    ✅ Entropía moderada — balance exploración/explotación")

        # Clip fraction
        if "train/clip_fraction" in data:
            d = data["train/clip_fraction"]
            mean_clip = float(np.mean(d["values"]))
            lines.append(f"\n  [CLIP FRACTION]")
            lines.append(f"    Media:             {mean_clip:.4f}")
            lines.append(f"    Primera:           {d['values'][0]:.4f}")
            lines.append(f"    Última:            {d['values'][-1]:.4f}")
            if mean_clip < 0.15:
                lines.append(f"    ✅ Clipping moderado — updates estables")
            elif mean_clip < 0.30:
                lines.append(f"    ⚠️  Clipping elevado — updates algo agresivos")
            else:
                lines.append(f"    ❌ Clipping excesivo — considerar reducir learning_rate o clip_range")

        # FPS
        if "time/fps" in data:
            d = data["time/fps"]
            mean_fps = float(np.mean(d["values"]))
            lines.append(f"\n  [THROUGHPUT]")
            lines.append(f"    FPS medio:         {mean_fps:.0f}")

        # Reward (if available)
        if "rollout/ep_rew_mean" in data:
            d = data["rollout/ep_rew_mean"]
            lines.append(f"\n  [REWARD]")
            lines.append(f"    Primera:           {d['values'][0]:.2f}")
            lines.append(f"    Última:            {d['values'][-1]:.2f}")
            lines.append(f"    Mejor:             {np.max(d['values']):.2f}")
            lines.append(f"    Peor:              {np.min(d['values']):.2f}")
            trend = np.polyfit(range(len(d["values"])), d["values"], 1)
            if trend[0] > 0:
                lines.append(f"    ✅ Tendencia ASCENDENTE (pendiente={trend[0]:.2f})")
            else:
                lines.append(f"    ⚠️  Tendencia descendente (pendiente={trend[0]:.2f})")

    # Checkpoint analysis
    if ckpt_results:
        lines.append(f"\n  {'─'*60}")
        lines.append(f"  📦 ANÁLISIS DE CHECKPOINTS ({len(ckpt_results)} evaluados)")
        lines.append(f"  {'─'*60}")

        first = ckpt_results[0]
        last = ckpt_results[-1]
        lines.append(f"    Total parámetros:  {first['total_params']:,}")
        lines.append(f"    Primer ckpt:       step={first['step']:,}")
        lines.append(f"    Último ckpt:       step={last['step']:,}")

        # Entropy evolution
        ent_first = first["entropy"]
        ent_last = last["entropy"]
        lines.append(f"\n    Entropía política (obs=0):")
        lines.append(f"      Inicio:          {ent_first:.4f}")
        lines.append(f"      Final:           {ent_last:.4f}")
        ent_change = (ent_last - ent_first) / (ent_first + 1e-8) * 100
        lines.append(f"      Cambio:          {ent_change:+.1f}%")

        # Action preference
        lines.append(f"\n    Distribución de acciones final:")
        action_names = ["MANTENER", "REESTRUCTURAR", "VENDER"]
        for name, p in zip(action_names, last["action_probs"]):
            bar = "█" * int(p * 30)
            lines.append(f"      {name:18s}: {p:.4f} {bar}")

    # Conclusiones
    lines.append(f"\n{'='*70}")
    lines.append(f"  🎯 CONCLUSIONES")
    lines.append(f"{'='*70}")

    # Detect issues
    issues = []
    recommendations = []

    for run_name, data in all_data.items():
        if "train/entropy_loss" in data:
            final_ent = data["train/entropy_loss"]["values"][-1]
            if abs(final_ent) < 0.01:
                issues.append(f"  - {run_name}: Entropía colapsada ({final_ent:.6f}) → política degenerada")
                recommendations.append(f"  - Aumentar ent_coef (actual: 0.0) a 0.01-0.05")

        if "train/explained_variance" in data:
            ev = np.mean(data["train/explained_variance"]["values"][-20:])
            if ev < 0:
                issues.append(f"  - {run_name}: Explained variance negativa → value function no aprende")
                recommendations.append(f"  - Reducir vf_coef o aumentar learning_rate")

        if "train/clip_fraction" in data:
            mc = np.mean(data["train/clip_fraction"]["values"])
            if mc > 0.3:
                issues.append(f"  - {run_name}: Clip fraction alto ({mc:.2f}) → updates inestables")
                recommendations.append(f"  - Reducir learning_rate o clip_range")

    if issues:
        lines.append(f"\n  ⚠️  PROBLEMAS DETECTADOS:")
        for i in issues:
            lines.append(i)
    else:
        lines.append(f"\n  ✅ No se detectaron problemas críticos")

    if recommendations:
        lines.append(f"\n  💡 RECOMENDACIONES:")
        for r in list(set(recommendations)):
            lines.append(r)

    return lines


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  ANÁLISIS DE APRENDIZAJE PPO — Optimizador Carteras NPL")
    print("=" * 70)

    # 1. Encontrar runs de TensorBoard
    print("\n📂 Buscando runs de TensorBoard...")
    runs = find_tb_runs()
    for name, path in runs.items():
        print(f"  [{name}] → {path}")

    if not runs:
        print("  ❌ No se encontraron runs de TensorBoard!")
        return

    # 2. Cargar datos
    print("\n📊 Cargando datos de TensorBoard...")
    all_data = {}
    for name, path in runs.items():
        all_data[name] = load_tb_scalars(path)
        n_tags = len(all_data[name])
        n_points = sum(len(v["values"]) for v in all_data[name].values())
        print(f"  [{name}] {n_tags} métricas, {n_points} puntos totales")

    # 3. Generar gráficos por run
    # Parse reward from training logs
    print("\n💰 Extrayendo reward de logs de entrenamiento...")
    _reward_log = _parse_reward_from_logs()

    print("\n📈 Generando gráficos...")
    for name, data in all_data.items():
        label = name.upper()
        plot_loss_curves(data, label, os.path.join(REPORTS_DIR, f"01_loss_curves_{name}.png"))
        plot_pg_and_kl(data, label, os.path.join(REPORTS_DIR, f"02_pg_and_kl_{name}.png"))
        plot_entropy_and_clip(data, label, os.path.join(REPORTS_DIR, f"03_entropy_clip_{name}.png"))
        plot_explained_variance(data, label, os.path.join(REPORTS_DIR, f"04_explained_variance_{name}.png"))
        plot_fps(data, label, os.path.join(REPORTS_DIR, f"05_fps_{name}.png"))
        plot_reward_curve(data, label, os.path.join(REPORTS_DIR, f"08_reward_{name}.png"), reward_log=_reward_log)

    # 4. Comparación loan vs portfolio
    print("\n🔄 Comparación Loan vs Portfolio...")
    plot_comparison(all_data, os.path.join(REPORTS_DIR, "09_portfolio_comparison.png"))

    # 5. Evaluación de Checkpoints
    print("\n📦 Evaluando checkpoints...")
    ckpt_results = evaluate_checkpoints()
    if ckpt_results:
        plot_checkpoint_evolution(ckpt_results, os.path.join(REPORTS_DIR, "06_checkpoint_evaluation.png"))
        plot_policy_distribution_evolution(ckpt_results, os.path.join(REPORTS_DIR, "07_policy_distribution.png"))
        print(f"  Evaluados {len(ckpt_results)} checkpoints")

    # 6. Dashboard resumen
    print("\n🎨 Generando dashboard resumen...")
    plot_summary_dashboard(all_data, ckpt_results, os.path.join(REPORTS_DIR, "10_summary_dashboard.png"))

    # 7. Inspección de modelos best
    print("\n🔍 Inspeccionando modelos best...")
    model_report = inspect_best_models()

    # 8. Diagnóstico textual
    print("\n📝 Generando diagnóstico...")
    diagnostic = generate_diagnostic(all_data, ckpt_results)

    # 9. Escribir reporte
    report_path = os.path.join(REPORTS_DIR, "learning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DE ANÁLISIS DE APRENDIZAJE PPO\n")
        f.write(f"Fecha: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for line in diagnostic:
            f.write(line + "\n")

        f.write("\n\n")
        for line in model_report:
            f.write(line + "\n")

    print(f"\n✅ Reporte guardado en: {report_path}")
    print(f"📁 Gráficos en: {REPORTS_DIR}")

    # 10. Imprimir diagnóstico en consola
    print("\n")
    for line in diagnostic:
        print(line)
    print()
    for line in model_report:
        print(line)

    print(f"\n{'='*70}")
    print(f"  🏁 ANÁLISIS COMPLETADO")
    print(f"  📁 {len(os.listdir(REPORTS_DIR))} archivos en {REPORTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
