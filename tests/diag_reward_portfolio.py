import numpy as np
from env.portfolio_env import PortfolioEnv

def main():
    env = PortfolioEnv(seed=42, scenario="baseline", top_k=5)
    n_steps = 300

    rewards = []
    by_action = {}

    obs, info = env.reset()
    for _ in range(n_steps):
        a = int(env.action_space.sample())
        obs, r, terminated, truncated, info = env.step(a)

        r = float(r)
        rewards.append(r)

        info = info or {}
        rb = info.get("reward_breakdown", {})
        if not isinstance(rb, dict):
            rb = {}

        by_action.setdefault(a, {"r": [], "eva_gain": [], "rel_cap": [], "risk": [], "pnl": []})

        by_action[a]["r"].append(r)
        by_action[a]["eva_gain"].append(float(rb.get("eva_gain", 0.0)))
        by_action[a]["rel_cap"].append(float(rb.get("rel_cap", 0.0)))
        by_action[a]["risk"].append(float(rb.get("risk_proxy_EL_lifetime", 0.0)))
        by_action[a]["pnl"].append(float(rb.get("pnl_scaled", 0.0)))

        if terminated or truncated:
            obs, info = env.reset()

    rewards = np.array(rewards, dtype=float)

    print("=== PORTFOLIO REWARD GLOBAL ===")
    print(f"n={len(rewards)} mean={rewards.mean():.4f} std={rewards.std():.4f} min={rewards.min():.4f} max={rewards.max():.4f}")

    print("\n=== PORTFOLIO POR ACCION ===")
    for a in sorted(by_action.keys()):
        arr = np.array(by_action[a]["r"], dtype=float)
        print(f"\nAction {a}: n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")

        def _ms(x):
            x = np.array(x, dtype=float)
            return float(x.mean()), float(x.std())

        eva_m, eva_s = _ms(by_action[a]["eva_gain"])
        cap_m, cap_s = _ms(by_action[a]["rel_cap"])
        risk_m, risk_s = _ms(by_action[a]["risk"])
        pnl_m, pnl_s = _ms(by_action[a]["pnl"])

        print(f"  eva_gain mean/std: {eva_m:.4f} / {eva_s:.4f}")
        print(f"  rel_cap  mean/std: {cap_m:.4f} / {cap_s:.4f}")
        print(f"  risk     mean/std: {risk_m:.4f} / {risk_s:.4f}")
        print(f"  pnl      mean/std: {pnl_m:.4f} / {pnl_s:.4f}")

if __name__ == "__main__":
    main()
