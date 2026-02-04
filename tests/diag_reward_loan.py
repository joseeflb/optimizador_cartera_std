import numpy as np
from env.loan_env import LoanEnv

def main():
    # Pool mínimo (puedes meter 20-50 préstamos si quieres)
    loan_pool = [
        {"segment":"CORPORATE","rating":"BBB","EAD":1_000_000.0,"PD":0.80,"LGD":0.85,"DPD":720.0,"RW":1.5},
        {"segment":"MORTGAGE","rating":"BB","EAD":150_000.0,"PD":0.65,"LGD":0.60,"DPD":210.0,"RW":1.0},
        {"segment":"CONSUMER","rating":"B","EAD":50_000.0,"PD":0.70,"LGD":0.75,"DPD":180.0,"RW":1.5},
    ]

    env = LoanEnv(loan_pool=loan_pool, seed=42)
    n_steps = 500

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

        by_action.setdefault(a, {"r": [], "eva_gain": [], "rel_cap": [], "risk": [], "pnl": [], "fire": []})

        by_action[a]["r"].append(r)
        by_action[a]["eva_gain"].append(float(rb.get("eva_gain", 0.0)))
        by_action[a]["rel_cap"].append(float(rb.get("rel_cap", 0.0)))
        by_action[a]["risk"].append(float(rb.get("risk_proxy_EL_lifetime", 0.0)))
        by_action[a]["pnl"].append(float(rb.get("pnl_scaled", 0.0)))
        by_action[a]["fire"].append(float(rb.get("fire_sale_flag", False)))

        if terminated or truncated:
            obs, info = env.reset()

    rewards = np.array(rewards, dtype=float)

    print("=== LOANENV REWARD GLOBAL ===")
    print(f"n={len(rewards)} mean={rewards.mean():.4f} std={rewards.std():.4f} min={rewards.min():.4f} max={rewards.max():.4f}")

    print("\n=== LOANENV POR ACCION ===")
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
        fire_rate = float(np.mean(np.array(by_action[a]["fire"], dtype=float) > 0.0)) if len(by_action[a]["fire"]) else 0.0

        print(f"  eva_gain mean/std: {eva_m:.4f} / {eva_s:.4f}")
        print(f"  rel_cap  mean/std: {cap_m:.4f} / {cap_s:.4f}")
        print(f"  risk     mean/std: {risk_m:.4f} / {risk_s:.4f}")
        print(f"  pnl      mean/std: {pnl_m:.4f} / {pnl_s:.4f}")
        print(f"  fire_sale rate:    {fire_rate:.2%}")

if __name__ == "__main__":
    main()
