import os, sys
import pandas as pd
import numpy as np

# --- asegurar paths ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../data
ROOT_DIR = os.path.dirname(THIS_DIR)                          # repo root
sys.path.insert(0, ROOT_DIR)                                  # para importar config.py
sys.path.insert(0, THIS_DIR)                                  # para importar generate_portfolio.py

import config as cfg
from generate_portfolio import generate_portfolio


def run(n=2000, seed=42):
    df = generate_portfolio(n=n, seed=seed)

    # --- 1) % NI < 0
    ni_neg = (df["NI"] < 0).mean()

    # --- 2) % EVA < 0
    eva_neg = (df["EVA"] < 0).mean()

    # --- 3) Distribución PD*LGD y comparación con rate
    pd_lgd = df["PD"] * df["LGD"]
    cost_fund = 0.006  # el que usas en generate_portfolio.py
    # NI/EAD = rate - PD*LGD - cost_fund
    ni_margin = df["rate"] - pd_lgd - cost_fund

    # EVA = NI - EAD*RW*hurdle   =>  EVA/EAD = (NI/EAD) - RW*hurdle
    hurdle = float(cfg.CONFIG.regulacion.hurdle_rate)
    eva_margin = ni_margin - df["RW"] * hurdle

    # --- 4) Confirmar segmento_id numérico (para el env)
    seg_is_numeric = pd.api.types.is_numeric_dtype(df["segmento_id"])

    # --- 5) Confirmar distribución RW (¿1.50 en el 100%?)
    rw_dist = df["RW"].value_counts(dropna=False, normalize=True).sort_index()

    # --- Prints
    print("=" * 80)
    print("SANITY CHECKS — GENERADOR NPL")
    print(f"n={len(df)} | seed={seed}")
    print("-" * 80)
    print(f"% NI < 0:  {ni_neg:.3f}")
    print(f"% EVA < 0: {eva_neg:.3f}")
    print("-" * 80)

    def q(x):  # quantiles rápidos
        return x.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).to_dict()

    print("PD*LGD quantiles:", q(pd_lgd))
    print("rate quantiles:  ", q(df["rate"]))
    print("NI/EAD margin = rate - PD*LGD - cost_fund quantiles:", q(ni_margin))
    print("EVA/EAD margin = NI/EAD - RW*hurdle quantiles:      ", q(eva_margin))

    print("-" * 80)
    print(f"segmento_id numeric? {seg_is_numeric} | dtype={df['segmento_id'].dtype}")
    print("FEATURE_COLUMNS dtypes:")
    for c in cfg.FEATURE_COLUMNS:
        if c in df.columns:
            print(f"  - {c}: {df[c].dtype}")
        else:
            print(f"  - {c}: MISSING")

    print("-" * 80)
    print("RW distribution (share):")
    for k, v in rw_dist.items():
        print(f"  RW={k}: {v:.3f}")

    # Heurística de “sell-all regime” (alertas)
    print("-" * 80)
    if eva_neg >= 0.90:
        print("[ALERTA] EVA negativa en >=90%: el dataset induce régimen 'value-destroyer'.")
    if ni_neg >= 0.70:
        print("[ALERTA] NI negativa en >=70%: casi todo loan pierde dinero con tu proxy NI.")
    if not seg_is_numeric:
        print("[ALERTA] segmento_id NO es numérico: puede degradar el aprendizaje/normalización.")
    if (rw_dist.index == 1.50).all() and rw_dist.iloc[0] > 0.95:
        print("[ALERTA] RW casi constante (~1.50): poca señal de capital/steering por RW.")

    print("=" * 80)
    return df

if __name__ == "__main__":
    run()
