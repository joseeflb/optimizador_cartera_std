from __future__ import annotations

import argparse
import os
import glob
import pandas as pd


ACTION_COL_CANDIDATES = [
    "Decision_Final", "decision_final", "final_decision", "Final_Decision",
    "action_final", "Action_Final", "Decision", "decision", "accion_final", "Accion_Final",
]

REASON_COL_CANDIDATES = [
    "Reason_Code", "reason_code", "RC", "rc", "Override_Reason", "override_reason",
]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _normalize_action_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.upper().str.strip()
    # Normaliza variantes típicas
    x = x.replace({
        "0": "MANTENER",
        "1": "REESTRUCTURAR",
        "2": "VENDER",
        "KEEP": "MANTENER",
        "HOLD": "MANTENER",
        "RESTRUCTURE": "REESTRUCTURAR",
        "SELL": "VENDER",
    })
    return x


def build_kpi_for_file(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    action_col = _pick_col(df, ACTION_COL_CANDIDATES)
    if action_col is None:
        raise ValueError(
            f"[KPI] No encuentro columna de acción en {os.path.basename(csv_path)}.\n"
            f"Columnas disponibles: {list(df.columns)}\n"
            f"Esperadas (alguna): {ACTION_COL_CANDIDATES}"
        )

    actions = _normalize_action_series(df[action_col])
    n = len(df)

    # Acciones: conteo y %
    counts = actions.value_counts(dropna=False).to_dict()
    pct = {k: (v / n) if n else 0.0 for k, v in counts.items()}

    # Métricas opcionales (si existen)
    def num(col: str) -> pd.Series | None:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return None

    def pick_num(*names: str) -> pd.Series | None:
        for name in names:
            s = num(name)
            if s is not None:
                return s
        return None

    dscr_post = pick_num("DSCR_post", "DSCR_Post", "dscr_post")
    pti_post = pick_num("PTI_post", "PTI_Post", "pti_post")

    eva_pre = pick_num("EVA_pre", "EVA_Pre")
    eva_post = pick_num("EVA_post", "EVA_Post")
    rwa_pre = pick_num("RWA_pre", "RWA_Pre")
    rwa_post = pick_num("RWA_post", "RWA_Post")

    sell_pnl = pick_num("Sell_PnL", "sell_pnl", "PnL_Sell")
    sell_price = pick_num("Sell_Price", "sell_price", "Price_Sell")

    reason_col = _pick_col(df, REASON_COL_CANDIDATES)
    restruct_mask = actions.eq("REESTRUCTURAR")

    out = {
        "file": os.path.basename(csv_path),
        "n_loans": n,
        "pct_mantener": float(pct.get("MANTENER", 0.0)),
        "pct_reestructurar": float(pct.get("REESTRUCTURAR", 0.0)),
        "pct_vender": float(pct.get("VENDER", 0.0)),
        "n_mantener": int(counts.get("MANTENER", 0)),
        "n_reestructurar": int(counts.get("REESTRUCTURAR", 0)),
        "n_vender": int(counts.get("VENDER", 0)),
    }

    # DSCR/PTI (solo sobre reestructurados)
    if dscr_post is not None:
        out["dscr_post_mean_restruct"] = float(dscr_post[restruct_mask].mean())
        out["dscr_post_p10_restruct"] = float(dscr_post[restruct_mask].quantile(0.10))
        out["violaciones_dscr_lt_1_10_restruct"] = int((dscr_post[restruct_mask] < 1.10).sum())
    else:
        out["dscr_post_mean_restruct"] = None
        out["dscr_post_p10_restruct"] = None
        out["violaciones_dscr_lt_1_10_restruct"] = None

    if pti_post is not None:
        out["pti_post_mean_restruct"] = float(pti_post[restruct_mask].mean())
        out["pti_post_nan_share_restruct"] = float(pti_post[restruct_mask].isna().mean())
    else:
        out["pti_post_mean_restruct"] = None
        out["pti_post_nan_share_restruct"] = None

    # EVA/RWA agregados si están
    if eva_pre is not None and eva_post is not None:
        out["eva_total_pre"] = float(eva_pre.sum())
        out["eva_total_post"] = float(eva_post.sum())
        out["eva_total_delta"] = float((eva_post - eva_pre).sum())
    else:
        out["eva_total_pre"] = None
        out["eva_total_post"] = None
        out["eva_total_delta"] = None

    if rwa_pre is not None and rwa_post is not None:
        out["rwa_total_pre"] = float(rwa_pre.sum())
        out["rwa_total_post"] = float(rwa_post.sum())
        out["rwa_total_delta"] = float((rwa_post - rwa_pre).sum())
    else:
        out["rwa_total_pre"] = None
        out["rwa_total_post"] = None
        out["rwa_total_delta"] = None

    # Venta agregada si está
    out["sell_pnl_total"] = float(sell_pnl.sum()) if sell_pnl is not None else None
    out["sell_price_total"] = float(sell_price.sum()) if sell_price is not None else None

    # RC breakdown si existe
    if reason_col is not None:
        rc_counts = df[reason_col].astype(str).value_counts().head(20)
        out["_rc_top"] = rc_counts.to_dict()
    else:
        out["_rc_top"] = {}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliverable-dir", required=True, help="Ruta a ..._DELIVERABLE")
    args = ap.parse_args()

    ddir = args.deliverable_dir
    pattern = os.path.join(ddir, "decisiones_audit_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"[KPI] No encuentro ficheros con patrón: {pattern}")

    rows = []
    rc_rows = []
    for f in files:
        kpi = build_kpi_for_file(f)
        posture = os.path.basename(f).replace("decisiones_audit_", "").replace(".csv", "")
        kpi["postura"] = posture
        rows.append({k: v for k, v in kpi.items() if not k.startswith("_")})
        for rc, cnt in kpi.get("_rc_top", {}).items():
            rc_rows.append({"postura": posture, "rc": rc, "count": cnt})

    df_kpi = pd.DataFrame(rows).sort_values("postura")
    df_rc = pd.DataFrame(rc_rows).sort_values(["postura", "count"], ascending=[True, False])

    out_xlsx = os.path.join(ddir, "kpis_posturas.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df_kpi.to_excel(xw, index=False, sheet_name="KPI_por_postura")
        df_rc.to_excel(xw, index=False, sheet_name="RC_top20")

    print("[KPI] Generado:", out_xlsx)
    print(df_kpi[[
        "postura", "pct_mantener", "pct_reestructurar", "pct_vender",
        "violaciones_dscr_lt_1_10_restruct", "pti_post_nan_share_restruct"
    ]])


if __name__ == "__main__":
    main()
