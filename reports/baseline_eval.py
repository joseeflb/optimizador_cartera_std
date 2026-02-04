# reports/baseline_eval.py
# Evaluación RL vs Baseline con columnas REALES del audit CSV del repo.
# Mide guardrails: fire-sale, sell-blocked, reestructuras no viables/missing inputs.
# Nota: No puede comparar EVA por acción (contrafactual) si no existe en el CSV; sí suma EVA_gain/PnL/capital_release del resultado elegido.

from __future__ import annotations
import os, glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Baseline por postura (solo preferencias, no guardrails)
# -----------------------------
@dataclass(frozen=True)
class BaselineParams:
    inertia_score_margin: float      # prudencial: no cambiar por mejoras marginales de score
    prefer_sell_margin: float        # desinversión: si SELL casi empata con el mejor, vende


BASELINE_BY_PROFILE: Dict[str, BaselineParams] = {
    "PRUDENCIAL":   BaselineParams(inertia_score_margin=0.15, prefer_sell_margin=0.00),
    "BALANCEADO":   BaselineParams(inertia_score_margin=0.05, prefer_sell_margin=0.00),
    "DESINVERSION": BaselineParams(inertia_score_margin=0.00, prefer_sell_margin=0.10),
}


# -----------------------------
# Utilidades (columnas)
# -----------------------------
def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def _latest_audit_csvs() -> Dict[str, str]:
    paths = glob.glob("reports/**/decisiones_audit_*.csv", recursive=True)
    if not paths:
        raise FileNotFoundError("No encuentro reports/**/decisiones_audit_*.csv. Ejecuta inferencia y reintenta.")

    by_profile: Dict[str, Tuple[float, str]] = {}
    for p in paths:
        base = os.path.basename(p).lower()
        prof = None
        if "prud" in base:
            prof = "PRUDENCIAL"
        elif "balan" in base:
            prof = "BALANCEADO"
        elif "desinv" in base or "desin" in base:
            prof = "DESINVERSION"
        if prof is None:
            continue
        mtime = os.path.getmtime(p)
        if (prof not in by_profile) or (mtime > by_profile[prof][0]):
            by_profile[prof] = (mtime, p)

    if len(by_profile) < 1:
        latest = max(paths, key=lambda x: os.path.getmtime(x))
        return {"UNKNOWN": latest}

    return {k: v[1] for k, v in by_profile.items()}


def _detect_action_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, [
        "Accion_final", "Accion_final", "Accion", "decision_final", "final_action", "action_final",
        "coordinator_action", "action"
    ])


def _detect_price_ratio_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, [
        "Price_to_EAD", "price_to_ead", "price_ratio_ead", "price_ratio",
        "price_ratio_book", "Price_to_Book"
    ])


def _detect_fire_sale_threshold_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, [
        "fire_sale_threshold", "fire_sale_threshold_book", "fire_sale_threshold_ead"
    ])


def _detect_fire_sale_flags(df: pd.DataFrame) -> List[str]:
    # Solo aceptamos flags explícitos de “trigger”.
    # NO usar FireSale_Triggers (texto) ni fire_sale (puede ser “estado/auxiliar”).
    cols = list(df.columns)
    c = _pick_col(cols, ["FireSale_Triggered"])
    return [c] if c else []



def _detect_sell_blocked_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, ["Sell_Blocked"])


def _detect_restruct_viable_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, ["restruct_viable"])


def _detect_missing_viability_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, ["Missing_Viability_Inputs"])


def _detect_pti_post_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, ["PTI_post", "pti_post"])


def _detect_dscr_post_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _pick_col(cols, ["DSCR_post", "dscr_post"])


def _detect_scores(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "HOLD": _pick_col(cols, ["Score_keep", "score_keep", "Score_hold", "score_hold"]),
        "RESTRUCTURE": _pick_col(cols, ["Score_restruct", "score_restruct"]),
        "SELL": _pick_col(cols, ["Score_sell", "score_sell"]),
    }


def _detect_outcome_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "EVA_gain": _pick_col(cols, ["EVA_gain", "ΔEVA"]),
        "pnl_realized": _pick_col(cols, ["pnl_realized", "pnl"]),
        "capital_release_realized": _pick_col(cols, ["capital_release_realized", "capital_liberado", "capital_release_cf"]),
        "cured": _pick_col(cols, ["cured"]),
    }


# -----------------------------
# Normalización acción
# -----------------------------
def _normalize_action(x) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    if isinstance(x, (int, float)):
        if int(x) == 0:
            return "HOLD"
        if int(x) == 1:
            return "RESTRUCTURE"
        if int(x) == 2:
            return "SELL"
    s = str(x).strip().upper()
    if "MANT" in s or "HOLD" in s or "KEEP" in s:
        return "HOLD"
    if "REE" in s or "REST" in s:
        return "RESTRUCTURE"
    if "VEND" in s or "SELL" in s:
        return "SELL"
    return s


# -----------------------------
# Guardrails
# -----------------------------
def _boolish(series: pd.Series) -> pd.Series:
    # convierte valores variados a bool (robusto)
    s = series.copy()
    if s.dtype == bool:
        return s.fillna(False)
    s2 = s.astype(str).str.strip().str.lower()
    return s2.isin(["true", "1", "yes", "y", "si", "sí"])


def fire_sale_triggered_row(row: pd.Series, col_ratio: Optional[str], col_thr: Optional[str], fire_flag_cols: List[str]) -> bool:
    # 1) flags explícitos
    # Caso especial: FireSale_Triggers suele venir como lista/str (p.ej. "['Price_to_EAD<thr']")
    for c in fire_flag_cols:
        if c.lower() == "firesale_triggers":
            v = row.get(c, None)
            if v is None:
                continue
            s = str(v).strip()
            if s and s not in ["[]", "nan", "None", "none"]:
                return True


    # 2) ratio < threshold (si existe)
    if col_ratio and col_thr:
        try:
            r = float(row[col_ratio])
            t = float(row[col_thr])
            if pd.notna(r) and pd.notna(t):
                return r < t
        except Exception:
            pass

    return False


def sell_allowed_row(row: pd.Series, col_ratio: Optional[str], col_thr: Optional[str], fire_flag_cols: List[str], col_sell_blocked: Optional[str]) -> bool:
    blocked = False
    if col_sell_blocked:
        try:
            blocked = bool(_boolish(pd.Series([row[col_sell_blocked]])).iloc[0])
        except Exception:
            blocked = False

    if blocked:
        return False

    if fire_sale_triggered_row(row, col_ratio, col_thr, fire_flag_cols):
        return False

    return True


def restruct_allowed_row(row: pd.Series, col_restruct_viable: Optional[str], col_missing_viab: Optional[str]) -> bool:
    # Si faltan inputs, por prudencia NO reestructuramos
    if col_missing_viab:
        try:
            miss = bool(_boolish(pd.Series([row[col_missing_viab]])).iloc[0])
            if miss:
                return False
        except Exception:
            pass

    if col_restruct_viable:
        try:
            return bool(_boolish(pd.Series([row[col_restruct_viable]])).iloc[0])
        except Exception:
            return False

    # si no existe indicador, no asumimos viable
    return False


# -----------------------------
# Baseline policy (determinista)
# -----------------------------
def baseline_decision_row(row: pd.Series, prof: str, colmap: dict) -> str:
    params = BASELINE_BY_PROFILE.get(prof, BASELINE_BY_PROFILE["BALANCEADO"])

    col_ratio = colmap["price_ratio"]
    col_thr = colmap["fire_thr"]
    fire_cols = colmap["fire_cols"]
    col_sell_blocked = colmap["sell_blocked"]
    col_restruct_viable = colmap["restruct_viable"]
    col_missing_viab = colmap["missing_viab"]
    scores = colmap["scores"]

    allow_hold = True
    allow_sell = sell_allowed_row(row, col_ratio, col_thr, fire_cols, col_sell_blocked)
    allow_restruct = restruct_allowed_row(row, col_restruct_viable, col_missing_viab)

    allowed = {"HOLD"}
    if allow_restruct:
        allowed.add("RESTRUCTURE")
    if allow_sell:
        allowed.add("SELL")

    # Preferencias por score si existen
    score_vals = {}
    for a, c in scores.items():
        if not c:
            continue
        try:
            score_vals[a] = float(row[c])
        except Exception:
            pass

    # Si no hay scores -> heurística simple por postura
    if not score_vals:
        if prof == "DESINVERSION":
            return "SELL" if "SELL" in allowed else ("RESTRUCTURE" if "RESTRUCTURE" in allowed else "HOLD")
        if prof == "BALANCEADO":
            return "RESTRUCTURE" if "RESTRUCTURE" in allowed else ("SELL" if "SELL" in allowed else "HOLD")
        # prudencial
        return "RESTRUCTURE" if "RESTRUCTURE" in allowed else "HOLD"

    # elegir mejor score dentro de allowed
    # fallback si falta score de alguna acción: muy bajo
    def s(a: str) -> float:
        return score_vals.get(a, float("-1e18"))

    best = max(list(allowed), key=s)

    # inercia prudencial/balanceado: si HOLD casi empata, HOLD
    if "HOLD" in allowed and best != "HOLD":
        if (s(best) - s("HOLD")) < params.inertia_score_margin:
            best = "HOLD"

    # desinversión: si SELL está permitido y casi empata, vende
    if prof == "DESINVERSION" and "SELL" in allowed:
        if (s("SELL") >= s(best) - params.prefer_sell_margin):
            best = "SELL"

    return best


# -----------------------------
# Métricas
# -----------------------------
def _action_mix(s: pd.Series) -> pd.Series:
    return s.value_counts(dropna=False).reindex(["HOLD", "RESTRUCTURE", "SELL", "UNKNOWN"], fill_value=0)

def _rate(condition: pd.Series) -> float:
    if condition is None or len(condition) == 0:
        return float("nan")
    return float(condition.mean())

def evaluate_one(df: pd.DataFrame, prof: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    action_col = _detect_action_col(df)

    col_ratio = _detect_price_ratio_col(df)
    col_thr = _detect_fire_sale_threshold_col(df)
    fire_cols = _detect_fire_sale_flags(df)
    col_sell_blocked = _detect_sell_blocked_col(df)

    col_restruct_viable = _detect_restruct_viable_col(df)
    col_missing_viab = _detect_missing_viability_col(df)

    col_pti = _detect_pti_post_col(df)
    col_dscr = _detect_dscr_post_col(df)

    scores = _detect_scores(df)
    outcomes = _detect_outcome_cols(df)

    colmap = {
        "action": action_col,
        "price_ratio": col_ratio,
        "fire_thr": col_thr,
        "fire_cols": fire_cols,
        "sell_blocked": col_sell_blocked,
        "restruct_viable": col_restruct_viable,
        "missing_viab": col_missing_viab,
        "pti": col_pti,
        "dscr": col_dscr,
        "scores": scores,
        "outcomes": outcomes,
    }

    # RL action
    if not action_col:
        df["action_rl"] = "UNKNOWN"
    else:
        df["action_rl"] = df[action_col].apply(_normalize_action)

    # Baseline action
    df["action_baseline"] = df.apply(lambda r: baseline_decision_row(r, prof, colmap), axis=1)

    # Flags RL
    df["fire_sale_triggered"] = df.apply(lambda r: fire_sale_triggered_row(r, col_ratio, col_thr, fire_cols), axis=1)
    df["sell_blocked_flag"] = False
    if col_sell_blocked:
        df["sell_blocked_flag"] = _boolish(df[col_sell_blocked])

    df["restruct_allowed_flag"] = False
    if col_restruct_viable:
        df["restruct_allowed_flag"] = _boolish(df[col_restruct_viable])
    if col_missing_viab:
        miss = _boolish(df[col_missing_viab])
        df["restruct_allowed_flag"] = df["restruct_allowed_flag"] & (~miss)

    # Violaciones (RL)
    df["is_fire_sale_rl"] = (df["action_rl"] == "SELL") & (df["fire_sale_triggered"])
    df["is_sell_blocked_rl"] = (df["action_rl"] == "SELL") & (df["sell_blocked_flag"])
    df["is_restruct_not_viable_rl"] = (df["action_rl"] == "RESTRUCTURE") & (~df["restruct_allowed_flag"])

    # Violaciones (Baseline)
    df["is_fire_sale_base"] = (df["action_baseline"] == "SELL") & (df["fire_sale_triggered"])
    df["is_sell_blocked_base"] = (df["action_baseline"] == "SELL") & (df["sell_blocked_flag"])
    df["is_restruct_not_viable_base"] = (df["action_baseline"] == "RESTRUCTURE") & (~df["restruct_allowed_flag"])

    # Outcomes RL (solo del resultado ejecutado, no contrafactual)
    for k, c in outcomes.items():
        if c:
            df[k] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[k] = pd.NA

    # Summary
    summary = pd.DataFrame({
        "profile": [prof],
        "n_loans": [len(df)],

        "mix_rl_hold": [_action_mix(df["action_rl"])["HOLD"]],
        "mix_rl_restructure": [_action_mix(df["action_rl"])["RESTRUCTURE"]],
        "mix_rl_sell": [_action_mix(df["action_rl"])["SELL"]],

        "mix_base_hold": [_action_mix(df["action_baseline"])["HOLD"]],
        "mix_base_restructure": [_action_mix(df["action_baseline"])["RESTRUCTURE"]],
        "mix_base_sell": [_action_mix(df["action_baseline"])["SELL"]],

        "fire_sale_rate_rl": [_rate(df["is_fire_sale_rl"])],
        "fire_sale_rate_base": [_rate(df["is_fire_sale_base"])],
        "sell_blocked_rate_rl": [_rate(df["is_sell_blocked_rl"])],
        "sell_blocked_rate_base": [_rate(df["is_sell_blocked_base"])],
        "restruct_not_viable_rate_rl": [_rate(df["is_restruct_not_viable_rl"])],
        "restruct_not_viable_rate_base": [_rate(df["is_restruct_not_viable_base"])],

        "eva_gain_sum_rl": [pd.to_numeric(df["EVA_gain"], errors="coerce").sum(min_count=1)],
        "pnl_realized_sum_rl": [pd.to_numeric(df["pnl_realized"], errors="coerce").sum(min_count=1)],
        "capital_release_sum_rl": [pd.to_numeric(df["capital_release_realized"], errors="coerce").sum(min_count=1)],
        "cure_rate_rl": [_rate(_boolish(df[outcomes["cured"]]) if outcomes["cured"] else pd.Series([pd.NA]*len(df)))],

        "pct_loans_action_diff": [float((df["action_rl"] != df["action_baseline"]).mean())],

        "used_action_col": [str(action_col)],
        "used_price_ratio_col": [str(col_ratio)],
        "used_fire_thr_col": [str(col_thr)],
        "used_fire_cols": [str(fire_cols)],
        "used_sell_blocked_col": [str(col_sell_blocked)],
        "used_restruct_viable_col": [str(col_restruct_viable)],
        "used_missing_viability_col": [str(col_missing_viab)],
        "used_score_cols": [str(scores)],
        "used_outcome_cols": [str(outcomes)],
    })

    # Detail export (más informativo)
    keep_cols = [
        "action_rl", "action_baseline",
        "fire_sale_triggered", "sell_blocked_flag", "restruct_allowed_flag",
        "is_fire_sale_rl", "is_sell_blocked_rl", "is_restruct_not_viable_rl",
        "is_fire_sale_base", "is_sell_blocked_base", "is_restruct_not_viable_base",
        "EVA_gain", "pnl_realized", "capital_release_realized",
    ]
    for c in [action_col, col_ratio, col_thr, col_sell_blocked, col_restruct_viable, col_missing_viab, col_pti, col_dscr,
              scores.get("HOLD"), scores.get("RESTRUCTURE"), scores.get("SELL"),
              outcomes.get("pnl_realized"), outcomes.get("EVA_gain"), outcomes.get("capital_release_realized"), outcomes.get("cured")]:
        if c and c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    detail = df[keep_cols].copy()
    return summary, detail


def main():
    csvs = _latest_audit_csvs()
    out_dir = os.path.join("reports", "summary")
    os.makedirs(out_dir, exist_ok=True)

    all_sum = []
    writer_path = os.path.join(out_dir, "baseline_vs_rl.xlsx")

    with pd.ExcelWriter(writer_path, engine="openpyxl") as writer:
        for prof, path in csvs.items():
            df = pd.read_csv(path)
            s, d = evaluate_one(df, prof)
            all_sum.append(s)

            sheet_s = f"{prof[:10]}_summary"
            sheet_d = f"{prof[:10]}_detail"
            s.to_excel(writer, index=False, sheet_name=sheet_s)
            d.to_excel(writer, index=False, sheet_name=sheet_d)

        pd.concat(all_sum, ignore_index=True).to_excel(writer, index=False, sheet_name="ALL_SUMMARY")

    print("OK - generado:", writer_path)
    print("CSV usados:")
    for prof, path in csvs.items():
        print(f"  - {prof}: {path}")


if __name__ == "__main__":
    main()
