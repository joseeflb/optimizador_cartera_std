# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd

CANON_COLS = [
    "loan_id",
    "segment",
    "EAD",
    "RW",

    # Coordinator (bank-ready)
    "Accion_micro",
    "Accion_macro",
    "Accion_final",
    "Macro_Selected",
    "Convergencia_Caso",
    "Reason_Code",
    "Decision_Governance",

    # Fire-sale / pricing (consistentes)
    "Fire_Sale",
    "Price_to_EAD",
    "fire_sale_threshold",

    # Viabilidad
    "PTI_pre",
    "PTI_post",
    "DSCR_pre",
    "DSCR_post",

    # Venta
    "precio_optimo",
    "pnl",
    "p5",
    "p50",
    "p95",

    # Capital / realized (para reporting homogéneo)
    "capital_release_cf",
    "capital_release_realized",
    "pnl_realized",
]

def _first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ---------------------------
    # 1) Normalización / aliases
    # ---------------------------

    # loan_id
    if "loan_id" not in out.columns:
        id_col = _first_existing(out, ["Loan_ID","LOAN_ID","LoanID","loanId","id","ID"])
        out["loan_id"] = out[id_col].astype(str) if id_col else ""

    # segment
    if "segment" not in out.columns:
        seg_col = _first_existing(out, ["segmento_banco","segment_raw","segmento","seg"])
        out["segment"] = out[seg_col] if seg_col else np.nan

    # RW
    if "RW" not in out.columns:
        rw_col = _first_existing(out, ["rw","Rw"])
        out["RW"] = pd.to_numeric(out[rw_col], errors="coerce") if rw_col else np.nan

    # Acciones (micro / final): si vienes de MICRO puro, normalmente tienes "Accion"
    if "Accion_micro" not in out.columns:
        if "Accion" in out.columns:
            out["Accion_micro"] = out["Accion"]
        else:
            out["Accion_micro"] = np.nan

    if "Accion_final" not in out.columns:
        if "Accion" in out.columns:
            out["Accion_final"] = out["Accion"]
        else:
            out["Accion_final"] = out.get("decision_final", np.nan)

    # Accion_macro: si no hay macro, explícitamente NaN (o NO_ASIGNADO si prefieres)
    if "Accion_macro" not in out.columns:
        out["Accion_macro"] = np.nan

    if "Macro_Selected" not in out.columns:
        out["Macro_Selected"] = False

    if "Convergencia_Caso" not in out.columns:
        out["Convergencia_Caso"] = np.nan

    # Fire-sale: unificamos "fire_sale" (micro) con "Fire_Sale" (coordinator)
    if "Fire_Sale" not in out.columns:
        if "fire_sale" in out.columns:
            out["Fire_Sale"] = out["fire_sale"].astype(bool)
        else:
            out["Fire_Sale"] = np.nan

    # precio_optimo/pnl aliases habituales
    if "precio_optimo" not in out.columns and "price" in out.columns:
        out["precio_optimo"] = out["price"]
    if "pnl" not in out.columns and "PnL" in out.columns:
        out["pnl"] = out["PnL"]

    # capital_release_realized: si tienes capital_liberado (micro/coordinator), mapéalo
    if "capital_release_realized" not in out.columns:
        if "capital_liberado" in out.columns:
            out["capital_release_realized"] = out["capital_liberado"]
        elif "capital_release" in out.columns:
            out["capital_release_realized"] = out["capital_release"]
        else:
            out["capital_release_realized"] = np.nan

    # ---------------------------
    # 2) Derivadas
    # ---------------------------
    # Price_to_EAD si no existe o está vacío: precio_optimo / EAD
    if "Price_to_EAD" not in out.columns:
        out["Price_to_EAD"] = np.nan

    try:
        ead = pd.to_numeric(out.get("EAD", np.nan), errors="coerce")
        px = pd.to_numeric(out.get("precio_optimo", np.nan), errors="coerce")
        mask = out["Price_to_EAD"].isna() & ead.notna() & (ead > 0) & px.notna()
        out.loc[mask, "Price_to_EAD"] = (px[mask] / ead[mask]).astype(float)
    except Exception:
        pass

    # ---------------------------
    # 3) Asegurar CANON_COLS
    # ---------------------------
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = np.nan

    # Orden: canónicas primero, resto al final
    rest = [c for c in out.columns if c not in CANON_COLS]
    return out[CANON_COLS + rest]
