# -*- coding: utf-8 -*-
# ============================================================
# data/ingest_portfolio.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Ingesta y validación de carteras reales (Excel/CSV) con mapping a esquema canónico vía mappings/real_portfolio_mapping.yaml.
# ============================================================
import os
import yaml
import pandas as pd
import numpy as np
import logging

import config as cfg

try:
    from reports.schema import enforce_schema
except ImportError:
    enforce_schema = lambda x: x

logger = logging.getLogger("ingest_portfolio")

# Path to mapping file
MAPPING_FILE = os.path.join(os.path.dirname(__file__), "mappings", "real_portfolio_mapping.yaml")

def load_mapping_config():
    if not os.path.exists(MAPPING_FILE):
        logger.warning(f"Mapping file not found at {MAPPING_FILE}. Using defaults.")
        return {}
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_and_validate_portfolio(file_path: str) -> pd.DataFrame:
    """
    Ingesta robusta de datos reales.
    1. Carga Excel/CSV.
    2. Aplica mapeo de columnas (YAML).
    3. Rellena valores por defecto.
    4. Valida tipos y rangos.
    5. Normaliza salida con reports.schema.enforce_schema.
    """
    logger.info(f"Ingesting portfolio from: {file_path}")
    
    # 1. Load File
    if str(file_path).endswith(".csv"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
             df = pd.read_csv(file_path, encoding="latin1") # Fallback muy común en banca legacy
    else:
        df = pd.read_excel(file_path) # Engine 'openpyxl' can be inferred or specified if needed
    
    logger.info(f"Loaded raw data with shape: {df.shape}")
    
    # 2. Apply Mapping
    config = load_mapping_config()
    col_map = config.get("column_map", {})
    defaults = config.get("defaults", {})
    validation = config.get("validation", {})
    
    # Create a mapping dictionary for rename
    rename_dict = {}
    for canon, potential_names in col_map.items():
        # Find the first match in df.columns
        for name in potential_names:
            # Case insensitive check
            # Create a map of lower_case -> real_name for the dataframe
            df_cols_lower = {c.lower(): c for c in df.columns}
            if name.lower() in df_cols_lower:
                real_name = df_cols_lower[name.lower()]
                rename_dict[real_name] = canon
                break
    
    if rename_dict:
        logger.info(f"Renaming columns: {rename_dict}")
        df = df.rename(columns=rename_dict)
    
    # 3. Apply Defaults / Ensure Schema
    target_cols = col_map.keys()
    for col in target_cols:
        if col not in df.columns:
            if col in defaults:
                val = defaults[col]
                logger.debug(f"Column '{col}' missing. Filling with default: {val}")
                df[col] = val
            else:
                logger.warning(f"Column '{col}' missing and no default provided. Setting to NaN.")
                df[col] = np.nan
                
    # 4. Validation & Typing
    # Ensure numeric types for numeric columns
    numeric_cols = ["EAD", "PD", "LGD", "RW", "interest_rate", "remaining_term", "collateral_value", "DPD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Apply constraints (Strict vs Permissive)
    constraints = validation.get("constraints", {})
    allow_clip = getattr(cfg, "ALLOW_CLIP_OUT_OF_RANGE", False)

    for col, bounds in constraints.items():
        if col in df.columns:
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            
            if min_val is not None:
                below_mask = df[col] < min_val
                if below_mask.any():
                    msg = f"Column '{col}' has {below_mask.sum()} values below min {min_val}."
                    if allow_clip:
                        logger.warning(f"{msg} Clipping (permissive mode).")
                        df.loc[below_mask, col] = min_val
                    else:
                        raise ValueError(f"STRICT INGESTION ERROR: {msg} (Set ALLOW_CLIP_OUT_OF_RANGE=True to bypass)")
                    
            if max_val is not None:
                above_mask = df[col] > max_val
                if above_mask.any():
                    msg = f"Column '{col}' has {above_mask.sum()} values above max {max_val}."
                    if allow_clip:
                        logger.warning(f"{msg} Clipping (permissive mode).")
                        df.loc[above_mask, col] = max_val
                    else:
                         raise ValueError(f"STRICT INGESTION ERROR: {msg} (Set ALLOW_CLIP_OUT_OF_RANGE=True to bypass)")

    # Required columns check
    required = validation.get("required_columns", [])
    missing_required = [c for c in required if c not in df.columns or df[c].isna().all()]
    if missing_required:
        raise ValueError(f"CRITICAL: Missing required columns (or all empty): {missing_required}")

        
    # Drop rows without loan_id or EAD if critical
    if "loan_id" in df.columns:
        df = df.dropna(subset=["loan_id"])
        
    # Feature Engineering for Model Compatibility
    # Derive fields required by the RL agents if not present in input data
    df = _derive_model_features(df)

    # 5. Enforce Schema (canonical output)
    final_df = enforce_schema(df)
    logger.info(f"Ingestion complete. Final shape: {final_df.shape}")
    
    return final_df


def _derive_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive financial features required by LoanEnv/PortfolioEnv
    when they are missing from the input portfolio.
    This enables usage with ANY bank portfolio, not just synthetic data.
    """
    CFG = cfg.CONFIG
    npl = CFG.npl
    reg = CFG.regulacion
    hurdle = float(reg.hurdle_rate)
    cap_ratio = float(reg.required_total_capital_ratio())

    # --- loan_id: ensure exists ---
    if "loan_id" not in df.columns:
        df["loan_id"] = [f"L{i:06d}" for i in range(len(df))]

    # --- EAD: must exist (validated above) ---
    df["EAD"] = pd.to_numeric(df["EAD"], errors="coerce").fillna(0.0)

    # --- PD: clamp to NPL conventions ---
    if "PD" not in df.columns:
        df["PD"] = npl.pd_default_floor
    df["PD"] = df["PD"].clip(npl.pd_default_floor, npl.pd_default_cap)

    # --- LGD: clamp ---
    if "LGD" not in df.columns:
        df["LGD"] = npl.lgd_default_floor
    df["LGD"] = df["LGD"].clip(npl.lgd_default_floor, npl.lgd_default_cap)

    # --- RW: resolve from Basel III STD if missing ---
    if "RW" not in df.columns:
        df["RW"] = 1.50  # default: unsecured NPL

    # --- DPD: enforce NPL floor ---
    if "DPD" not in df.columns:
        df["DPD"] = npl.dpd_floor
    df["DPD"] = df["DPD"].clip(npl.dpd_floor, npl.dpd_cap)

    # --- RWA (Risk Weighted Assets) ---
    if "RWA" not in df.columns:
        df["RWA"] = df["EAD"] * df["RW"]

    # --- EL (Expected Loss) ---
    if "EL" not in df.columns:
        horizon_years = float(getattr(CFG.sensibilidad_reestructura, "horizon_months", 24)) / 12.0
        df["EL"] = df["EAD"] * df["PD"] * df["LGD"] * horizon_years

    # --- NI (Net Income proxy) ---
    if "NI" not in df.columns:
        int_rate = df.get("interest_rate", pd.Series(0.05, index=df.index))
        int_rate = pd.to_numeric(int_rate, errors="coerce").fillna(0.05)
        df["NI"] = df["EAD"] * int_rate - df["EL"]

    # --- EVA (Economic Value Added) ---
    if "EVA" not in df.columns:
        capital_locked = df["RWA"] * cap_ratio
        cost_of_capital = capital_locked * hurdle
        df["EVA"] = df["NI"] - cost_of_capital

    # --- RORWA (Return on RWA) ---
    if "RORWA" not in df.columns:
        df["RORWA"] = np.where(
            df["RWA"].abs() > 1.0,
            df["NI"] / df["RWA"],
            0.0
        )

    # --- RONA (Return on Net Assets) ---
    if "RONA" not in df.columns:
        df["RONA"] = df["RORWA"]

    # --- rating_num ---
    if "rating_num" not in df.columns:
        rating_map = {"AAA": 7, "AA": 6, "A": 5, "BBB": 4, "BB": 3, "B": 2, "CCC": 1}
        if "rating" in df.columns:
            df["rating_num"] = df["rating"].map(rating_map).fillna(1).astype(int)
        else:
            df["rating_num"] = 1  # CCC default for NPL

    # --- segmento_id ---
    if "segmento_id" not in df.columns:
        seg_map = {
            "large_corporate": 0, "corporate": 1, "midcap": 2, "sme": 3,
            "project_finance": 4, "mortgage": 5, "retail": 6, "consumer": 7,
            "sovereign": 8, "bank": 9, "leasing": 10,
        }
        if "segment" in df.columns:
            df["segmento_id"] = df["segment"].astype(str).str.lower().str.strip().map(seg_map).fillna(1).astype(int)
        else:
            df["segmento_id"] = 1

    # --- secured ---
    if "secured" not in df.columns:
        if "collateral_value" in df.columns:
            df["secured"] = (pd.to_numeric(df["collateral_value"], errors="coerce").fillna(0) > 0).astype(int)
        elif "segment" in df.columns:
            df["secured"] = df["segment"].astype(str).str.lower().str.contains("mortgage").astype(int)
        else:
            df["secured"] = 0

    # --- coverage_rate & provisions ---
    if "coverage_rate" not in df.columns:
        df["coverage_rate"] = df["LGD"].clip(0.3, 0.95)
    if "provisions" not in df.columns:
        df["provisions"] = df["EAD"] * df["coverage_rate"]
    if "book_value" not in df.columns:
        df["book_value"] = df["EAD"] - df["provisions"]
    if "book_value" in df.columns:
        df["book_value"] = df["book_value"].clip(lower=0.0)

    # --- PTI / DSCR ---
    if "monthly_payment" not in df.columns:
        remaining = pd.to_numeric(df.get("maturity_months_remaining",
                     df.get("remaining_term", pd.Series(12, index=df.index))), errors="coerce").fillna(12).clip(lower=1)
        df["maturity_months_remaining"] = remaining
        int_rate = pd.to_numeric(df.get("interest_rate", pd.Series(0.05, index=df.index)), errors="coerce").fillna(0.05)
        monthly_r = int_rate / 12.0
        df["monthly_payment"] = np.where(
            monthly_r > 0,
            df["EAD"] * monthly_r / (1 - (1 + monthly_r) ** (-remaining)),
            df["EAD"] / remaining,
        )

    if "monthly_income" not in df.columns:
        df["monthly_income"] = df["monthly_payment"] * 3.5  # typical PTI ~28%
    if "monthly_cfo" not in df.columns:
        df["monthly_cfo"] = df["monthly_income"] * 0.85

    if "PTI_pre" not in df.columns:
        inc = pd.to_numeric(df["monthly_income"], errors="coerce").fillna(1.0).clip(lower=1.0)
        df["PTI_pre"] = df["monthly_payment"] / inc
    if "DSCR_pre" not in df.columns:
        pmt = pd.to_numeric(df["monthly_payment"], errors="coerce").fillna(1.0).clip(lower=1.0)
        df["DSCR_pre"] = pd.to_numeric(df["monthly_cfo"], errors="coerce").fillna(0) / pmt

    n_derived = sum(1 for c in ("EVA", "RORWA", "RWA", "EL", "NI", "rating_num", "segmento_id")
                    if c in df.columns)
    logger.info(f"[DERIVE] {n_derived} model features available. Portfolio ready for inference.")

    return df

if __name__ == "__main__":
    # Test execution
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        df_out = load_and_validate_portfolio(path)
        print(df_out.head())
    else:
        print("Usage: python ingest_portfolio.py <path_to_excel_or_csv>")
