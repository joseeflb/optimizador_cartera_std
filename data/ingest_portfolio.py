# -*- coding: utf-8 -*-
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
        
    # Feature Engineering for Model Compatibility (if needed)
    # e.g., mapping rating string to number if not done elsewhere
    # But usually policy_inference handles that, or expects mapped values.
    
    # 5. Enforce Schema (canonical output)
    final_df = enforce_schema(df)
    logger.info(f"Ingestion complete. Final shape: {final_df.shape}")
    
    return final_df

if __name__ == "__main__":
    # Test execution
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        df_out = load_and_validate_portfolio(path)
        print(df_out.head())
    else:
        print("Usage: python ingest_portfolio.py <path_to_excel_or_csv>")
