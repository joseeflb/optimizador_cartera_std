# -*- coding: utf-8 -*-
# ============================================================
# reports/backtesting_light.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: What-If sobre decisiones congeladas: aplica shocks a posteriori sin re-entrenamiento.
# ============================================================

import os
import sys
import argparse
import yaml
import logging
import pandas as pd
import numpy as np

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config as cfg
# Import stress utilities to apply shocks
from engines.stress_engine import apply_shocks

logger = logging.getLogger("backtesting_light")
logging.basicConfig(level=logging.INFO)

def run_backtesting_light(tag: str, stress_scenarios_path: str):
    """
    Takes the FINAL DECISIONS from a previous run (tag) and re-evaluates
    them under stress scenarios WITHOUT changing the decision.
    """
    logger.info(f"Starting Backtesting Light for tag='{tag}'")
    
    # 1. Locate Base Runs
    # Try multiple patterns for where the "run" output might be.
    # a) reports/runs/<tag>_...
    # b) reports/stress_<tag>_...
    
    runs_dir = os.path.join(ROOT_DIR, "reports", "runs")
    reports_dir = os.path.join(ROOT_DIR, "reports")
    
    candidates = []
    
    # helper
    def scan_dir(d, pattern):
        if not os.path.exists(d): return []
        return [os.path.join(d, x) for x in os.listdir(d) 
                if pattern in x and os.path.isdir(os.path.join(d, x))]

    candidates.extend(scan_dir(runs_dir, tag))
    candidates.extend(scan_dir(reports_dir, f"stress_{tag}"))
    
    if not candidates:
        logger.error(f"No run folder found for tag: {tag}")
        return
    
    # Sort by name (usually timestamp) and pick latest
    candidates.sort(reverse=True)
    target_run_dir = candidates[0]
    logger.info(f"Using source directory: {target_run_dir}")
    
    # 2. Load Stress Scenarios
    if not os.path.exists(stress_scenarios_path):
        logger.error(f"Scenarios file not found: {stress_scenarios_path}")
        return

    with open(stress_scenarios_path, 'r') as f:
         scenarios_cfg = yaml.safe_load(f)
    scenarios = scenarios_cfg.get("scenarios", scenarios_cfg)

    results = []
    postures = ["prudencial", "balanceado", "desinversion"]
    
    # Financial Constants
    hurdle = float(cfg.CONFIG.regulacion.hurdle_rate)
    cost_fund = 0.006 

    for posture in postures:
        # Navigate to posture folder
        # Standard: <run>/<posture>/
        # Stress: <run>/baseline/<posture>/ (because "stress" usually runs a baseline scenario first)
        
        # We prefer the "un-stressed" or "baseline" decisions to stress-test them.
        # If the run folder itself is a stress run, we should look for the 'baseline' subfolder.
        
        p_dir = os.path.join(target_run_dir, posture)
        if not os.path.exists(p_dir):
            p_dir = os.path.join(target_run_dir, "baseline", posture)
        
        if not os.path.exists(p_dir):
            logger.warning(f"Posture dir not found: {p_dir}")
            continue
            
        # Find decision file
        # NOTE: stress_engine output might be nested one level deeper: <p_dir>/<timestamp>_tag/...
        # We search recursively or just check if there's a directory inside.
        
        decision_file = None
        # Helper to find file in directory recursively (depth 1)
        candidates_files = []
        for root, dirs, files in os.walk(p_dir):
            for f in files:
                # Accept both naming conventions
                if (f.startswith("decisiones_explicadas") or f.startswith("decisiones_finales")) and f.endswith(".xlsx"):
                    candidates_files.append(os.path.join(root, f))
            # Limit depth to avoid deep scan if unwanted
            if root != p_dir and os.path.dirname(root) != p_dir:
                 pass

        if candidates_files:
            # Pick the most recent if multiple
            candidates_files.sort(key=os.path.getmtime, reverse=True)
            decision_file = candidates_files[0]
        
        if not decision_file:
            logger.warning(f"No decisions file in {p_dir}")
            continue
            
        logger.info(f"Loading decisions for {posture} from {decision_file}")
        try:
            df_decisions = pd.read_excel(decision_file)
        except Exception as e:
            logger.error(f"Failed to load {decision_file}: {e}")
            continue

        # Keep only MANTENER/REESTRUCTURAR loans (sold loans are already off-book)
        if "Accion_final" in df_decisions.columns:
            df_book = df_decisions[df_decisions["Accion_final"] != "VENDER"].copy()
            logger.info(f"  Posture {posture}: {len(df_book)}/{len(df_decisions)} loans remaining on book after decisions")
        else:
            df_book = df_decisions.copy()

        # 3. Iterate Scenarios
        for sc_name, sc_data in scenarios.items():
            shocks = sc_data.get("shocks", {})
            
            # Apply shocks to a copy
            try:
                # IMPORTANT: Ingestion/Validation might fail if we shock too hard.
                # But here we just want to see the impact.
                df_shocked = apply_shocks(df_book, sc_name, shocks)
            except ValueError as e:
                logger.warning(f"Skipping {sc_name} validation error: {e}")
                continue
            
            # 4. Re-calculate Metrics
            # RWA
            if "RW" in df_shocked.columns and "EAD" in df_shocked.columns:
                df_shocked["RWA_stressed"] = df_shocked["EAD"] * df_shocked["RW"]
            else:
                 df_shocked["RWA_stressed"] = 0.0

            # EL
            if "PD" in df_shocked.columns and "LGD" in df_shocked.columns and "EAD" in df_shocked.columns:
                 df_shocked["EL_stressed"] = df_shocked["PD"] * df_shocked["LGD"] * df_shocked["EAD"]
            else:
                 df_shocked["EL_stressed"] = 0.0
                 
            # NI & EVA
            if "interest_rate" in df_shocked.columns:
                rate = df_shocked["interest_rate"]
            else:
                rate = 0.05
            
            ni = df_shocked["EAD"] * (rate - cost_fund) - df_shocked["EL_stressed"]
            df_shocked["EVA_stressed"] = ni - (hurdle * df_shocked["RWA_stressed"])
            
            # Haircut on Sales
            if "bid_haircut_mult" in shocks and "precio_optimo" in df_shocked.columns:
                 # Reducing price by haircut multiplier (e.g. 1.3x haircut or price/1.3)
                 # Engine logic usually implies simpler shocks. Let's start with division.
                 mult = float(shocks["bid_haircut_mult"])
                 if mult > 0:
                    df_shocked.loc[df_shocked["Accion_final"]=="VENDER", "precio_optimo"] /= mult
            
            total_eva = df_shocked["EVA_stressed"].sum()
            total_rwa = df_shocked["RWA_stressed"].sum()
            total_el  = df_shocked["EL_stressed"].sum()
            
            results.append({
                "tag": tag,
                "posture": posture,
                "scenario": sc_name,
                "EVA_stressed": total_eva,
                "RWA_stressed": total_rwa,
                "EL_stressed": total_el,
            })
            
    # 5. Save Report
    if results:
        df_res = pd.DataFrame(results)
        out_csv = os.path.join(ROOT_DIR, "reports", f"backtesting_light_{tag}.csv")
        out_md = os.path.join(ROOT_DIR, "reports", f"backtesting_light_{tag}.md")
        
        df_res.to_csv(out_csv, index=False)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(f"# Backtesting Light Report: {tag}\n\n")
            try:
                f.write(df_res.to_markdown())
            except ImportError:
                f.write(df_res.to_string(index=False))
            
        logger.info(f"Backtesting Complete. Saved to {out_csv}")
        # print(df_res.to_markdown())
    else:
        logger.error("No backtesting results produced.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stress-scenarios", default="configs/stress_scenarios.yaml")
    args = parser.parse_args()
    
    run_backtesting_light(args.tag, args.stress_scenarios)
