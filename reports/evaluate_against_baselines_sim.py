# -*- coding: utf-8 -*-
# ============================================================
# reports/evaluate_against_baselines_sim.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Versión simulada de la evaluación contra baselines: reproduce decisiones en entorno controlado.
# ============================================================

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import glob
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config as cfg
# Import Baseline Policies
from baselines.baseline_policies import BaselinePrudencialRules, BaselineDesinversionRules, ACTION_KEEP, ACTION_SELL, ACTION_RESTRUCT

# Import Environment for Simulation
from env.loan_env import LoanEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_portfolio_kpis_from_df(df: pd.DataFrame, action_col: str = "Accion_final") -> Dict[str, float]:
    """Calcula métricas agregadas del portafolio desde DF de resultados RL."""
    if df.empty:
        return {}

    col_eva = next((c for c in ["EVA_post", "EVA_final", "EVA"] if c in df.columns), None)
    col_rwa = next((c for c in ["RWA_post", "RWA_final", "RWA"] if c in df.columns), None)
    col_cap_release = next((c for c in ["capital_release", "Capital_Release", "capital_liberado"] if c in df.columns), None)
    
    total_eva = df[col_eva].sum() if col_eva else 0.0
    total_rwa = df[col_rwa].sum() if col_rwa else 0.0
    total_cap_release = df[col_cap_release].sum() if col_cap_release else 0.0

    counts = {"sell_count": 0, "restruct_count": 0, "keep_count": 0}
    if action_col in df.columns:
        val_counts = df[action_col].value_counts().to_dict()
        for val, count in val_counts.items():
            s_val = str(val).upper()
            if "VEN" in s_val or "SELL" in s_val or s_val == "1":
                counts["sell_count"] += count
            elif "REEST" in s_val or "RESTRUCT" in s_val or s_val == "2":
                counts["restruct_count"] += count
            else:
                counts["keep_count"] += count

    return {
        "final_total_eva": total_eva,
        "final_total_rwa": total_rwa,
        "final_capital_release": total_cap_release,
        **counts
    }

import re

def get_rl_results(tag: str, select: str = "latest") -> List[Dict[str, Any]]:
    """Finds RL result folders, filters duplicates (latest per posture), and parses metrics from JSON."""
    results = []
    reports_dir = os.path.join(ROOT_DIR, "reports")
    # Search for folders matching tag
    pattern = os.path.join(reports_dir, f"*coordinated_inference*{tag}*")
    all_folders = glob.glob(pattern)
    
    # 1. Group by Posture
    posture_map = {} # posture -> list of (timestamp, folder_path)
    
    for folder in all_folders:
        if not os.path.isdir(folder):
            continue
            
        # Detect posture
        lower_name = os.path.basename(folder).lower()
        posture = "unknown"
        if "prudencial" in lower_name: posture = "prudencial"
        elif "balanceado" in lower_name: posture = "balanceado"
        elif "desinversion" in lower_name: posture = "desinversion"
        
        # Extract timestamp: ..._YYYYMMDD_HHMMSS...
        ts_match = re.search(r"(\d{8}_\d{6})", lower_name)
        timestamp = ts_match.group(1) if ts_match else "00000000_000000"
        
        if posture not in posture_map:
            posture_map[posture] = []
        posture_map[posture].append((timestamp, folder))
    
    # 2. Select Latest per Posture
    selected_folders = []
    for posture, items in posture_map.items():
        if select == "latest":
            # Sort by timestamp descending
            items.sort(key=lambda x: x[0], reverse=True)
            if items:
                best = items[0]
                selected_folders.append((posture, best[1]))
                logger.info(f"Selected latest run for {posture}: {os.path.basename(best[1])}")
        else:
            # All
            for ts, f in items:
                selected_folders.append((posture, f))

    # 3. Parse KPI
    for posture, folder in selected_folders:
        # Prefer JSON
        json_path = os.path.join(folder, f"portfolio_kpis_{posture}.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                # Use final_state
                fs = data.get("final_state", {})
                ac = fs.get("action_counts", {})
                
                # Count overrides
                override_path = os.path.join(folder, f"overrides_log_{posture}.csv")
                ov_count = 0
                ov_guard = 0
                ov_macro = 0
                if os.path.exists(override_path):
                    try:
                        df_ov = pd.read_csv(override_path)
                        ov_count = len(df_ov)
                        if not df_ov.empty and "level" in df_ov.columns:
                             ov_guard = len(df_ov[df_ov["level"].astype(str).str.contains("GUARDRAIL", case=False, na=False)])
                             ov_macro = len(df_ov[df_ov["level"].astype(str).str.contains("MACRO", case=False, na=False)])
                    except: pass

                res = {
                     "method": f"RL_{posture.upper()}",
                     "final_total_eva": fs.get("total_eva", 0.0),
                     "final_total_rwa": fs.get("total_rwa", 0.0),
                     "final_capital_release": fs.get("total_capital_release", 0.0),
                     "sell_count": ac.get("VENDER", ac.get("SELL", 0)),
                     "restruct_count": ac.get("REESTRUCTURAR", ac.get("RESTRUCT", 0)),
                     "keep_count": ac.get("MANTENER", ac.get("KEEP", 0)),
                     "override_count_total": ov_count,
                     "override_guardrail": ov_guard,
                     "override_macro": ov_macro,
                     "violations": 0
                }
                results.append(res)
            except Exception as e:
                logger.error(f"Error parsing JSON for {posture}: {e}")
        else:
            logger.warning(f"JSON not found for {posture} in {folder}")
            
    return results

def run_simulation(policy, df_portfolio: pd.DataFrame) -> Dict[str, Any]:
    """Runs a full simulation using LoanEnv for the Baseline Policy."""
    
    # 1. Prepare Environment
    # Ensure dataframe has required columns for LoanEnv
    # LoanEnv is resilient but needs basic fields.
    # We pass records directly.
    loan_pool = df_portfolio.to_dict('records')
    env = LoanEnv(loan_pool=loan_pool)
    
    total_eva = 0.0
    total_rwa = 0.0
    total_cap = 0.0
    
    actions_taken = {0: 0, 1: 0, 2: 0}
    
    # 2. Iterate Loans
    # LoanEnv internal index handling is: reset() takes _next_loan_from_pool.
    # pool_index increments on access.
    # We loop exactly len(pool) times.
    
    n_loans = len(loan_pool)
    logger.info(f"Simulating {n_loans} loans for policy {policy.name}...")
    
    for i in range(n_loans):
        # Reset to load NEXT loan
        obs, info = env.reset()
        
        # Current state
        row = pd.Series(env.state)
        
        # Policy Prediction
        action = policy.predict(row)
        actions_taken[action] = actions_taken.get(action, 0) + 1
        
        # Environment Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log Metrics
        metrics = info.get("metrics", {})
        
        # Fix 3: Get RWA/EVA correctly (Handle missing RWA in metrics)
        rwa_val = metrics.get("RWA", 0.0)
        # If metric missing but not terminated (i.e. Keep/Restruct), try env.state
        if "RWA" not in metrics and not terminated:
             rwa_val = float(env.state.get("RWA", 0.0))
        
        # EVA Logic:
        cap_released_step = 0.0
        
        if action == ACTION_SELL:
            pnl = metrics.get("pnl_venta", 0.0)
            # Try to get cap from metrics or calculate fallback
            cap = metrics.get("capital_liberado", 0.0)
            
            # Fallback if metrics missing but RWA implies capital release
            if cap == 0.0:
                 # Use prev RWA from state if metrics RWA is cleared (0)
                 # If Sell, rwa_val becomes 0 below, so use env.state["RWA"] or current rw_val before zeroing
                 state_rwa = float(env.state.get("RWA", 0.0))
                 if state_rwa > 0:
                      cap = state_rwa * 0.08

            cap_released_step = cap
            
            # Heuristic from RL approx ~7.5x CapRel + PnL
            eva_val = pnl + (cap_released_step * 7.5)
            rwa_val = 0.0 # Clear RWA on sell
        else:
             eva_val = metrics.get("EVA", 0.0)
             # Fallback to state if missing in metrics
             if "EVA" not in metrics:
                  eva_val = float(env.state.get("EVA", 0.0))
                  
             # Capital Release on Restructure if RWA reduces
             cap_released_step = metrics.get("capital_liberado", 0.0)
        
        total_eva += eva_val
        total_rwa += rwa_val
        total_cap += cap_released_step
        
        if i < n_loans - 1:
            env.reset()
            
    return {
        "method": policy.name,
        "final_total_eva": total_eva,
        "final_total_rwa": total_rwa,
        "final_capital_release": total_cap,
        "sell_count": actions_taken[ACTION_SELL],
        "restruct_count": actions_taken[ACTION_RESTRUCT],
        "keep_count": actions_taken[ACTION_KEEP],
        "override_count_total": 0,
        "override_guardrail": 0,
        "override_macro": 0,
        "violations": 0
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL Policies against Deterministic Baselines (Simulated)")
    parser.add_argument("--tag", required=True, help="Tag of the RL runs to compare")
    parser.add_argument("--select", default="latest", choices=["latest", "all"], help="Selection mode for RL runs")
    parser.add_argument("--portfolio", default="data/portfolio_synth.xlsx", help="Path to portfolio file")
    parser.add_argument("--out", default="reports/evaluation_pc6.csv", help="Output path")
    parser.add_argument("--report", default="reports/evaluation_report.md", help="Markdown report path")
    
    args = parser.parse_args()
    logger.info(f"[U1F680] Starting Evaluation (Simulated). Tag: {args.tag}, Mode: {args.select}")
    
    # 1. Get RL Results
    rl_metrics = get_rl_results(args.tag, select=args.select)
    if not rl_metrics:
        logger.warning("[WARN][UFE0F] No RL results found.")
    else:
        logger.info(f"Found {len(rl_metrics)} RL runs.")

    # 2. Load Portfolio
    df_portfolio = None
    if rl_metrics:
         # Find one enriched file from the found folders
         # Re-scan to find path for portfolio load
         reports_dir = os.path.join(ROOT_DIR, "reports")
         pattern = os.path.join(reports_dir, f"*coordinated_inference*{args.tag}*")
         all_folders = glob.glob(pattern)
         for f in all_folders:
             files = glob.glob(os.path.join(f, "decisiones_finales_*.xlsx"))
             if files:
                 logger.info(f"Loading portfolio from {files[0]}")
                 df_portfolio = pd.read_excel(files[0])
                 break
    
    if df_portfolio is None:
        logger.info(f"Loading raw portfolio from {args.portfolio}")
        df_portfolio = pd.read_excel(args.portfolio)
    
    # 3. Baselines
    # ... (rest same) -> Need to fix rest of main to use args.report and fillna
    baselines = [BaselinePrudencialRules(), BaselineDesinversionRules()]
    baseline_metrics = []
    
    for b in baselines:
        logger.info(f"Running Simulation for: {b.name}")
        try:
            res = run_simulation(b, df_portfolio)
            baseline_metrics.append(res)
        except Exception as e:
            logger.error(f"Simulation failed for {b.name}: {e}", exc_info=True)
            
    # 4. Compare
    all_results = rl_metrics + baseline_metrics
    df_res = pd.DataFrame(all_results)
    
    # Fix NaNs
    df_res = df_res.fillna(0)
    
    # Save
    df_res.to_csv(args.out, index=False)
    
    # Report
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# [U1F4CA] Evaluation Report: RL vs Baselines (Simulated)\n\n")
        f.write(f"**Tag:** `{args.tag}`\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Summary Table\n\n")
        try:
            f.write(df_res.to_markdown(index=False))
        except ImportError:
            f.write(df_res.to_string(index=False))
        f.write("\n\n## Key Findings\n")
        
        if not df_res.empty and "final_total_eva" in df_res.columns:
            best_eva = df_res.loc[df_res["final_total_eva"].idxmax()]
            f.write(f"- **Best EVA:** {best_eva['method']} ({best_eva['final_total_eva']:,.0f}€)\n")
        
    logger.info(f"[OK] Report saved to {args.report}")
    logger.info(f"Summary:\n{df_res.to_string(index=False)}")

if __name__ == "__main__":
    main()
