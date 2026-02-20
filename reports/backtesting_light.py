# -*- coding: utf-8 -*-
# ============================================
# reports/backtesting_light.py
# "What-If" Analysis on Frozen Decisions
# ============================================

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
    
    Logic:
    1. Find 'final_decisions_{tag}_{posture}.xlsx' in reports/runs/*
    2. For each scenario in stress_scenarios_path:
       a. Apply shocks to the loan features (PD, LGD, RW)
       b. Re-calculate Financial KPIs (EVA, RWA, Capital Release) assuming SAME action.
    """
    
    # 1. Locate Base Runs
    runs_dir = os.path.join(ROOT_DIR, "reports", "runs") # or cfg.base_output_dir
    # We look for the folder matching the tag. It usually has a timestamp prefix.
    # We scan for folders containing the tag.
    
    candidate_dirs = [d for d in os.listdir(runs_dir) if tag in d and os.path.isdir(os.path.join(runs_dir, d))]
    if not candidate_dirs:
        logger.error(f"No run folder found for tag: {tag}")
        return
    
    # Sort by name (usually timestamp) and pick latest
    candidate_dirs.sort(reverse=True)
    target_run_dir = os.path.join(runs_dir, candidate_dirs[0])
    logger.info(f"Using run directory: {target_run_dir}")
    
    # 2. Load Scenarios
    with open(stress_scenarios_path, 'r') as f:
         scenarios_cfg = yaml.safe_load(f)
    scenarios = scenarios_cfg.get("scenarios", scenarios_cfg)

    results = []
    
    postures = ["prudencial", "balanceado", "desinversion"]
    
    for posture in postures:
        posture_dir = os.path.join(target_run_dir, posture)
        
        # Try finding the decision file
        decision_file = None
        if os.path.exists(posture_dir):
            files = os.listdir(posture_dir)
            # Pattern: final_decisions_*.xlsx
            for f in files:
                if f.startswith("final_decisions") and f.endswith(".xlsx"):
                    decision_file = os.path.join(posture_dir, f)
                    break
        
        if not decision_file:
            logger.warning(f"No decision file found for posture {posture} in {posture_dir}")
            continue
            
        logger.info(f"Processing Posture: {posture}")
        df_decisions = pd.read_excel(decision_file)
        
        # Check if necessary columns exist
        required = ["EAD", "Accion_final"]
        if not all(c in df_decisions.columns for c in required):
             logger.warning(f"Skipping {posture}: Missing columns in decision file.")
             continue

        # 3. Iterate Scenarios
        for sc_name, sc_data in scenarios.items():
            shocks = sc_data.get("shocks", {})
            
            # Apply shocks to a copy
            try:
                df_shocked = apply_shocks(df_decisions, sc_name, shocks)
            except ValueError as e:
                logger.warning(f"Skipping {sc_name} validation error: {e}")
                continue
            
            # 4. Re-calculate Metrics (Simplified Financial Engine)
            # We need to re-compute EVA/RWA based on NEW PD/LGD/RW but OLD Action.
            
            # Constants
            hurdle = float(cfg.CONFIG.regulacion.hurdle_rate)
            
            # Basic RWA Recalc
            # RWA = EAD * RW (Shocked)
            # If Action was Sell, RWA becomes 0 (usually). 
            # If Action was Restruct, RW might have changed? For "Light" backtest, assume RW_shock applies to base, 
            # and if restruct happened, maybe we should apply the restruct improvement factor? 
            # Simplification: Apply shock to current 'RW' column which reflects the post-action state?
            # Actually, 'final_decisions' usually contains the FINAL state features.
            # So if we shock them, we are shocking the post-action state. This is valid for "future stress".
            
            # Re-calculate RWA
            if "RW" in df_shocked.columns and "EAD" in df_shocked.columns:
                df_shocked["RWA_stressed"] = df_shocked["EAD"] * df_shocked["RW"]
            else:
                 df_shocked["RWA_stressed"] = 0.0

            # Re-calculate EL (Expected Loss) = PD * LGD * EAD
            if "PD" in df_shocked.columns and "LGD" in df_shocked.columns:
                 df_shocked["EL_stressed"] = df_shocked["PD"] * df_shocked["LGD"] * df_shocked["EAD"]
            else:
                 df_shocked["EL_stressed"] = 0.0
                 
            # Re-calculate NI (Net Income) roughly
            # NI = (Interest - Funding - EL)
            # We use existing NI if available and adjust by Delta EL?
            # Better: NI = EAD * (InterestRate - FundingCost) - EL
            # Funding Cost 
            cost_fund = 0.006 
            if "interest_rate" in df_shocked.columns:
                rate = df_shocked["interest_rate"]
            else:
                rate = 0.05 # Default assumption
            
            ni = df_shocked["EAD"] * (rate - cost_fund) - df_shocked["EL_stressed"]
            
            # EVA = NI - Hurdle * RWA_stressed
            df_shocked["EVA_stressed"] = ni - (hurdle * df_shocked["RWA_stressed"])
            
            # Impact of SALES under stress (Liquidity Crunch)
            # If Action was VENDER, and "bid_haircut_mult" is in shocks, we lose more money.
            # Realized PnL = Price - Book. Price decreases.
            if "bid_haircut_mult" in shocks:
                 haircut_mult = shocks["bid_haircut_mult"]
                 # If we have "precio_optimo" columns
                 if "precio_optimo" in df_shocked.columns:
                      # Reduce price
                      # If haircut mult > 1 means stricter haircut -> lower price? 
                      # Usually scenario defines "haircut increase". 
                      # Let's assume bid_haircut_mult applied to the haircut %?
                      # Or simply Price = Price / Mult? 
                      # Let's assume Price = Price * (1 - (Mult-1)*0.1) generic? 
                      # Simplest: Price = Price * (1/Mult) if Mult represents "Difficulty"
                      
                      # Let's interpret bid_haircut_mult: 1.3 means 30% more haircut.
                      # Original Price ~ EAD * (1-Haircut). New Price ~ EAD * (1 - Haircut*1.3).
                      # We don't know original haircut easily.
                      # HEURISTIC: Reduce 'precio_optimo' by 5% per 0.1 increment above 1.0?
                      # Better: Price_Stressed = Price * (1 / bid_haircut_mult)
                      
                      df_shocked.loc[df_shocked["Accion_final"]=="VENDER", "precio_optimo"] /= haircut_mult
                      
                      # Re-calc PnL Realized
                      # PnL = Price - Book
                      if "book_value" in df_shocked.columns:
                           df_shocked["pnl_realized"] = df_shocked["precio_optimo"] - df_shocked["book_value"]
            
            # Aggregation
            total_eva = df_shocked["EVA_stressed"].sum()
            total_rwa = df_shocked["RWA_stressed"].sum()
            total_el  = df_shocked["EL_stressed"].sum()
            
            # Realized Capital Release (if Sell/Restruct)
            # CapRel = RWA_pre - RWA_post.
            # In stress, RWA_pre (baseline) would ALSO be higher, so maybe the delta is similar?
            # Let's just track the TOTAL EVA and TOTAL RWA of the portfolio after shocks.
            
            results.append({
                "tag": tag,
                "posture": posture,
                "scenario": sc_name,
                "EVA_stressed": total_eva,
                "RWA_stressed": total_rwa,
                "EL_stressed": total_el,
                "shock_desc": str(shocks)
            })
            
    # 5. Save Report
    if results:
        df_res = pd.DataFrame(results)
        out_path = os.path.join(ROOT_DIR, "reports", f"backtesting_light_{tag}.csv")
        df_res.to_csv(out_path, index=False)
        logger.info(f"Backtesting Complete. Saved to {out_path}")
        print(df_res.to_markdown())
        
        # Markdown summary
        md_path = out_path.replace(".csv", ".md")
        with open(md_path, "w") as f:
            f.write(f"# Backtesting Light Report: {tag}\n\n")
            f.write(df_res.to_markdown())
    else:
        logger.error("No backtesting results produced.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stress-scenarios", default="configs/stress_scenarios.yaml")
    args = parser.parse_args()
    
    run_backtesting_light(args.tag, args.stress_scenarios)
