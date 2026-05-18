# -*- coding: utf-8 -*-
# ============================================================
# reports/compare_postures.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Comparativa de outputs entre posturas (Prudencial/Balanceado/Desinversión) recogiendo los runs recientes.
# ============================================================
import os
import glob
import json
import logging
import argparse
import pandas as pd
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))

def get_recent_run_folders(tag, limit=3):
    """
    Finds the most recent run folders matching the tag in reports/.
    Expects folders like: coordinated_inference_<tag>_* or similar containing the tag.
    """
    # Search for directories containing the tag and starting with coordinated_inference_
    candidates = []
    for item in os.listdir(REPORTS_DIR):
        full_path = os.path.join(REPORTS_DIR, item)
        if os.path.isdir(full_path) and tag in item and item.startswith("coordinated_inference_"):
            candidates.append(full_path)
    
    # Sort by modification time (descending)
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Group by posture and take latest
    selected = {}
    for folder in candidates:
        posture = parse_posture_from_path(folder)
        if posture not in selected and posture != 'unknown':
            selected[posture] = folder
            
    # Return the selected folders as a list
    return list(selected.values())

def parse_posture_from_path(folder_path):
    # Try to infer posture from folder name
    name = os.path.basename(folder_path).lower()
    if 'prudencial' in name:
        return 'prudencial'
    if 'balanceado' in name:
        return 'balanceado'
    if 'desinversion' in name:
        return 'desinversion'
    return 'unknown'

def load_kpis(folder_path, posture):
    # Look for portfolio_kpis_*.json
    # It might be portfolio_kpis_prudencial.json or just portfolio_kpis.json
    # We'll search glob
    json_files = glob.glob(os.path.join(folder_path, "portfolio_kpis*.json"))
    if not json_files:
        logger.warning(f"No KPI json found in {folder_path}")
        return {}
    
    # Pick the most likely one
    target_file = json_files[0]
    for f in json_files:
        if posture in f:
            target_file = f
            break
            
    with open(target_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_overrides(folder_path, posture):
    csv_files = glob.glob(os.path.join(folder_path, "overrides_log*.csv"))
    if not csv_files:
        return pd.DataFrame()
    
    target_file = csv_files[0]
    for f in csv_files:
        if posture in f:
            target_file = f
            break
            
    try:
        df = pd.read_csv(target_file)
        return df
    except Exception as e:
        logger.warning(f"Error reading overrides {target_file}: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Compare 3 postures results")
    parser.add_argument("--tag", required=True, help="Common tag used in run")
    args = parser.parse_args()
    
    folders = get_recent_run_folders(args.tag, limit=3)
    if len(folders) < 3:
        logger.error(f"Found less than 3 folders for tag {args.tag}: {folders}")
        # We proceed anyway if we have at least 1, but warn
    
    logger.info(f"Analyzing folders: {folders}")
    
    records = []
    
    for folder in folders:
        posture = parse_posture_from_path(folder)
        kpis = load_kpis(folder, posture)
        overrides_df = load_overrides(folder, posture)
        
        # Flatten KPI data
        # Structure: 
        # pre_state: {total_eva, total_rwa, ...}
        # final_state: {total_eva, total_rwa, total_capital_release, action_counts: {...}}
        
        pre = kpis.get('pre_state', {})
        final = kpis.get('final_state', {})
        
        rec = {
            'posture': posture,
            'folder': os.path.basename(folder),
            'pre_total_eva': pre.get('total_eva', 0),
            'final_total_eva': final.get('total_eva', 0),
            'delta_total_eva': final.get('total_eva', 0) - pre.get('total_eva', 0),
            
            'pre_total_rwa': pre.get('total_rwa', 0),
            'final_total_rwa': final.get('total_rwa', 0),
            'delta_total_rwa': final.get('total_rwa', 0) - pre.get('total_rwa', 0),
            
            'final_capital_release': final.get('total_capital_release', 0),
        }
        
        # Action counts
        actions = final.get('action_counts', {})
        rec['final_action_keep'] = actions.get('MANTENER', 0)
        rec['final_action_sell'] = actions.get('VENDER', 0)
        rec['final_action_restruct'] = actions.get('REESTRUCTURAR', 0)
        
        # Overrides
        if not overrides_df.empty:
            rec['override_count_total'] = len(overrides_df)
            # Count by level
            if 'level' in overrides_df.columns:
                # E.g. GUARDRAIL, MACRO_PRUDENTIAL
                counts = overrides_df['level'].value_counts().to_dict()
                rec['override_count_guardrail'] = sum(v for k,v in counts.items() if 'GUARDRAIL' in str(k).upper())
                rec['override_count_macro_prudential'] = sum(v for k,v in counts.items() if 'MACRO' in str(k).upper())
                # Add others if needed
            else:
                rec['override_count_guardrail'] = 0
                rec['override_count_macro_prudential'] = 0
        else:
            rec['override_count_total'] = 0
            rec['override_count_guardrail'] = 0
            rec['override_count_macro_prudential'] = 0
            
        records.append(rec)
        
    df = pd.DataFrame(records)
    
    # Sort logically
    order_map = {'prudencial': 0, 'balanceado': 1, 'desinversion': 2}
    df['sort_key'] = df['posture'].map(order_map).fillna(99)
    df = df.sort_values('sort_key').drop(columns=['sort_key'])
    
    output_path = os.path.join(REPORTS_DIR, f"compare_postures_{args.tag}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison saved to {output_path}")
    print("\n=== POSTURE COMPARISON TABLE ===")
    print(df.to_string(index=False))
    
    # ---------------------------------------------------------
    # VALIDATION LOGIC (Task 3)
    # ---------------------------------------------------------
    # Expected monotonies
    # Sell count: Prudencial <= Balanceado <= Desinversion
    # Capital Release: Prudencial <= Balanceado <= Desinversion
    # RWA Final: Desinversion <= Balanceado <= Prudencial
    
    # Helper to get value
    def get_val(p, col):
        row = df[df['posture'] == p]
        if row.empty:
            return None
        return row.iloc[0][col]
    
    pass_validation = True
    
    print("\n=== VALIDATION CHECKS ===")
    
    # 1. SELL COUNT
    s_pru = get_val('prudencial', 'final_action_sell')
    s_bal = get_val('balanceado', 'final_action_sell')
    s_des = get_val('desinversion', 'final_action_sell')
    
    if s_pru is not None and s_bal is not None and s_des is not None:
        check_sell = (s_pru <= s_bal) and (s_bal <= s_des)
        print(f"Check Sell Count (P<=B<=D): {s_pru} <= {s_bal} <= {s_des} -> {'PASS' if check_sell else 'FAIL'}")
        if not check_sell: pass_validation = False
    else:
        print("Check Sell Count: MISSING DATA")
        pass_validation = False

    # 2. CAPITAL RELEASE
    c_pru = get_val('prudencial', 'final_capital_release')
    c_bal = get_val('balanceado', 'final_capital_release')
    c_des = get_val('desinversion', 'final_capital_release')

    if c_pru is not None and c_bal is not None and c_des is not None:
        check_cap = (c_pru <= c_bal) and (c_bal <= c_des)
        print(f"Check Cap Release (P<=B<=D): {c_pru:,.0f} <= {c_bal:,.0f} <= {c_des:,.0f} -> {'PASS' if check_cap else 'FAIL'}")
        if not check_cap: pass_validation = False
    else:
        print("Check Cap Release: MISSING DATA")
        pass_validation = False

    # 3. RWA FINAL
    r_pru = get_val('prudencial', 'final_total_rwa')
    r_bal = get_val('balanceado', 'final_total_rwa')
    r_des = get_val('desinversion', 'final_total_rwa')

    if r_pru is not None and r_bal is not None and r_des is not None:
        # RWA: Desinversion <= Balanceado <= Prudencial
        check_rwa = (r_des <= r_bal) and (r_bal <= r_pru)
        print(f"Check RWA Final (D<=B<=P): {r_des:,.0f} <= {r_bal:,.0f} <= {r_pru:,.0f} -> {'PASS' if check_rwa else 'FAIL'}")
        if not check_rwa: pass_validation = False
    else:
        print("Check RWA Final: MISSING DATA")
        pass_validation = False
        
    if not pass_validation:
        print("\nOVERALL STATUS: FAIL (Some checks failed, but pipeline continues)")
        # We don't exit(1) here as per instructions "sin romper pipeline todavía"
    else:
        print("\nOVERALL STATUS: PASS")

if __name__ == "__main__":
    main()
