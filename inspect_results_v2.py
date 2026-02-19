import pandas as pd
import json
import os
import glob
import sys
import shutil

# 1. Inspect pytest output
print("=== COMPLETE PYTEST OUTPUT ===")
try:
    with open("pytest_output.txt", "r", encoding="utf-8") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading pytest_output.txt: {e}")

# Try to find the folder from the user request context (the one I ran recently)
# coordinated_inference_manual_test_20260219_231820_prudencial
# But if it's not clear, just find the latest created prudencial folder in reports
reports_list = glob.glob("reports/coordinated_inference_*_prudencial")
if not reports_list:
    print("No prudencial reports found.")
    sys.exit(1)

# Sort by creation time
reports_list.sort(key=os.path.getctime)
target_dir = reports_list[-1]
print(f"\nAnalyzing folder: {target_dir}")

# 3. Load Excel
excel_file = os.path.join(target_dir, "decisiones_finales_prudencial.xlsx")
if os.path.exists(excel_file):
    print(f"\nLoading Excel: {excel_file}")
    try:
        df = pd.read_excel(excel_file)
        
        # Columns requested
        req_cols = [
            "override_applied", "override_level", "override_from", "override_to",
            "macro_selected", "macro_action_used", "macro_applied", "macro_conflict"
        ]
        
        # Filter existing columns
        existing_cols = [c for c in req_cols if c in df.columns]
        
        print(f"\n=== 15 ROWS AUDIT EXPORT ===")
        print(df[existing_cols].head(15).to_string(index=False))
        
        print("\n=== VALIDACION NaNs ===")
        for c in existing_cols:
            n_nan = df[c].isna().sum()
            print(f"- {c}: {n_nan} NaNs")
        
    except Exception as e:
        print(f"Error detecting NaNs in Excel: {e}")
else:
    print(f"Excel file not found: {excel_file}")

# 4. JSON
json_file = os.path.join(target_dir, "portfolio_kpis_prudencial.json")
if os.path.exists(json_file):
    print(f"\n=== PORTFOLIO KPIS JSON ===")
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading JSON: {e}")
else:
    print(f"JSON file not found: {json_file}")

# 5. Override Log CSV
csv_file = os.path.join(target_dir, "overrides_log_prudencial.csv")
if os.path.exists(csv_file):
    print(f"\n=== OVERRIDE LOG CSV HEAD (5 lines) ===")
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            for i in range(5):
                line = f.readline()
                if not line: break
                print(line.strip())
    except Exception as e:
        print(f"Error reading CSV: {e}")
else:
    print(f"CSV file not found: {csv_file}")
