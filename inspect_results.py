import os
import glob
import json
import pandas as pd

# Find the latest prudencial folder
base_dir = "reports"
# Look for folders starting with coordinated_inference_run1_ and ending with prudencial
pattern = os.path.join(base_dir, "coordinated_inference_run1_*_prudencial")
files = glob.glob(pattern)

if not files:
    print(f"No folder found matching {pattern}")
    exit(1)

latest = max(files, key=os.path.getctime)
print(f"Using report folder: {latest}")

# 1. KPIs
kpi_path = os.path.join(latest, "portfolio_kpis_prudencial.json")
if os.path.exists(kpi_path):
    with open(kpi_path, "r") as f:
        print("\n=== portfolio_kpis.json ===")
        print(json.dumps(json.load(f), indent=2))
else:
    print(f"File not found: {kpi_path}")

# 2. Override Log
log_path = os.path.join(latest, "overrides_log_prudencial.csv")
print("\n=== override_log.csv (Header + 5 lines) ===")
if os.path.exists(log_path):
    with open(log_path, "r", encoding='utf-8') as f:
        for i in range(6):
            line = f.readline()
            if not line: break
            print(line.strip())
else:
    print(f"File not found: {log_path}")

# 3. Decisiones Finales (Excel)
excel_path = os.path.join(latest, "decisiones_finales_prudencial.xlsx")
print("\n=== Export Columns (10 rows) ===")
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    cols = [
        "Accion_micro", "Accion_final", "override_level", "override_from", "override_to",
        "macro_selected", "macro_action_used", "macro_applied", "macro_conflict"
    ]
    # Filter only existing columns
    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].head(10).to_string())
else:
    print(f"File not found: {excel_path}")
