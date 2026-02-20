# -*- coding: utf-8 -*-
"""
reports/make_committee_pack.py

Creates a reproducible Committee Pack containing all key artifacts, logs,
configuration, and execution manifest (git hash, python env, checksums).

Usage:
    python -m reports.make_committee_pack --tag pc5_postures_validation
"""

import os
import sys
import argparse
import logging
import shutil
import json
import glob
import hashlib
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def calculate_file_hash(filepath: str) -> Optional[str]:
    """Calculates SHA256 hash of a file."""
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_info() -> Dict[str, str]:
    """Retrieves current git commit hash and status."""
    try:
        # Check if git is available
        subprocess.check_output(["git", "--version"], stderr=subprocess.STDOUT)
        
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT_DIR).decode().strip()
        return {"commit": commit, "status": "clean" if not status else "dirty"}
    except Exception as e:
        logger.warning(f"Could not retrieve git info: {e}")
        return {"commit": "unknown", "status": "unknown"}

def find_latest_run_folder(tag: str, posture: str) -> Optional[str]:
    """Finds the latest run folder for a given tag and posture."""
    # Search for pattern *coordinated_inference*<tag>*<posture>*
    search_pattern = os.path.join(ROOT_DIR, "reports", f"*coordinated_inference*{tag}*{posture}*")
    folders = glob.glob(search_pattern)
    folders = [f for f in folders if os.path.isdir(f)]
    
    if not folders:
        return None
        
    # Sort by creation time ensures we get the latest run
    folders.sort(key=os.path.getctime, reverse=True)
    return folders[0]

def main():
    parser = argparse.ArgumentParser(description="Create Committee Pack for auditing.")
    parser.add_argument("--tag", default="pc5_postures_validation", help="Tag of the RL run cycle to package.")
    args = parser.parse_args()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_name = f"committee_pack_{args.tag}_{timestamp_str}"
    pack_dir = os.path.join(ROOT_DIR, "reports", pack_name)
    
    os.makedirs(pack_dir, exist_ok=True)
    logger.info(f"📦 Creating Committee Pack: {pack_dir}")

    manifest: Dict[str, Any] = {
        "timestamp": timestamp_str,
        "tag": args.tag,
        "git": get_git_info(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment": {},
        "models_checksums": {},
        "data_checksums": {},
        "artifacts": []
    }

    # 1. Environment Info (pip freeze)
    pip_freeze_path = os.path.join(pack_dir, "pip_freeze.txt")
    try:
        with open(pip_freeze_path, "w") as f:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f, check=True)
        manifest["environment"]["pip_freeze"] = "pip_freeze.txt"
    except Exception as e:
        logger.error(f"Failed to generate pip freeze: {e}")

    # 2. Config Snapshot (Extended for Audit)
    try:
        sys.path.append(ROOT_DIR)
        import config
        
        # Extract critical config values
        manifest["config_snapshot"] = {
            "COORDINATOR_PRIORITY": getattr(config, "COORDINATOR_PRIORITY", "N/A"),
            "STRICT_CONTRACT_VALIDATION": getattr(config, "STRICT_CONTRACT_VALIDATION", "N/A"),
            "GUARDRAILS": {k: v for k, v in config.__dict__.items() if k.startswith("GR_")}
        }
        
        # Copy config.py
        shutil.copy(os.path.join(ROOT_DIR, "config.py"), os.path.join(pack_dir, "config.py"))
        manifest["artifacts"].append("config.py")
    except Exception as e:
        logger.error(f"Failed to snapshot config: {e}")

    # 3. Models and Checksums
    models_dir = os.path.join(ROOT_DIR, "models")
    model_files = [
        "best_model_loan.zip",
        "best_model_portfolio.zip",
        "vecnormalize_loan.pkl",
        "vecnormalize_loan.meta.json",
        "training_metadata_loan.json",
        "obs_feature_order_loan.json"
    ]
    
    for m_file in model_files:
        src = os.path.join(models_dir, m_file)
        if os.path.exists(src):
            dst = os.path.join(pack_dir, m_file)
            shutil.copy(src, dst)
            manifest["models_checksums"][m_file] = calculate_file_hash(src)
            manifest["artifacts"].append(m_file)
        else:
            logger.warning(f"Model file not found: {m_file}")

    # 4. Data Checksums
    data_dir = os.path.join(ROOT_DIR, "data")
    data_files = ["portfolio_synth.xlsx", "portfolio_synth_smoke.xlsx"]
    
    for d_file in data_files:
        src = os.path.join(data_dir, d_file)
        if os.path.exists(src):
            # We don't copy data files to pack usually (too big/sensitive), but we hash them
            manifest["data_checksums"][d_file] = calculate_file_hash(src)

    # 4b. Runbook
    runbook_path = os.path.join(ROOT_DIR, "RUNBOOK_COMMITTEE.md")
    if os.path.exists(runbook_path):
        shutil.copy2(runbook_path, pack_dir)
        manifest["artifacts"].append("RUNBOOK_COMMITTEE.md")
    else:
        logger.warning(f"RUNBOOK_COMMITTEE.md not found at {runbook_path}")

    # 5. RL Artifacts (Latest per posture)
    postures = ["prudencial", "balanceado", "desinversion"]
    for posture in postures:
        folder = find_latest_run_folder(args.tag, posture)
        if not folder:
            logger.warning(f"⚠️ No run folder found for posture: {posture} (tag: {args.tag})")
            continue
        
        logger.info(f"Found run for {posture}: {os.path.basename(folder)}")
        
        # Determine source files
        files_to_copy = [
            f"portfolio_kpis_{posture}.json",
            f"overrides_log_{posture}.csv"
        ]
        
        # Excel file (pattern match)
        excel_files = glob.glob(os.path.join(folder, "decisiones_finales_*.xlsx"))
        
        # Destination subfolder inside pack to avoid filename collisions
        # Structure: reports/committee_pack_.../run_prudencial/...
        dest_subfolder = os.path.join(pack_dir, f"run_{posture}")
        os.makedirs(dest_subfolder, exist_ok=True)

        # Copy KPIs and Overrides
        for fname in files_to_copy:
            src = os.path.join(folder, fname)
            if os.path.exists(src):
                shutil.copy2(src, dest_subfolder)
                manifest["artifacts"].append(f"run_{posture}/{fname}")
            else:
                logger.warning(f"  Missing artifact: {fname} in {folder}")

        # Copy Excel
        if excel_files:
            excel_name = os.path.basename(excel_files[0])
            shutil.copy2(excel_files[0], dest_subfolder)
            manifest["artifacts"].append(f"run_{posture}/{excel_name}")
        else:
            logger.warning(f"  Result Excel not found in {folder}")

    # 4. Report Artifacts
    # Attempt to find compare_postures CSV
    compare_pattern = os.path.join(ROOT_DIR, "reports", f"compare_postures_*.csv")
    found_compares = glob.glob(compare_pattern)
    # Filter somewhat by tag if possible
    found_compares_tag = [f for f in found_compares if args.tag in f]
    
    report_files_src = []
    
    if found_compares_tag:
        # Latest by mtime
        found_compares_tag.sort(key=os.path.getmtime, reverse=True)
        report_files_src.append(found_compares_tag[0])
    elif found_compares:
        # Fallback to any compare posture
        found_compares.sort(key=os.path.getmtime, reverse=True)
        report_files_src.append(found_compares[0])
        
    # Explicit reports
    explicit_reports = [
        "reports/evaluation_pc6.csv",
        "reports/evaluation_report.md"
    ]
    for r in explicit_reports:
        r_abs = os.path.join(ROOT_DIR, r)
        if os.path.exists(r_abs):
            report_files_src.append(r_abs)
    
    # Copy reports
    for r_src in report_files_src:
        if os.path.exists(r_src):
            shutil.copy2(r_src, pack_dir)
            manifest["artifacts"].append(os.path.basename(r_src))
        else:
            logger.warning(f"⚠️ Report not found: {r_src}")

    # 5. Log Artifacts
    log_files = [
        "logs/qa_checkpoint5_evidence.txt",
        "logs/qa_checkpoint6_evidence.txt"
    ]
    for l_file in log_files:
        src = os.path.join(ROOT_DIR, l_file)
        if os.path.exists(src):
            shutil.copy2(src, pack_dir)
            manifest["artifacts"].append(os.path.basename(l_file))
        else:
            logger.warning(f"⚠️ Log not found: {l_file}")

    # 6. Checksums (Validation & Cleanup)
    # (Models and Data Checksums are already handled in previous sections)
    
    # Check if artifacts list matches reality or needs deduplication
    manifest["artifacts"] = sorted(list(set(manifest["artifacts"])))

    # 7. Generate Outputs List (Recursive)
    outputs_list = []
    for root, dirs, files in os.walk(pack_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, pack_dir)
            outputs_list.append(rel_path.replace("\\", "/"))
            
    manifest["outputs_list"] = sorted(outputs_list)

    # 8. Write Manifest
    manifest_path = os.path.join(pack_dir, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    
    logger.info(f"✅ Committee Pack generated successfully at: {pack_dir}")
    print(f"PACK_PATH={pack_dir}") # Output for callers

if __name__ == "__main__":
    main()
