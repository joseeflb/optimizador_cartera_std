# -*- coding: utf-8 -*-
"""
reports/make_committee_pack.py

Creates a reproducible Committee Pack containing all key artifacts, logs,
configuration, and execution manifest (git hash, python env, checksums).
Includes PC9 artifacts (Stress Testing & Backtesting).

Usage:
    python -m reports.make_committee_pack --tag pc9_final
"""

import os
import sys
import argparse
import logging
import shutil
import glob
import hashlib
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

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
        
        # Use simple try-except block for git commands
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR).decode().strip()
        except:
            commit = "unknown"
            
        try:
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT_DIR).decode().strip()
        except:
            status = "unknown"
            
        return {"commit": commit, "status": "clean" if not status else "dirty"}
    except Exception as e:
        logger.warning(f"Could not retrieve git info: {e}")
        return {"commit": "unknown", "status": "unknown"}

def main():
    parser = argparse.ArgumentParser(description="Create Committee Pack for auditing.")
    parser.add_argument("--tag", default="pc9_final", help="Tag of the run cycle to package.")
    args = parser.parse_args()

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_name = f"committee_pack_{args.tag}_{timestamp_str}"
    pack_dir = os.path.join(ROOT_DIR, "reports", pack_name)
    
    os.makedirs(pack_dir, exist_ok=True)
    logger.info(f"[U1F4E6] Creating Committee Pack for tag '{args.tag}': {pack_dir}")

    manifest: Dict[str, Any] = {
        "timestamp": timestamp_str,
        "tag": args.tag,
        "git": get_git_info(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "models_checksums": {},
        "artifacts": []
    }

    # 0. Models checksums
    models_dir = os.path.join(ROOT_DIR, "models")
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            fpath = os.path.join(models_dir, fname)
            if os.path.isfile(fpath):
                manifest["models_checksums"][fname] = calculate_file_hash(fpath)

    # 1. Config Snapshot
    try:
        shutil.copy(os.path.join(ROOT_DIR, "config.py"), os.path.join(pack_dir, "config.py"))
        manifest["artifacts"].append("config.py")
    except Exception as e:
        logger.error(f"Failed to copy config.py: {e}")

    # 2. Key Documentation
    docs = ["RUNBOOK_COMMITTEE.md", "README.md", "MEMO_COMMITTEE.md", "CHECKLIST_COMMITTEE.md"]
    for doc in docs:
        src = os.path.join(ROOT_DIR, doc)
        if os.path.exists(src):
            try:
                shutil.copy2(src, pack_dir)
                manifest["artifacts"].append(doc)
            except Exception as e:
                logger.warning(f"Could not copy {doc}: {e}")

    # 3. PC9 Artifacts: Stress Summary & Backtesting Light
    # Pattern: reports/stress_summary_<tag>.csv
    # Pattern: reports/backtesting_light_<tag>.csv / .md
    
    # Try exact match first
    stress_sum = os.path.join(ROOT_DIR, "reports", f"stress_summary_{args.tag}.csv")
    if not os.path.exists(stress_sum):
        # Fallback: maybe timestamped? Or user ran verification and wants final pack?
        # If tag is pc9_final, but verification run was pc9_verification...
        # We look for ANY recent stress summary if exact not found?
        pass

    if os.path.exists(stress_sum):
        shutil.copy2(stress_sum, pack_dir)
        manifest["artifacts"].append(os.path.basename(stress_sum))
        logger.info(f"Included stress summary: {stress_sum}")
    else:
        logger.warning(f"Stress summary not found: {stress_sum}")

    # Also include pricing_only stress summary if it exists (pricing_crunch fix verification)
    pricing_only_sum = os.path.join(ROOT_DIR, "reports", f"stress_summary_{args.tag}_pricing_only.csv")
    if not os.path.exists(pricing_only_sum):
        # Try canonical naming: stress_summary_pc9_pricing_only_postfix.csv -> strip trailing _postfix variant
        base_tag = args.tag.replace("_postfix", "").replace("_final", "")
        pricing_only_sum = os.path.join(ROOT_DIR, "reports", f"stress_summary_{base_tag}_pricing_only.csv")
    if os.path.exists(pricing_only_sum):
        shutil.copy2(pricing_only_sum, pack_dir)
        manifest["artifacts"].append(os.path.basename(pricing_only_sum))
        logger.info(f"Included pricing_only stress summary: {pricing_only_sum}")
        
    backtest_csv = os.path.join(ROOT_DIR, "reports", f"backtesting_light_{args.tag}.csv")
    if os.path.exists(backtest_csv):
        shutil.copy2(backtest_csv, pack_dir)
        manifest["artifacts"].append(os.path.basename(backtest_csv))
        logger.info(f"Included backtest csv: {backtest_csv}")
    else:
        logger.warning(f"Backtest CSV not found: {backtest_csv}")
    
    backtest_md = os.path.join(ROOT_DIR, "reports", f"backtesting_light_{args.tag}.md")
    if os.path.exists(backtest_md):
        shutil.copy2(backtest_md, pack_dir)
        manifest["artifacts"].append(os.path.basename(backtest_md))

    # 4. Evaluation Report (if exists)
    eval_rep = os.path.join(ROOT_DIR, "reports", f"evaluation_{args.tag}.csv") # Or typical name
    if not os.path.exists(eval_rep):
         eval_rep = os.path.join(ROOT_DIR, "reports", "evaluation_pc6.csv")
         
    if os.path.exists(eval_rep):
         shutil.copy2(eval_rep, pack_dir)
         manifest["artifacts"].append(os.path.basename(eval_rep))

    eval_md = os.path.join(ROOT_DIR, "reports", "evaluation_report.md")
    if os.path.exists(eval_md):
        shutil.copy2(eval_md, pack_dir)
        manifest["artifacts"].append("evaluation_report.md")

    # 5. QA evidence
    qa_ev = os.path.join(ROOT_DIR, "logs", "qa_checkpoint6_evidence.txt")
    if os.path.exists(qa_ev):
        shutil.copy2(qa_ev, pack_dir)
        manifest["artifacts"].append("qa_checkpoint6_evidence.txt")

    qa_ev9 = os.path.join(ROOT_DIR, "logs", "qa_checkpoint9_evidence.txt")
    if os.path.exists(qa_ev9):
        shutil.copy2(qa_ev9, pack_dir)
        manifest["artifacts"].append("qa_checkpoint9_evidence.txt")

    # 6. pip freeze
    try:
        pip_out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], cwd=ROOT_DIR
        ).decode("utf-8", errors="replace")
        pip_path = os.path.join(pack_dir, "pip_freeze.txt")
        with open(pip_path, "w", encoding="utf-8") as f:
            f.write(pip_out)
        manifest["artifacts"].append("pip_freeze.txt")
    except Exception as e:
        logger.warning(f"Could not generate pip freeze: {e}")

    # 7. Per-posture run folders with Excel decision files
    postures = ["prudencial", "balanceado", "desinversion"]
    for posture in postures:
        # Search for the most recent run for this tag + posture
        run_src = None
        # Try: reports/<tag>_<posture>/ or reports/inference_*_<tag>*_<posture>/
        patterns = [
            os.path.join(ROOT_DIR, "reports", f"*{args.tag}*{posture}*"),
            os.path.join(ROOT_DIR, "reports", "*", f"*{posture}*"),
            os.path.join(ROOT_DIR, "reports", "runs", f"*{args.tag}*", posture),
        ]
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(pat))
        candidates = [c for c in candidates if os.path.isdir(c)]
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            run_src = candidates[0]

        pack_posture_dir = os.path.join(pack_dir, f"run_{posture}")
        if run_src:
            # Copy Excel decision files — only create folder if we have something to copy
            for root_w, dirs_w, files_w in os.walk(run_src):
                for fname in files_w:
                    if fname.endswith(".xlsx"):
                        src_f = os.path.join(root_w, fname)
                        os.makedirs(pack_posture_dir, exist_ok=True)
                        shutil.copy2(src_f, pack_posture_dir)
                        manifest["artifacts"].append(f"run_{posture}/{fname}")
                        break  # Only first xlsx per posture is enough
                break  # Only top-level (one level deep)  

    # 8. Reproducibility artefacts: scenarios YAML, ingesta, mapping
    repro_files = [
        (os.path.join(ROOT_DIR, "configs", "stress_scenarios.yaml"), "stress_scenarios.yaml"),
        (os.path.join(ROOT_DIR, "data", "mappings", "real_portfolio_mapping.yaml"), "real_portfolio_mapping.yaml"),
        (os.path.join(ROOT_DIR, "data", "ingest_portfolio.py"), "ingest_portfolio.py"),
        (os.path.join(ROOT_DIR, "MEMO_COMMITTEE.md"), "MEMO_COMMITTEE.md"),
        (os.path.join(ROOT_DIR, "RUNBOOK_COMMITTEE.md"), "RUNBOOK_COMMITTEE.md"),
    ]
    for src_path, dest_name in repro_files:
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(pack_dir, dest_name))
            manifest["artifacts"].append(dest_name)

    # 9. Most recent ci_local log (for this tag or any pc9_final log)
    log_patterns = [
        os.path.join(ROOT_DIR, "logs", f"ci_local_{args.tag}_*.log"),
        os.path.join(ROOT_DIR, "logs", "ci_local_pc9_final_*.log"),
    ]
    ci_log_src = None
    for pat in log_patterns:
        matches = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if matches:
            ci_log_src = matches[0]
            break
    if ci_log_src:
        dest_log = os.path.join(pack_dir, os.path.basename(ci_log_src))
        shutil.copy2(ci_log_src, dest_log)
        manifest["artifacts"].append(os.path.basename(ci_log_src))
        logger.info(f"Included CI log: {ci_log_src}")

    # 10. Manifest — deduplicate artifact list preserving order
    manifest["artifacts"] = list(dict.fromkeys(manifest["artifacts"]))
    with open(os.path.join(pack_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=4)
        
    logger.info(f"[OK] Committee Pack generated successfully at: {pack_dir}")
    # Print machine-readable path for subprocess callers (tests / CI)
    print(f"PACK_PATH={pack_dir}")

if __name__ == "__main__":
    main()
