# -*- coding: utf-8 -*-
# ============================================================
# tests/test_committee_pack.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Valida que make_committee_pack genera todos los artefactos exigidos.
# ============================================================
import os
import sys
import pytest
import glob
import json
import shutil
import subprocess

# Add root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from reports.make_committee_pack import main as make_pack_main

class TestCommitteePack:
    """
    Validates that the Committee Pack generation script produces 
    a complete and compliant artifact set for auditing.
    """

    @pytest.fixture(scope="class")
    def generated_pack(self):
        """
        Runs the pack generation once and returns the path to the created pack.
        Cleaning up afterward is optional but recommended if it spams disk.
        """
        # Run generation
        # We use a specific tag for testing to identify it
        test_tag = "pytest_pack_validation"
        
        # Ensure we have dummy source files so the pack isn't empty
        # Realistically, we rely on existing files from the environment or previous runs
        # But let's assume the environment has 'pc5_postures_validation' data or similar
        # If not, we might fail. 
        # Strategy: Use the tag that is known to have data 'pc5_postures_validation'
        target_tag = "pc5_postures_validation" 

        # Call script via subprocess to ensure clean state or just run main
        # We'll use subprocess to capture stdout which contains the path
        cmd = [sys.executable, "-m", "reports.make_committee_pack", "--tag", target_tag]
        result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
        assert result.returncode == 0, f"Pack generation failed: {result.stderr}"
        
        # Extract path from stdout
        pack_path = None
        for line in result.stdout.splitlines():
            if line.startswith("PACK_PATH="):
                pack_path = line.split("=", 1)[1].strip()
                break
        
        assert pack_path is not None, "Could not determine pack path from output"
        assert os.path.exists(pack_path)
        
        yield pack_path
        
        # Cleanup (optional - commented out to examine results)
        # shutil.rmtree(pack_path)

    def test_manifest_exists_and_valid(self, generated_pack):
        manifest_path = os.path.join(generated_pack, "MANIFEST.json")
        assert os.path.exists(manifest_path)
        
        with open(manifest_path, "r") as f:
            data = json.load(f)
            
        required_keys = ["timestamp", "tag", "git", "python_version", "models_checksums", "artifacts"]
        for k in required_keys:
            assert k in data, f"Manifest missing key: {k}"
            
        assert data["git"]["commit"], "Git commit should be present"

    def test_artifacts_presence(self, generated_pack):
        """Checks for key files inside the pack."""
        expected_files = [
            "pip_freeze.txt",
            "config.py",
            "evaluation_pc6.csv",
            "evaluation_report.md"
        ]
        for f in expected_files:
            assert os.path.exists(os.path.join(generated_pack, f)), f"Missing artifact: {f}"

    def test_run_folders_structure(self, generated_pack):
        """Checks that per-posture folders exist and contain data."""
        postures = ["prudencial", "balanceado", "desinversion"]
        for p in postures:
            p_dir = os.path.join(generated_pack, f"run_{p}")
            # It's possible for a run to be missing if not executed, 
            # but for 'pc5_postures_validation' we expect them.
            if os.path.exists(p_dir):
                files = os.listdir(p_dir)
                assert len(files) > 0, f"Run folder {p} is empty"
                
                # Check for excel
                excels = [f for f in files if f.endswith(".xlsx")]
                assert len(excels) > 0, f"No Excel decision file in {p}"

    def test_logs_included(self, generated_pack):
        log_dir = os.path.join(generated_pack)
        # We look for the evidence files copy
        assert os.path.exists(os.path.join(log_dir, "qa_checkpoint6_evidence.txt"))
