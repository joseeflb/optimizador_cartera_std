# CHECKLIST FOR COMMITTEE / AUDIT REVIEW (PC8)

**Project**: NPL Portfolio Optimizer (L1.5 Model)
**Date**: February 20, 2026
**Reviewer**: ____________________

This checklist certifies that the delivered "Committee Pack" meets all regulatory and technical requirements for Model Governance review.

## 1. Integrity & Reproducibility
- [ ] **Git Status Clean**: The repository was in a clean state (no uncommitted changes) when the pack was generated.
- [ ] **Source Code Versioned**: `MANIFEST.json` contains the exact git commit hash.
- [ ] **Environment Pinned**: `pip_freeze.txt` included in the pack matches the execution environment.
- [ ] **Data Checksums**: Input data (`portfolio_synth.xlsx`) SHA256 hash matches the authorized dataset.
- [ ] **Model Checksums**: Weights (`best_model_*.zip`) and normalization stats (`vecnormalize_*.pkl`) are hashed and included.

## 2. Prudential Controls (Hard Constraints)
- [ ] **Contract Validation**: `STRICT_CONTRACT_VALIDATION = True` validated in `config_snapshot`.
- [ ] **Restructuring Guardrails**:
    - [ ] `GR_PTI_MAX` (Payment-to-Income) <= 45%
    - [ ] `GR_DSCR_MIN` (Debt Service Coverage) >= 1.10x
- [ ] **Sales Guardrails**:
    - [ ] `GR_SELL_MIN_BID_PCT_EAD` >= 5%
    - [ ] Fire-sale loss caps applied for 'Prudential' posture.
- [ ] **Override Logging**: Any deviation from strict policy is logged in `overrides_log_*.csv`.

## 3. Evidence of Performance
- [ ] **Monotonicity**: Postures show logical progression in Risk/Return trade-off.
    - [ ] Prudential: Max EVA, Min Capital Release.
    - [ ] Desinversión: Max Capital Release, Lower EVA.
- [ ] **Benchmarking**: RL Agent strictly outperforms naive baselines (Holding / Mass Selling) in Net Value (EVA).
- [ ] **QA Checkpoints**:
    - [ ] `qa_checkpoint5_evidence.txt` (Environment/Step checks)
    - [ ] `qa_checkpoint6_evidence.txt` (Evaluation metrics)
    - [ ] `qa_checkpoint7_committee_pack_evidence.txt` (Packaging/Manifest)

## 4. Documentation & Artifacts
- [ ] **Executive Memo**: `MEMO_COMMITTEE.md` included (1-2 pages summary).
- [ ] **Runbook**: `RUNBOOK_COMMITTEE.md` included (Step-by-step execution guide).
- [ ] **Test Suite**: `pytest` output confirms 100% pass rate (26/26 tests).
- [ ] **Financial Reports**:
    - [ ] `compare_postures_*.csv` (Cross-strategy indicators)
    - [ ] `evaluation_report.md` (Detailed baseline comparison)
    - [ ] `decisiones_finales_*.xlsx` (Loan-level action list)

## 5. Known Limitations (PC8)
- [ ] **Synthetic Data**: Results based on 'sanity' dataset, not full production dump.
- [ ] **Calibration**: Pricing models (LGD/Haircuts) pending final calibration with Risk Dept.
- [ ] **Macro Horizon**: Current optimization is single-step (12-24m forward view).

---
**Sign-off**:
[ ] Validated by Data Science Lead
[ ] Ready for Model Validation (Internal Audit)
