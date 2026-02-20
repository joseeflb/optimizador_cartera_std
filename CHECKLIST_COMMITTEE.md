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

## 6. Robustness under Stress (PC9)
- [x] **Stress Engine (Multi-Scenario)**: `engines/stress_engine.py` re-runs inference under 4 macro scenarios
      (baseline, mild, severe, pricing_crunch) x 3 postures (12 runs). KPIs verified to be non-zero.
- [x] **Backtesting Light**: `reports/backtesting_light.py` re-applies shocks to frozen decisions.
      Posture differentiation confirmed: prudencial 445/500 on-book, desinversion 92/500 on-book.
- [x] **Monotonicity under Stress**: EVA and capital_release degrade coherently under mild/severe.
- [x] **pricing_crunch not a no-op**: `cfg.BID_HAIRCUT_GLOBAL` injected before coordinator_inference;
      price_simulator applies 30% NPL price haircut; test `test_stress_pricing_crunch_effect.py` validates effect.
- [x] **Strict Ingestion (Bank-Ready)**: `ALLOW_CLIP_OUT_OF_RANGE=False` by default; negative test
      (PD=1.5) confirmed raises `ValueError` within < 1s.
- [x] **No Silent Legacy Fallback**: `ALLOW_LEGACY_PORTFOLIO_LOAD=False`; coordinator aborts with
      actionable message if ingestion fails.
- [x] **QA Evidence PC9**: `logs/qa_checkpoint9_evidence.txt` produced with artifact paths, CSV extracts,
      pytest output, and CI log reference.
- [x] **Committee Pack Updated**: `make_committee_pack.py` now includes `stress_scenarios.yaml`,
      `real_portfolio_mapping.yaml`, `ingest_portfolio.py`, `MEMO_COMMITTEE.md`, and CI log.

---
**Sign-off**:
[ ] Validated by Data Science Lead
[ ] Validated by Risk/Quantitative Model team (PC9 stress scenarios)
[ ] Ready for Model Validation (Internal Audit)
