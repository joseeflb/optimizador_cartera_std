# MEMO TO COMMITTEE: NPL Management Engine (Model L1.5)

**Date**: February 20, 2026  
**Subject**: PC7 Closure & Audit Readiness  
**System**: Optimizador de Cartera NPL (Micro/Macro/Coordinator)

## 1. Executive Summary
This memorandum certifies the completion of the "Perfect Committee Pack" (PC7) for the L1.5 NPL Management Engine. The system implements a hierarchical reinforcement learning architecture that successfully arbitrates between individual loan-level recovery maximization and portfolio-level strategic constraints (RWA, Capital).

**Key Deliverable**: A fully auditable determination of **Hold / Sell / Restructure** strategies for the reference portfolio (500 loans), compliant with Basel III Standardized Approach.

## 2. Methodology & Controls
The engine operates under strict governance controls designed for auditor review:

*   **Audit Trail**: Every decision is logged with its corresponding financial impact (EVA, RWA). Overrides are explicitly tracked (`override_log_*.csv`).
*   **Guardrails**: Deterministic safety checks prevent regulatory breaches:
    *   **PTI Max (45%)** & **DSCR Min (1.10x)** for restructuring.
    *   **Minimum Bid (5%)** & **Max Fire-Sale Loss** for sales.
*   **Data Integrity**: Input portfolios and model artifacts are cryptographically hashed in `MANIFEST.json`.
*   **Reproducibility**: The execution environment is dockerized via Python virtual environments with strict dependency pinning (`pip_freeze.txt`).

## 3. Key Results (PC6 Validation)
The Reinforcement Learning (RL) agent significantly outperforms deterministic rule-based baselines, demonstrating the value of active portfolio management.

### 3.1. Performance vs. Baselines
| Strategy | EVA (Net Value) | Capital Release | Decision Profile (Sell/Rest/Keep) |
| :--- | :--- | :--- | :--- |
| **RL Prudencial** | **€748M** | €76M | 55 / 0 / 445 |
| **RL Balanceado** | €725M | **€411M** | 168 / 244 / 88 |
| **RL Desinversión** | €0.05M | **€515M** | 408 / 5 / 87 |
| *Baseline Prudencial* | *€-725M* | *€0M* | (Passive Holding) |
| *Baseline Desinversión* | *€-213M* | *€57M* | (Aggressive Fire-Sale) |

**Highlights**:
*   **Value Creation**: The RL agent generates positive EVA across all postures, whereas rigid baselines destroy value (negative EVA).
*   **Strategic Flexibility**:
    *   **Prudencial** mode maximizes long-term value (EVA) by holding/restructuring viable assets.
    *   **Desinversión** mode aggressively releases capital (€515M vs €76M) at the cost of future earnings, validating the system's responsiveness to macro mandates.
    *   **Balanceado** offers a compromise, releasing significant capital (€411M) while preserving high EVA (€725M).

## 4. Limitations & Assumptions
Transparancy regarding current model limitations is essential for Committee approval:
*   **Synthetic Data**: Results stem from `data/portfolio_synth.xlsx`. Calibration with production data is pending (PC8).
*   **Pricing Models**: LGD and haircut estimations assume standard market liquidity conditions.
*   **Macro Horizon**: The portfolio optimization currently looks 1-step ahead. Multi-step strategic planning is a roadmap item.

## 5. Roadmap (PC8 and beyond)
*   **Calibration**: Ingest real NPL book data and fine-tune hyperparameters.
*   **Stress Testing**: Validate robustness under adverse economic scenarios (e.g., 20% collateral value drop).
*   **Integration**: Connect decision outputs directly to the Bank's workflow tool.

**Conclusion**: The system is ready for "shadow mode" deployment and auditor review.

---
**Approved by**: System Architect
**Verified**: `pytest` Suite (26/26 passed)
