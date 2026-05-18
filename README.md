# Optimizador de Carteras NPL — Basilea III (Método Estándar)

**Autor:** José María Fernández-Ladreda Ballvé
**Trabajo de Fin de Grado (TFG)** — Optimización por *Reinforcement Learning* (PPO) de decisiones
de gestión de carteras de préstamos en *default* (NPL) bajo el Método Estándar de Basilea III.

---

## 1. Resumen ejecutivo

Este repositorio implementa un **sistema multi-agente de aprendizaje por refuerzo** que decide,
préstamo a préstamo y al mismo tiempo a nivel de cartera completa, entre tres acciones:

1. **MANTENER** el préstamo en balance.
2. **REESTRUCTURAR** (renegociar plazo, tipo y/o quita) con guardrails PTI/DSCR.
3. **VENDER** en mercado secundario, con un simulador de precio NPL y guardrail de *fire-sale*.

El sistema combina:

- Un **subagente MICRO** (`LoanEnv`, PPO): observa 10 variables del préstamo, elige una de 3 acciones.
- Un **subagente MACRO** (`PortfolioEnv`, PPO): observa 308 features de cartera, elige una de 12 acciones.
- Un **Coordinador** que arbitra ambas decisiones (con políticas auditables por postura) y produce
  la acción final con su *Reason Code* y *governance trail*.
- Tres **posturas** parametrizables: `PRUDENCIAL`, `BALANCEADO`, `DESINVERSION`.
- Una capa de **hardening bank-ready**: guardrails duros, completitud de métricas, *macro steering*,
  *Final Reasoning Seal* y empaquetado reproducible para comité.

El objetivo es producir decisiones **auditables, regulatoriamente coherentes y financieramente
justificadas** sobre carteras NPL, con un *committee pack* trazable end-to-end.

---

## 2. Arquitectura

```
┌──────────────────────────────────────────────────────────────────────┐
│                         PIPELINE COMPLETO                            │
└──────────────────────────────────────────────────────────────────────┘

  data/generate_portfolio.py   ──►  cartera sintética NPL (Basilea III)
            │                         │
            │                         ├─►  data/ingest_portfolio.py
            │                         │     (carga de carteras reales)
            ▼                         ▼
  ┌─────────────────────┐   ┌─────────────────────┐
  │   env/loan_env.py   │   │ env/portfolio_env.py│
  │   (MICRO, 10 obs,   │   │ (MACRO, 308 obs,    │
  │    3 acciones)      │   │  12 acciones)       │
  └──────────┬──────────┘   └──────────┬──────────┘
             │                         │
             ▼                         ▼
  ┌─────────────────────────────────────────────┐
  │ agent/train_subagents.py  (PPO + callbacks) │
  │  ─ checkpoints, VecNormalize, TensorBoard    │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────┐
  │ agent/coordinator_inference.py              │
  │  ─ arbitraje MICRO+MACRO, fire-sale,        │
  │    contrafactuals coherentes, schema estable │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────┐
  │ reports/hardening.py                        │
  │  ─ Final Reasoning Seal + Macro Steering    │
  │  ─ métricas post completas (no NaN)         │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────┐
  │ engines/stress_engine.py  (BCE/EBA multi-Q) │
  │ engines/sensitivities_engine.py             │
  │ reports/backtesting_dynamic.py              │
  └─────────────────────┬───────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────┐
  │ reports/make_committee_pack.py              │
  │  ─ artefactos + logs + config + manifest    │
  │    (git hash, env, checksums)               │
  └─────────────────────────────────────────────┘
```

Tres capas verticales atraviesan todo el pipeline:

- **`config.py`**: única fuente de verdad para parámetros financieros, regulatorios y de RL.
- **`optimizer/`**: guardrails y herramientas analíticas deterministas (no RL).
- **`risk/gates.py`**: funciones puras de gates (PTI/DSCR, fire-sale) reutilizadas en todo el grafo.

---

## 3. Estructura del repositorio

```
optimizador_cartera_std/
├── main.py                          # Orquestador del pipeline completo
├── config.py                        # Configuración global (Basilea III + PPO + posturas)
├── analyze_learning.py              # Análisis del aprendizaje PPO (TB + checkpoints)
├── install_requirements_smart.py    # Instalador inteligente (Python 3.14 compatible)
│
├── agent/                           # Capa RL (entrenamiento e inferencia)
│   ├── train_agent.py               # Entrenamiento PPO genérico (Loan o Portfolio)
│   ├── train_subagents.py           # Entrena MICRO + MACRO por separado
│   ├── coordinator_inference.py     # Arbitraje MICRO+MACRO (bank-ready)
│   ├── policy_inference.py          # Inferencia micro + heurística financiera
│   ├── policy_inference_coordinated.py
│   └── policy_inference_portfolio.py
│
├── env/                             # Entornos Gymnasium
│   ├── loan_env.py                  # MICRO: 10 obs / 3 acciones
│   └── portfolio_env.py             # MACRO: 308 obs / 12 acciones
│
├── data/                            # Generación e ingesta de carteras
│   ├── generate_portfolio.py        # Cartera sintética NPL
│   ├── ingest_portfolio.py          # Cartera real (mapping YAML)
│   ├── sanity_generator_checks.py   # Sanity checks del generador
│   └── mappings/                    # Mappings cartera real → schema canónico
│
├── engines/                         # Motores macro
│   ├── stress_engine.py             # Stress multi-Q (BCE/EBA)
│   └── sensitivities_engine.py      # Sensibilidades financieras
│
├── optimizer/                       # Optimizadores deterministas
│   ├── guardrails.py                # Restricciones duras REESTRUCT/VENDER
│   ├── restructure_optimizer.py     # Optimizador híbrido reestructuración
│   └── price_simulator.py           # Simulador precio NPL (mercado secundario)
│
├── baselines/                       # Políticas heurísticas (no RL)
│   └── baseline_policies.py         # Prudencial / Desinversión
│
├── multiagent/                      # Wrappers multi-agente
│   ├── coordinator_agent.py
│   ├── pricing_agent.py
│   └── restruct_agent.py
│
├── risk/
│   └── gates.py                     # Gates puros (PTI/DSCR, fire-sale)
│
├── reports/                         # Reporting, hardening y committee pack
│   ├── hardening.py                 # Final Reasoning Seal + Macro Steering
│   ├── schema.py                    # Esquema canónico CANON_COLS
│   ├── make_committee_pack.py       # Empaquetado reproducible
│   ├── backtesting_dynamic.py       # Backtesting multi-trimestre
│   ├── backtesting_light.py         # What-If sobre decisiones congeladas
│   ├── baseline_eval.py             # RL vs Baseline (real)
│   ├── compare_postures.py          # Comparativa entre las 3 posturas
│   ├── evaluate_against_baselines.py
│   ├── evaluate_against_baselines_sim.py
│   ├── export_financial_decisions.py
│   ├── export_styled_excel.py       # Excel corporativo CIB
│   ├── export_styled_excel_summary.py
│   ├── npl_posture_analysis.py      # Envelopes negociación
│   ├── posture_kpi_report.py        # KPIs ejecutivos por postura
│   ├── results_summary.py           # Consolidación RUN-LEVEL
│   ├── segment_dispersion_analysis.py
│   └── theoretical_optimal_bound.py # Cota óptima teórica V*
│
├── dashboard/
│   └── api_server.py                # Backend Flask (Lovable Matrix)
│
├── tests/                           # Suite pytest (smoke + unit + compliance)
│   ├── conftest.py
│   ├── smoke_test.py
│   ├── test_bank_ready_compliance.py
│   ├── test_committee_pack.py
│   ├── test_coordinator_overrides.py
│   ├── test_guardrails_extended.py
│   ├── test_guardrails_restructure.py
│   ├── test_guardrails_sell.py
│   ├── test_postures_differentiation.py
│   ├── test_stress_pricing_crunch_effect.py
│   └── test_stress_summary_pricing_kpis.py
│
├── configs/
│   └── stress_scenarios.yaml        # Escenarios stress (BCE/EBA)
│
├── docs/
│   └── pc10_pricing_kpis.md         # Documentación KPIs pricing
│
├── n8n/                             # Workflows n8n (opcional)
├── models/                          # Checkpoints PPO (.zip) + VecNormalize
├── logs/                            # Logs ejecución + TensorBoard
├── reports/                         # Outputs runtime (CSV/JSON/Excel/MD)
└── image/                           # Figuras del TFG
```

---

## 4. Módulos en detalle

### 4.1. Núcleo (`config.py`, `main.py`)

- **`config.py`** — Configuración única del proyecto. Define:
  - Parámetros financieros (tasas, plazos, BOOK_VALUE, coverage).
  - Parámetros regulatorios STD: pesos de riesgo (RW), `capital_ratio`, factores de severidad.
  - Hiperparámetros PPO (lr, n_steps, batch_size, gamma, gae, clip_range, ent_coef).
  - Posturas (`PRUDENCIAL`, `BALANCEADO`, `DESINVERSION`) y sus *bank strategies*.
  - Guardrails: PTI máximo, DSCR mínimo, threshold de *fire-sale*, mejora mínima EVA.
  - Rutas (modelos, logs, reports) y logging.
- **`main.py`** — Punto de entrada que orquesta el pipeline: generación → entrenamiento →
  inferencia coordinada → reporting.

### 4.2. Entornos RL (`env/`)

- **`env/loan_env.py`** — `LoanEnv` v6.4: entorno Gymnasium de **préstamo individual**.
  - Observación: vector de 10 features (PD, LGD, EAD, RW, DPD, secured, rating_num, segment_id,
    EVA_pre, viability_flag).
  - Acciones: `MANTENER` (0), `REESTRUCTURAR` (1), `VENDER` (2).
  - Reward NPL-aware: combina Δ EVA, capital liberado, viabilidad PTI/DSCR y penalización
    explícita de *fire-sale*.
- **`env/portfolio_env.py`** — `PortfolioEnv`: entorno macro de **cartera completa**.
  - Observación: 308 features (estadísticos agregados + distribuciones por segmento/rating).
  - Acciones: 12 acciones macro (combinaciones de intensidad sell/restruct y prioridad por segmento).
  - Reward agregada con restricciones de concentración y guardrails de cartera.

### 4.3. Agentes (`agent/`)

- **`train_subagents.py`** — Entrena por separado MICRO y MACRO con PPO (Stable-Baselines3),
  `VecNormalize`, callbacks de evaluación, *early stopping* y checkpointing audit-ready.
- **`train_agent.py`** — Variante genérica reutilizable para cualquier `gym.Env`.
- **`coordinator_inference.py`** — Inferencia coordinada bank-ready: combina la decisión MICRO
  (préstamo a préstamo) y la decisión MACRO (postura de cartera) mediante un arbitraje
  determinista por postura, con *fire-sale* robusto, contrafactuals coherentes, merge defensivo
  con la cartera original y esquema de salida estable.
- **`policy_inference.py`** — Inferencia MICRO con heurística financiera prioritaria + PPO como
  desempate y gates de viabilidad.
- **`policy_inference_coordinated.py`** — Decisión coordinada estilo analista IB con audit trail.
- **`policy_inference_portfolio.py`** — Inferencia MACRO con `VecNormalize` robusto por shape.

### 4.4. Optimización determinista (`optimizer/`)

- **`guardrails.py`** — Restricciones duras `check_restructure_constraints()` (PTI, DSCR, mejora
  EVA) y `check_sell_constraints()` (fire-sale, capital liberado, PnL vs book).
- **`restructure_optimizer.py`** — Optimizador híbrido (local + global) sobre rejilla
  *plazo × tipo × quita*, con *write-off* contra `BOOK_VALUE` y outputs auditables.
- **`price_simulator.py`** — Simulador de precio NPL: cálculo realista de precio en mercado
  secundario con *override* de *fire-sale*, triggers auditables y PnL contra book.

### 4.5. Riesgo (`risk/`)

- **`gates.py`** — Funciones puras, deterministas, libres de estado: `check_restruct_viability()`
  (DSCR mínimo) y triggers de fire-sale. Reutilizadas en todo el grafo de decisión.

### 4.6. Datos (`data/`)

- **`generate_portfolio.py`** — Generador de cartera sintética NPL bajo Basilea III STD:
  100% en default, RW vía mapping, PD forward 12–24 meses, métricas contables (book value,
  coverage, provisions) y de viabilidad (PTI/DSCR).
- **`ingest_portfolio.py`** — Ingesta de cartera real (Excel/CSV) con mapping YAML al esquema
  canónico (`mappings/real_portfolio_mapping.yaml`).
- **`sanity_generator_checks.py`** — Sanity checks sobre las distribuciones generadas.

### 4.7. Motores macro (`engines/`)

- **`stress_engine.py`** — Motor de stress multi-trimestre (BCE/EBA) que aplica shocks
  parametrizados a la cartera y reejecuta la coordinación completa.
- **`sensitivities_engine.py`** — Análisis de sensibilidad financiera sobre variables clave
  (PD, LGD, fire-sale threshold, rate).

### 4.8. Reporting y empaquetado (`reports/`)

- **`hardening.py`** — Capa post-proceso bank-ready: **Final Reasoning Seal**
  (`Reason_Code_Micro` / `Reason_Code_Macro` / `Reason_Code_Final` 100% consistentes con
  `Accion_final`), **Macro Steering** por postura (tasa de intervención ≥ 20%) y completitud
  de métricas post (sin NaN para MANTENER/REESTRUCTURAR).
- **`schema.py`** — `CANON_COLS` y `enforce_schema()` para estabilizar salidas.
- **`make_committee_pack.py`** — Empaqueta un committee pack reproducible con artefactos,
  logs, configuración y *manifest* (git hash, env, checksums SHA-256).
- **`backtesting_dynamic.py`** — Backtesting multi-trimestre: evolución de la cartera tras
  las decisiones (curación, redefault, KPIs por Q).
- **`backtesting_light.py`** — What-If sobre decisiones congeladas, sin re-entrenamiento.
- **`baseline_eval.py`, `evaluate_against_baselines{_sim}.py`** — RL vs Baseline.
- **`compare_postures.py`** — Comparativa cuantitativa entre las 3 posturas.
- **`export_styled_excel{,_summary}.py`** — Exportación Excel corporativo CIB con comentarios
  hover ejecutivos.
- **`export_financial_decisions.py`** — Informe financiero detallado por préstamo.
- **`npl_posture_analysis.py`** — Envelopes de negociación por postura + casos frontera.
- **`posture_kpi_report.py`** — KPIs ejecutivos por postura.
- **`results_summary.py`** — Consolidación RUN-LEVEL multi-summary.
- **`segment_dispersion_analysis.py`** — Análisis de dispersión por segmento y rating.
- **`theoretical_optimal_bound.py`** — Cota óptima teórica V* (DP exhaustiva micro + greedy
  rollout macro) y baselines random/hold.

### 4.9. Baselines (`baselines/`)

- **`baseline_policies.py`** — Políticas heurísticas deterministas (Prudencial / Desinversión)
  usadas como referencia frente al RL.

### 4.10. Multi-agente (`multiagent/`)

- **`coordinator_agent.py`** — Coordinador del sistema multi-agente (combina RestructAgent +
  PricingAgent con arbitraje).
- **`restruct_agent.py`, `pricing_agent.py`** — Wrappers ligeros que delegan en los
  optimizadores deterministas.

### 4.11. Dashboard (`dashboard/`)

- **`api_server.py`** — Backend Flask que sirve los outputs reales del proyecto como JSON al
  *frontend* (Lovable Matrix). Puerto 3000 por defecto.

### 4.12. Análisis (`analyze_learning.py`)

Análisis end-to-end del aprendizaje PPO: lectura de TensorBoard, evaluación de checkpoints,
distribución de la policy, diagnósticos de divergencia/colapso. Outputs en
`reports/learning_analysis/`.

---

## 5. Configuración (`config.py`, `configs/`)

- **`config.py`** — único punto de configuración runtime (parámetros financieros, regulatorios,
  PPO, posturas, guardrails, rutas).
- **`configs/stress_scenarios.yaml`** — escenarios de stress (BCE/EBA, shocks PD/LGD/rate).
- **`data/mappings/real_portfolio_mapping.yaml`** — mapping de cartera real → esquema canónico.

---

## 6. Instalación

```powershell
# 1. Clonar el repositorio
git clone <repo-url> optimizador_cartera_std
cd optimizador_cartera_std

# 2. Crear entorno virtual (Python 3.10+; testeado con 3.14)
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias (instalador inteligente, omite paquetes incompatibles con 3.14)
py install_requirements_smart.py
# alternativa:
.\install_requirements_smart.bat
```

Dependencias principales: `stable-baselines3`, `gymnasium`, `torch`, `numpy`, `pandas`,
`openpyxl`, `pyyaml`, `matplotlib`, `tensorboard`, `flask` (dashboard), `pytest` (tests),
`python-docx` (post-procesado de informes).

Para entornos CI ligeros: `pip install -r requirements-ci.txt`.

---

## 7. Ejecución — Scripts batch

| Batch                                  | Propósito                                                                  |
|----------------------------------------|----------------------------------------------------------------------------|
| `run_full_pipeline.bat`                | Pipeline completo: generación → entrenamiento → inferencia → reportes      |
| `run_pipeline.bat`                     | Pipeline estándar (sin re-entrenamiento full)                              |
| `run_inference_only.bat`               | Solo inferencia coordinada (usa checkpoints existentes)                    |
| `run_recalibrated_inference.bat`       | Inferencia tras recalibración                                              |
| `run_3_postures_executability.bat`     | Ejecuta las 3 posturas y verifica diferenciación                           |
| `train_all_postures.bat`               | Entrena posturas Prudencial / Balanceado / Desinversión                    |
| `train_pru_bal.bat`                    | Entrena posturas Prudencial + Balanceado                                   |
| `train_macro_pru_bal.bat`              | Entrena solo MACRO en Prudencial + Balanceado                              |
| `run_repro_test.bat`                   | Test de reproducibilidad                                                   |
| `smoke_test.bat`                       | Smoke tests rápidos (pytest)                                               |
| `ci_local.bat`                         | CI local (lint + tests + smoke)                                            |
| `install_requirements_smart.bat`       | Instalador inteligente de dependencias                                     |

---

## 8. Ejecución — comandos Python

```powershell
# 1) Generar cartera sintética NPL
py -m data.generate_portfolio

# 2) Entrenar subagentes (MICRO + MACRO) en una postura
py -m agent.train_subagents --posture PRUDENCIAL

# 3) Inferencia coordinada bank-ready
py -m agent.coordinator_inference --tag pc10_final

# 4) Stress multi-trimestre
py -m engines.stress_engine --tag pc10_final

# 5) Sensibilidades financieras
py -m engines.sensitivities_engine --tag pc10_final

# 6) Reporting consolidado
py -m reports.results_summary --tag pc10_final

# 7) Committee Pack reproducible
py -m reports.make_committee_pack --tag pc10_final

# 8) Comparativa de posturas
py -m reports.compare_postures --tag pc10_final

# 9) Dashboard local (Flask)
py dashboard/api_server.py
```

---

## 9. Tests

Suite `pytest` en `tests/`:

```powershell
pytest tests/                                # toda la suite
pytest tests/smoke_test.py                   # smoke
pytest tests/test_bank_ready_compliance.py   # cumplimiento PTI/DSCR/EVA
pytest tests/test_committee_pack.py          # empaquetado reproducible
pytest tests/test_coordinator_overrides.py   # arbitraje del coordinador
pytest tests/test_guardrails_*.py            # guardrails (sell/restruct/extended)
pytest tests/test_postures_differentiation.py
pytest tests/test_stress_*.py                # motor de stress
```

---

## 10. Outputs del pipeline

| Carpeta         | Contenido                                                                       |
|-----------------|---------------------------------------------------------------------------------|
| `models/`       | Checkpoints PPO (`.zip`) + `VecNormalize` (`.pkl`) por postura y por subagente  |
| `logs/`         | Logs de ejecución, TensorBoard (`logs/tb/`, `logs/tensorboard/`)                |
| `reports/`      | CSV/JSON/Excel/MD por run (`coordinated_inference_<tag>_*`, `stress_*`, ...)    |
| `image/`        | Figuras del TFG (incluye `anexo_g_tb/` con diagnósticos TensorBoard)            |
| `dashboard/`    | Backend + assets estáticos del dashboard                                        |

Cada `committee_pack_<tag>_<timestamp>/` incluye:

- Artefactos clave (CSV/Excel/MD consolidados).
- Logs filtrados.
- Snapshot de configuración (`config.py`, `mappings/`).
- `MANIFEST.json`: git hash, entorno Python, checksums SHA-256, timestamps.

---

## 11. Posturas

| Postura          | Filosofía                                                | Sesgo               |
|------------------|----------------------------------------------------------|---------------------|
| `PRUDENCIAL`     | Maximizar viabilidad; preferir REESTRUCTURAR             | Conservador         |
| `BALANCEADO`     | Trade-off EVA / capital / riesgo                         | Neutral             |
| `DESINVERSION`   | Maximizar capital liberado; preferir VENDER si Δ EVA ≈ 0 | Agresivo en ventas  |

El arbitraje entre MICRO y MACRO se rige por `COORDINATOR_PRIORITY` (`PRUDENCIAL_FIRST`,
`MACRO_FIRST`, ...) y por reglas deterministas auditables (ver
`tests/test_coordinator_overrides.py`).

---

## 12. Compliance bank-ready

Toda decisión final lleva trazabilidad completa:

- `Accion_micro`, `Accion_macro`, `Accion_final`, `Macro_Selected`, `Convergencia_Caso`.
- `Reason_Code_Micro`, `Reason_Code_Macro`, `Reason_Code_Final` (sellados por `hardening`).
- `Decision_Governance_Final` (regla aplicada en el arbitraje).
- Métricas post completas: `EVA_post`, `PTI_post`, `DSCR_post`, `Capital_Released`, `Price_to_EAD`,
  `Fire_Sale`, `fire_sale_threshold`.
- Guardrails: `check_restructure_constraints` y `check_sell_constraints` validan cada decisión
  antes de fijarla.

---

## 13. TFG y anexos

El documento final del TFG (`TFG_FINAL.docx` si está presente) contiene los anexos
metodológicos. En particular, el **Anexo G.3** documenta los diagnósticos TensorBoard (varianza
explicada, value loss, entropy loss y KL aproximada) de los entrenamientos PPO de los dos
subagentes, cuyas figuras se generan en `image/anexo_g_tb/`.

---

## 14. Cabeceras de los scripts

Todos los scripts Python del proyecto llevan al inicio una cabecera estandarizada:

```python
# -*- coding: utf-8 -*-
# ============================================================
# <ruta/relativa/al/script.py>
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: <descripción breve del propósito del script>
# ============================================================
```

(Se excluyen únicamente los snapshots inmutables dentro de `reports/committee_pack_*/`.)

---

## 15. Licencia y autoría

- **Autor:** José María Fernández-Ladreda Ballvé.
- **Contexto académico:** Trabajo de Fin de Grado.
- **Uso:** académico / demostrativo. No constituye asesoramiento financiero ni regulatorio.
