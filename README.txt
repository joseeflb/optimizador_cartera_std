=====================================================================
OPTIMIZADOR DE CARTERAS NPL (STD)
Basilea III STD · Banco L1.5 · RL Micro + Macro Coordinado
=====================================================================

Autor: José María Fernández-Ladreda Ballvé
Contexto: POC profesional (CIB / Risk / NPL desks)
Objetivo: Tomar decisiones expertas, justificadas y coherentes
          sobre carteras de préstamos en default (NPL)

=====================================================================
1. OBJETIVO DEL PROYECTO
=====================================================================

Este repositorio implementa un sistema completo y automatizado para
la optimización de carteras de préstamos en default (NPL) bajo el
Método Estándar de Basilea III.

El sistema está diseñado para comportarse como un ANALISTA EXPERTO
DE BANCA (orientado a decisiones reales), no como un clasificador naïve.

Principio clave:
    NO se decide únicamente por EVA negativa.
    Se decide ponderando (micro + macro):
        - Valor económico (EVA, RORWA, RONA)
        - Capital regulatorio (RWA, capital liberado, carry cost)
        - Riesgo y estabilidad (PD/LGD, volatilidad, estrés)
        - Precio realista de mercado secundario NPL (bid-side)
        - Opcionalidad temporal (mantener vs fire-sale)
        - Viabilidad real de reestructuración (PTI / DSCR / cure)
        - Concentración y diversificación (HHI por segmento/rating)
        - Coherencia con la postura del banco

El sistema soporta tres posturas coherentes con banca real:
    - PRUDENCIAL
    - BALANCEADO
    - DESINVERSION

=====================================================================
2. ESTRUCTURA DEL REPOSITORIO
=====================================================================

OPTIMIZADOR_CARTERA_STD/
│
├─ .github/
│  └─ workflows/
│     └─ smoke.yml                          (CI smoke: validación mínima)
│
├─ agent/
│  ├─ train_agent.py                        (entrenamiento micro / single-agent)
│  ├─ train_subagents.py                    (entrenamiento macro + subagentes)
│  ├─ policy_inference.py                   (inferencia micro por préstamo)
│  ├─ policy_inference_portfolio.py         (inferencia macro por cartera)
│  ├─ policy_inference_coordinated.py       (wrapper “micro↔macro” + guardrails)
│  └─ coordinator_inference.py              (decisión final coherente + explicación)
│
├─ data/
│  ├─ generate_portfolio.py                 (genera carteras sintéticas)
│  ├─ sanity_generator_checks.py            (checks del generador / smoke data)
│  └─ portfolio_synth_smoke.xlsx            (input demo / smoke)
│
├─ engines/
│  ├─ stress_engine.py                      (escenarios / estrés)
│  └─ sensitivities_engine.py               (sensibilidades paramétricas)
│
├─ env/
│  ├─ loan_env.py                           (Gym env micro por préstamo)
│  └─ portfolio_env.py                      (Gym env macro por cartera)
│
├─ multiagent/
│  ├─ pricing_agent.py                      (helper de venta / pricing)
│  ├─ restruct_agent.py                     (helper de reestructuración)
│  └─ coordinator_agent.py                  (lógica agentic de convergencia)
│
├─ optimizer/
│  ├─ price_simulator.py                    (simulador de precio NPL)
│  └─ restructure_optimizer.py              (motor determinista de restruct)
│
├─ reports/
│  ├─ schema.py                             (contrato de outputs / columnas)
│  ├─ results_summary.py                    (consolidado ejecutivo)
│  ├─ export_financial_decisions.py         (export “bank-ready” por postura)
│  ├─ export_styled_excel.py                (excel formateado detalle)
│  ├─ export_styled_excel_summary.py        (excel formateado resumen)
│  └─ runs/                                 (salidas por timestamp + postura)
│
├─ tests/
│  ├─ smoke_test.py                         (smoke e2e rápido)
│  ├─ diag_reward_loan.py                   (diagnóstico reward micro)
│  └─ diag_reward_portfolio.py              (diagnóstico reward macro)
│
├─ logs/                                    (logs de ejecución)
├─ models/                                  (artefactos: .zip, normalizers, orden features)
│
├─ config.py                                (gobierno del sistema)
├─ config_snapshot.json                     (snapshot reproducible)
├─ requirements-ci.txt                      (deps mínimas para CI smoke)
├─ install_requirements_smart.py/.bat       (instalación robusta en Windows)
├─ run_pipeline.bat                         (pipeline end-to-end en Windows)
├─ run_inference_only.bat                   (solo inferencia en Windows)
├─ smoke_test.bat                           (smoke local en Windows)
└─ main.py                                  (entrypoint / orquestador CLI)

=====================================================================
3. DIAGRAMA DE FLUJO DEL PIPELINE (MICRO ↔ MACRO)
=====================================================================

                 ┌──────────────────────────┐
                 │ data/generate_portfolio  │
                 │ (cartera de entrada)     │
                 └─────────────┬────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
┌─────────▼──────────┐                   ┌──────────▼──────────┐
│ env/loan_env.py     │                   │ env/portfolio_env.py │
│ Micro: préstamo     │                   │ Macro: cartera       │
│ individual NPL      │                   │ completa NPL         │
└─────────┬──────────┘                   └──────────┬──────────┘
          │                                         │
┌─────────▼──────────┐                   ┌──────────▼──────────┐
│ agent/train_agent.py│                   │ agent/train_subagents│
│ PPO micro (LoanEnv) │                   │ PPO macro / subagentes│
└─────────┬──────────┘                   └──────────┬──────────┘
          │                                         │
┌─────────▼──────────┐                   ┌──────────▼──────────┐
│ policy_inference.py │                   │ policy_inference_    │
│ Inferencia micro    │                   │ portfolio.py         │
└─────────┬──────────┘                   │ Macro + trayectoria   │
          │                               └──────────┬──────────┘
          └───────────────────────┬──────────────────┘
                                  │
                  ┌───────────▼──────────────────┐
                  │ policy_inference_coordinated │
                  │ (wrapper micro↔macro)        │
                  └───────────┬──────────────────┘
                              │
                  ┌───────────▼──────────────┐
                  │ coordinator_inference.py │
                  │ Decisión conjunta final  │
                  └───────────┬──────────────┘
                              │
                     ┌────────▼────────┐
                     │ reports / runs   │
                     │ + summary Excel  │
                     └─────────────────┘

Orquestación:
    - main.py centraliza el pipeline (y los .bat lo automatizan).
    - CI smoke: .github/workflows/smoke.yml ejecuta tests mínimos.

=====================================================================
4. CONCEPTO CLAVE: DECISIÓN EXPERTA MICRO + MACRO
=====================================================================

El sistema NO vende toda la cartera aunque:
    - PD ≈ 100%
    - RW ≈ 100–150%
    - EVA individual muy negativa

Porque un analista experto sabe que:
    - Vender NPL implica asumir una pérdida REAL vía precio de mercado
    - Mantener preserva opcionalidad (timing, recovery, workout)
    - Reestructurar puede transformar RW, PD y EVA futura
    - El capital tiene coste (carry), pero también tiene valor estratégico

Por ello, la decisión se construye así:

MICRO (LoanEnv):
    - ¿Este préstamo es reestructurable?
    - ¿Tiene sentido venderlo al precio esperado?
    - ¿Mantenerlo destruye o preserva valor esperado?

MACRO (PortfolioEnv):
    - ¿La cartera destruye valor agregado?
    - ¿El capital está mal asignado?
    - ¿Hay concentración o volatilidad excesiva?
    - ¿Qué bloque de préstamos conviene mover ahora?

COORDINACIÓN:
    - El macro decide QUÉ HACER (táctica agregada)
    - El micro decide A QUIÉN EXACTAMENTE (selección fina)
    - Si hay conflicto → prevalece la lógica prudencial (guardrails)

=====================================================================
5. DESCRIPCIÓN DE SCRIPTS (FUNCIONALIDAD + APORTE AL PIPELINE)
=====================================================================

5.1 config.py
--------------
Gobierno total del sistema.

Qué hace:
- Define parametría Basilea III STD (RW mapping, capital ratio, buffers).
- Define BankProfile: PRUDENCIAL / BALANCEADO / DESINVERSION.
- Deriva BankStrategy y RewardParams coherentes (micro y macro).

Qué aporta:
- Coherencia regulatoria y trazabilidad: una “fuente de verdad”
  para estrategia, penalizaciones, guardrails y umbrales.

Se puede fijar perfil vía:
    BANK_PROFILE=PRUDENCIAL | BALANCEADO | DESINVERSION

-------------------------------------------------

5.2 main.py
-----------
CLI / orquestador central del repositorio.

Qué hace:
- Ejecuta el flujo end-to-end (generate → train → infer → report/summary).
- Centraliza rutas, logging y snapshots (config_snapshot.json).

Qué aporta:
- Reproducibilidad operativa (misma ejecución = mismas salidas).
- Evita ejecuciones “a mano” dispersas.

-------------------------------------------------

5.3 run_pipeline.bat / run_inference_only.bat / smoke_test.bat
--------------------------------------------------------------
Wrappers Windows para acelerar la operación:

- run_pipeline.bat         → flujo completo
- run_inference_only.bat   → solo inferencia (si ya hay modelos)
- smoke_test.bat           → smoke local rápido

Qué aportan:
- Operación “1 click” en Windows, reduce errores de ejecución y paths.

-------------------------------------------------

5.4 install_requirements_smart.py / install_requirements_smart.bat
------------------------------------------------------------------
Instalación robusta de dependencias (entorno Windows).

Qué hace:
- Instala deps evitando conflictos típicos (pip/venv/paths).
- Homogeneiza entornos para ejecución y CI local.

Qué aporta:
- Reduce fricción de setup, mejora reproducibilidad.

-------------------------------------------------

5.5 requirements-ci.txt
-----------------------
Lista mínima de dependencias para CI smoke.

Qué aporta:
- CI rápido y estable (no requiere stack completo de entrenamiento pesado).

-------------------------------------------------

5.6 data/generate_portfolio.py
------------------------------
Genera una cartera sintética Basilea III STD:
    - Segmento, rating, PD, LGD, EAD, RW, RWA
    - Métricas económicas (EVA/RORWA/RONA) según parametría

Qué aporta:
- Fuente de datos estándar para entrenar, inferir y testear (smoke/demo).

-------------------------------------------------

5.7 data/sanity_generator_checks.py
-----------------------------------
Checks del generador (smoke data).

Qué hace:
- Valida rangos, consistencia de columnas y distribuciones básicas.

Qué aporta:
- Evita entrenar/inferir con datos corruptos → reduce resultados “sin sentido”.

-------------------------------------------------

5.8 env/loan_env.py
-------------------
Entorno RL micro para un préstamo NPL individual.

Acciones:
    0 = MANTENER
    1 = REESTRUCTURAR
    2 = VENDER

Observación:
    vector de features (ordenado por models/feature_order.json)

Simula:
    - Deterioro real en default
    - Reestructuración factible (PTI / DSCR / cure)
    - Venta con impacto económico real (precio + costes)

Qué aporta:
- “Laboratorio controlado” donde el PPO aprende la decisión por préstamo.

-------------------------------------------------

5.9 env/portfolio_env.py
------------------------
Entorno macro de cartera NPL.

Qué hace:
- Define estado agregado (EVA_total, RWA_total, riesgo, HHI, volatilidad, carry).
- Soporta acciones agregadas (macro) y re-ranking con micro (si aplica).

Qué aporta:
- Evita optimizar “miopes” por préstamo: controla concentración y capital.

-------------------------------------------------

5.10 optimizer/restructure_optimizer.py
--------------------------------------
Motor determinista de reestructuración.

Qué hace:
- Grid de plazo / tasa / quita + restricciones PTI / DSCR.
- Estima mejora PD/LGD/EVA post y detecta cure.

Qué aporta:
- Núcleo de “workout” realista para que la acción REESTRUCTURAR sea bancaria.

-------------------------------------------------

5.11 optimizer/price_simulator.py
--------------------------------
Simulador realista de precio NPL (bid-side).

Qué hace:
- Pricing con escenarios / costes transaccionales / P&L realista.
- Modela capital liberado por venta.

Qué aporta:
- La acción VENDER deja de ser “mágica”: tiene pérdida económica creíble.

-------------------------------------------------

5.12 engines/stress_engine.py
-----------------------------
Motor de estrés (escenarios macro).

Qué aporta:
- Robustez: valida decisiones bajo shocks (PD/LGD/spreads), útil para comité.

-------------------------------------------------

5.13 engines/sensitivities_engine.py
------------------------------------
Motor de sensibilidades (shocks paramétricos).

Qué aporta:
- Atribución: identifica qué inputs mueven la decisión/reward.

-------------------------------------------------

5.14 agent/train_agent.py
-------------------------
Entrenamiento del agente PPO micro (LoanEnv) o single-agent.

Qué hace:
- Entrena política micro con reward definido en config.py.
- Guarda artefactos (modelos + normalización si aplica).

Qué aporta:
- Capacita la selección fina A QUIÉN aplicar la acción en cartera.

-------------------------------------------------

5.15 agent/train_subagents.py
-----------------------------
Entrenamiento del stack macro (PortfolioEnv) y subagentes.

Qué hace:
- Entrena política macro y/o componentes auxiliares.
- Alinea reward macro con estrategia (PRUDENCIAL/BALANCEADO/DESINVERSION).

Qué aporta:
- Decide QUÉ HACER a nivel cartera, evitando decisiones inconexas préstamo a préstamo.

-------------------------------------------------

5.16 agent/policy_inference.py
------------------------------
Inferencia micro por préstamo.

Qué hace:
- Evalúa cada préstamo en LoanEnv (con normalización consistente si existe).
- Devuelve acción + explicación (drivers económicos/prudenciales).

Qué aporta:
- Auditoría por caso: “por qué este préstamo”.

-------------------------------------------------

5.17 agent/policy_inference_portfolio.py
----------------------------------------
Inferencia macro experta.

Qué hace:
- Genera trayectoria macro, objetivos y selección de bloques.
- Aplica capa prudencial explícita y usa PPO como refinamiento táctico.

Qué aporta:
- Coherencia agregada: capital, concentración, timing (evita fire-sale irracional).

-------------------------------------------------

5.18 agent/policy_inference_coordinated.py
------------------------------------------
Wrapper de coordinación micro↔macro (pre-coordinator).

Qué hace:
- Alinea salidas micro y macro antes de la decisión final.
- Aplica guardrails y reglas de desempate consistentes.

Qué aporta:
- Reduce contradicciones y “saltos” entre recomendaciones micro y macro.

-------------------------------------------------

5.19 agent/coordinator_inference.py
-----------------------------------
Pieza CLAVE del sistema: decisión conjunta final.

Qué hace:
- Coordina micro + macro, resuelve conflictos y produce UNA decisión final.
- Construye explicación adaptada al préstamo y al estado global de cartera.

Qué aporta:
- Convierte el sistema en “analista experto” (no un ensemble caótico).

-------------------------------------------------

5.20 multiagent/pricing_agent.py
--------------------------------
Helper modular para venta/pricing.

Qué aporta:
- Encapsula lógica de pricing para reutilización en policies y reporting.

-------------------------------------------------

5.21 multiagent/restruct_agent.py
---------------------------------
Helper modular para reestructuración.

Qué aporta:
- Encapsula lógica de workout (restricciones y outputs) para consistencia.

-------------------------------------------------

5.22 multiagent/coordinator_agent.py
------------------------------------
Lógica agentic de convergencia.

Qué aporta:
- Permite extender coordinación con más “roles” (risk, collections, IC) sin
  reescribir el core.

-------------------------------------------------

5.23 reports/schema.py
----------------------
Contrato de outputs (columnas, nombres, semántica).

Qué aporta:
- “Bank-ready”: evita exports inconsistentes y facilita QA + trazabilidad.

-------------------------------------------------

5.24 reports/export_financial_decisions.py
------------------------------------------
Export principal “bank-ready” por postura.

Qué hace:
- Construye ficheros de salida por postura (PRUDENCIAL/BALANCEADO/DESINVERSION).
- Asegura columnas: micro, macro, decisión final + argumentación.

Qué aporta:
- Deliverable directo para comité / IC / validación interna.

-------------------------------------------------

5.25 reports/export_styled_excel.py
-----------------------------------
Excel de detalle formateado (por préstamo).

Qué aporta:
- Lectura humana (negocio) sin tocar datos → reduce fricción con stakeholders.

-------------------------------------------------

5.26 reports/export_styled_excel_summary.py
-------------------------------------------
Excel de resumen formateado (KPIs + agregados).

Qué aporta:
- “One-pager” ejecutivo: KPIs, mix de decisiones, impacto EVA/RWA, etc.

-------------------------------------------------

5.27 reports/results_summary.py
-------------------------------
Consolidado ejecutivo (métricas, charts, comparativas).

Qué aporta:
- Cierra el loop de reporting y permite comparar posturas.

-------------------------------------------------

5.28 tests/smoke_test.py
------------------------
Smoke test “end-to-end” (rápido).

Qué hace:
- Valida imports, rutas, generación básica de outputs sin entrenamiento pesado.

Qué aporta:
- Garantía mínima de integridad del pipeline (local y CI).

Ejecución:
    python -m tests.smoke_test

-------------------------------------------------

5.29 tests/diag_reward_loan.py
------------------------------
Diagnóstico del reward micro.

Qué aporta:
- Detecta degeneración del reward (p.ej., todo vende / todo mantiene).

-------------------------------------------------

5.30 tests/diag_reward_portfolio.py
-----------------------------------
Diagnóstico del reward macro.

Qué aporta:
- Valida que macro no empuja decisiones triviales o incoherentes con estrategia.

-------------------------------------------------

5.31 .github/workflows/smoke.yml
--------------------------------
CI smoke (matriz OS/Python) para asegurar reproducibilidad mínima.

Qué hace:
- Instala deps con requirements-ci.txt
- Ejecuta: python -m tests.smoke_test

Qué aporta:
- “No se rompe en main”: control mínimo de calidad continuo.

=====================================================================
6. RESULTADO FINAL
=====================================================================

El output del sistema es:

- Automatizado
- Auditable
- Coherente con banca real
- Explicado caso a caso
- Adaptado a la estrategia del banco (3 posturas)

NO es un modelo académico.
NO es un clasificador simple.
ES un motor de decisión experto para carteras NPL.

Definición de “DONE” (tarea completada):
    1) Smoke pasa en local y en CI (Windows + Ubuntu).
    2) Inferencia genera outputs completos para 3 posturas:
       - decisiones_finales_prudencial.xlsx
       - decisiones_finales_balanceado.xlsx
       - decisiones_finales_desinversion.xlsx
    3) Semántica de acciones coherente extremo a extremo
       (env → policy → coordinator → reporting).
    4) Normalización (VecNormalize) consistente en training e inference
       (sin mismatches de shape / orden de features).
    5) Reporting trazable:
       - Auditoría por préstamo (CSV)
       - Racional económico-prudencial en Excel
       - (Requisito bank-ready) carpeta con “1 Excel por decisión/préstamo”
         separando columnas micro/macro/decisión final + argumentación.
         *Si aún no está generado automáticamente, se considera PENDIENTE.*

=====================================================================
FIN DEL README
=====================================================================
