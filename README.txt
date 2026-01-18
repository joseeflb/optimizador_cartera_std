=====================================================================
OPTIMIZADOR DE CARTERAS NPL
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
DE BANCA DE INVERSIÓN, no como un clasificador naïve.

Principio clave:
    NO se decide únicamente por EVA negativa.
    Se decide ponderando:
        - Valor económico (EVA, RORWA, RONA)
        - Capital regulatorio (RWA, capital liberado, carry cost)
        - Riesgo y estabilidad
        - Precio real de mercado secundario NPL
        - Opcionalidad temporal (mantener vs fire-sale)
        - Viabilidad real de reestructuración (PTI / DSCR)
        - Concentración y volatilidad de la cartera

El sistema soporta tres estrategias coherentes con banca real:
    - PRUDENTE
    - BALANCEADO
    - DESINVERSION

=====================================================================
2. ESTRUCTURA DEL REPOSITORIO
=====================================================================

OPTIMIZADOR_CARTERA_NPL/
│
├─ agent/
│  ├─ coordinator_inference.py
│  ├─ policy_inference.py
│  ├─ policy_inference_portfolio.py
│  ├─ policy_inference_coordinated.py
│  ├─ train_agent.py
│  └─ train_subagents.py
│
├─ data/
│  ├─ generate_portfolio.py
│  ├─ portfolio_synth.xlsx
│  └─ portfolio_synth_smoke.xlsx
│
├─ engines/
│  ├─ sensitivities_engine.py
│  └─ stress_engine.py
│
├─ env/
│  ├─ loan_env.py
│  └─ portfolio_env.py
│
├─ multiagent/
│  ├─ coordinator_agent.py
│  ├─ pricing_agent.py
│  └─ restruct_agent.py
│
├─ optimizer/
│  ├─ price_simulator.py
│  └─ restructure_optimizer.py
│
├─ reports/
│  └─ (salidas de inferencia por timestamp)
│  ├─ export_financial_decisions.py
│  ├─ export_styled_excel.py
│  ├─ export_styled_excel_summary.py
│  ├─ results_summary.py
│  ├─ training_evaluations.csv
│  └─ training_evolution.png
│
├─ logs/
├─ models/
├─ config.py
├─ config_snapshot.json
├─ install_requirements_smart.py
├─ install_requirements_smart.bat
└─ main.py

=====================================================================
3. DIAGRAMA DE FLUJO DEL PIPELINE (MICRO ↔ MACRO)
=====================================================================

                 ┌──────────────────────────┐
                 │        data/generate_portfolio  │
                 │        (cartera de entrada)     │
                 └─────────────┬────────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
┌─────────▼──────────┐                   ┌──────────▼──────────┐
│ env/loan_env.py          │                   │  env/portfolio_env.py     │
│ Micro: préstamo          │                   │       Macro: cartera      │
│ individual NPL           │                   │       completa NPL        │
└─────────┬──────────┘                   └──────────┬──────────┘
             │                                               │
┌─────────▼──────────┐                   ┌──────────▼──────────┐
│ train_agent.py           │                   │       train_subagents.py  │
│ PPO micro (LoanEnv)      │                   │       PPO macro / agentes │
└─────────┬──────────┘                   └──────────┬──────────┘
             │                                                │
┌─────────▼──────────┐                   ┌──────────▼──────────┐
│ policy_inference.py      │                   │       policy_inference_   │
│ Inferencia micro         │                   │       portfolio.py        │
└─────────┬──────────┘                   │       Macro + re-ranking  │
             │                                 │       con PPO micro       │
             │                                  └─────────┬──────────┘
             └────────────────┬────────────────────┘
                                  │
                  ┌───────────▼──────────────┐
                  │        coordinator_inference.py │
                  │        Decisión conjunta final  │
                  └───────────┬──────────────┘
                                 │
                     ┌────────▼────────┐
                     │     reports / summary│
                     │     CSV + Excel      │
                     └─────────────────┘

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
    - El macro decide QUÉ HACER
    - El micro decide A QUIÉN EXACTAMENTE
    - Si hay conflicto → prevalece la lógica prudencial

=====================================================================
5. DESCRIPCIÓN DE SCRIPTS (FUNCIONALIDAD)
=====================================================================

5.1 config.py
--------------
Gobierno total del sistema.

- Basilea III STD (RW mapping)
- Regulación (hurdle, capital ratio)
- BankProfile: PRUDENTE / BALANCEADO / DESINVERSION
- BankStrategy: pesos, umbrales, penalizaciones
- RewardParams derivados automáticamente
- Parámetros micro y macro coherentes

Se puede fijar perfil vía:
    BANK_PROFILE=PRUDENTE | BALANCEADO | DESINVERSION

-------------------------------------------------

5.2 env/loan_env.py
-------------------
Entorno RL micro para un préstamo NPL individual.

Acciones:
    0 = MANTENER
    1 = REESTRUCTURAR
    2 = VENDER

Observación (10 features):
    [EAD, PD, LGD, RW, EVA, RONA, RORWA,
     rating_num, segmento_id, DPD/30]

Simula:
    - Deterioro real en default
    - Reestructuración factible (PTI / DSCR)
    - Venta con impacto económico real

-------------------------------------------------

5.3 optimizer/restructure_optimizer.py
--------------------------------------
Motor determinista de reestructuración.

- Grid de plazo / tasa / quita
- Restricciones PTI / DSCR
- Mejora PD / LGD / EVA post
- Identificación de cure

-------------------------------------------------

5.4 optimizer/price_simulator.py
--------------------------------
Simulador realista de precio NPL.

- Monte Carlo multi-escenario
- Bid-side market pricing
- Costes transaccionales
- P&L real vs recovery IFRS
- Capital liberado (RWA × ratio)

-------------------------------------------------

5.5 env/portfolio_env.py
------------------------
Entorno macro de cartera NPL.

- Acciones agregadas (12)
- Re-ranking de candidatos usando PPO micro
- Métricas:
    EVA_total, RWA_total, riesgo,
    HHI (segmento/rating),
    volatilidad EVA, carry cost
- Reward macro económico y prudencial

-------------------------------------------------

5.6 policy_inference_portfolio.py
---------------------------------
Inferencia macro experta.

- Capa financiera prudencial explícita
- PPO actúa como refinamiento táctico
- Soporta:
    prudencial / balanceado / desinversion
- Exporta:
    trayectoria, cartera final, summary

-------------------------------------------------

5.7 policy_inference.py
-----------------------
Inferencia micro por préstamo.

- Decisión explicada por caso
- Compatible con LoanEnv
- Exportable a reporting financiero

-------------------------------------------------

5.8 coordinator_inference.py
----------------------------
Pieza CLAVE del sistema.

- Coordina micro + macro
- Resuelve conflictos
- Produce UNA decisión final coherente
- Explicación adaptada a cada préstamo
  y al estado global de la cartera

Conceptualmente:
    SÍ → es la capa que convierte el sistema
         en un “analista experto”.

-------------------------------------------------

5.9 multiagent/
---------------
Arquitectura agentic modular:

- pricing_agent.py      → venta
- restruct_agent.py     → reestructuración
- coordinator_agent.py  → lógica de convergencia

-------------------------------------------------

5.10 summary/
-------------
Reporting listo para negocio:

- Excels formateados
- KPIs agregados
- Resultados de entrenamiento
- Outputs para comité / IC

=====================================================================
6. RESULTADO FINAL
=====================================================================

El output del sistema es:

- Automatizado
- Auditable
- Coherente con banca real
- Explicado caso a caso
- Adaptado a la estrategia del banco

NO es un modelo académico.
NO es un clasificador simple.
ES un motor de decisión experto para carteras NPL.

=====================================================================
FIN DEL README
=====================================================================

