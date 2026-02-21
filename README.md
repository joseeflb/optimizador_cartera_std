**POC — OPTIMIZADOR DE CARTERAS EN DEFAULT**
[![CI smoke](https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml)
Banco L1.5 · Basilea III Método Estándar · PC10 (feat/pc10-pricing-kpis-stress)



**======================================================================**
**ÍNDICE**
**======================================================================**

&nbsp;&nbsp; 1. Visión general y flujo del pipeline
&nbsp;&nbsp; 2. Configuración global
&nbsp;&nbsp; 3. Flujo paso a paso — qué hace cada script
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.1 Generación de cartera sintética
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.2 Ingesta de cartera real
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.3 Motores financieros (cerebro financiero)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.4 Entornos de aprendizaje por refuerzo
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.5 Entrenamiento de agentes RL
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.6 Inferencia micro (préstamo a préstamo)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.7 Inferencia macro (cartera completa)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.8 Inferencia coordinada multi-agente
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.9 Motor de estrés multi-escenario
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.10 Comparativa de posturas y backtesting
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3.11 Committee pack
&nbsp;&nbsp; 4. Resultados: dónde encontrarlos y qué se ve en cada uno
&nbsp;&nbsp; 5. Escenarios y posturas de riesgo
&nbsp;&nbsp; 6. Scripts de automatización y CI
&nbsp;&nbsp; 7. Estructura de carpetas completa
&nbsp;&nbsp; 8. Trazabilidad y reproducibilidad



**======================================================================**
**1. VISIÓN GENERAL Y FLUJO DEL PIPELINE**
**======================================================================**



El proyecto implementa un pipeline completo para optimizar carteras en default (NPL) bajo Basilea III método estándar. Combina motores financieros deterministas con agentes de aprendizaje por refuerzo (RL) y una capa de overrides prudenciales, produciendo decisiones explicables y trazables para comité.

El flujo de extremo a extremo es el siguiente:

```
[1] DATOS
    data/generate_portfolio.py   <- genera cartera sintética (Excel)
    data/ingest_portfolio.py     <- ingiere cartera real del banco (Excel/CSV)
         |
         v
[2] CONFIGURACIÓN GLOBAL
    config.py                    <- parámetros regulatorios, RL, motores, rutas
    configs/stress_scenarios.yaml<- definición de los 4 escenarios macro
         |
         v
[3] ENTRENAMIENTO (solo primera vez o cuando cambia la cartera)
    agent/train_subagents.py     <- entrena PPO micro (LoanEnv) y PPO macro (PortfolioEnv)
      via main.py train
         |
         v
[4] INFERENCIA COORDINADA (producción)
    agent/coordinator_inference.py   <- por postura (prudencial / balanceado / desinversion)
    multiagent/coordinator_agent.py  <- lógica jerárquica macro + micro + overrides
    multiagent/restruct_agent.py     <- motor de reestructuración
    multiagent/pricing_agent.py      <- motor de pricing NPL
         |
         v
[5] ESTRÉS MULTI-ESCENARIO
    engines/stress_engine.py     <- 4 escenarios x 3 posturas = 12 ejecuciones
         |
         v
[6] REPORTING Y VALIDACIÓN
    reports/compare_postures.py  <- comparativa de las 3 posturas (monotonía)
    reports/backtesting_light.py <- PPO vs baselines de referencia
    reports/make_committee_pack.py <- empaquetado auditable para comité
```

**Orquestador principal:** `main.py` centraliza los subcomandos `generate / train / infer / summary`.
Para una ejecución completa desatendida, se utiliza `ci_local.bat`.



**======================================================================**
**2. CONFIGURACIÓN GLOBAL**
**======================================================================**



**2.1. config.py — Núcleo común del proyecto**

Todos los módulos importan de `config.py`. Es la única fuente de verdad para parámetros regulatorios, RL, motores y rutas. Cambiar un valor aquí afecta a toda la cadena automáticamente.

Bloques principales:

* **Regulacion**: hurdle RAROC, CET1/Capital mínimos, buffers Basilea III.
* **BaselSTDMapping**: ponderaciones de riesgo (RW) por segmento, rating y estado (performing/default).
* **ReestructuraParams**: rejillas de plazo, tasa y quita; umbrales PTI/DSCR máximos.
* **SensibilidadReestructura**: reducción de PD/LGD tras reestructuración, ventana de curación, RW "cured".
* **PrecioVentaParams**: haircuts por segmento, rango bid, coste legal, horizonte de recuperación.
* **RewardParams**: pesos del reward RL (EVA, capital, HHI, P&L ventas, penalizaciones fire-sale).
* **PPOParams**: hiperparámetros del agente (lr, batch_size, n_steps, gamma, clip_range, ent_coef).
* **EnvParams**: dimensiones de estado, top-k, semillas globales.
* **BID_HAIRCUT_GLOBAL**: multiplicador inyectado en el motor de pricing en el escenario `pricing_crunch`.
* Rutas estándar: DATA_PATH, MODELS_PATH, REPORTS_PATH, LOGS_PATH, CHECKPOINTS_PATH.

**2.2. configs/stress_scenarios.yaml — Los 4 escenarios macro**

* **baseline**: sin shocks. Estado de referencia.
* **mild**: deterioro moderado. PD×1.2, LGD+0.05, RW×1.1, colateral×0.95.
* **severe**: recesión severa. PD×1.5, LGD+0.15, RW×1.25, colateral×0.85, rates+200bps.
* **pricing_crunch**: crisis de liquidez. bid_haircut×1.3 -> precios NPL 30% más bajos, más ventas bloqueadas.

**2.3. data/mappings/real_portfolio_mapping.yaml**

Traduce columnas del fichero bruto del banco al esquema interno del pipeline. Lo usa `data/ingest_portfolio.py` para transformar la cartera real con tipado, renombrado y conversión de unidades.



**======================================================================**
**3. FLUJO PASO A PASO — QUÉ HACE CADA SCRIPT**
**======================================================================**



---

**3.1. data/generate_portfolio.py — Generación de cartera sintética**

**Qué aporta:** Crea una cartera de préstamos NPL sintética con distribuciones realistas de segmentos, ratings, EAD, PD y LGD. Sirve para desarrollo, pruebas y CI cuando no se dispone de cartera real.

Función principal: `generate_portfolio(n)` -> DataFrame; `export_excel(df, out_path)` -> Excel corporativo.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`python main.py generate --n 1000 --out data/portfolio_synth.xlsx`

**Salidas:**

&nbsp;&nbsp;· `data/portfolio_synth.xlsx` — cartera completa para entrenamiento e inferencia.
&nbsp;&nbsp;· `data/portfolio_synth_smoke.xlsx` — cartera reducida para smoke tests.

---

**3.2. data/ingest_portfolio.py — Ingesta de cartera real**

**Qué aporta:** Transforma la cartera real del banco al esquema interno del pipeline, aplicando el mapeo de `data/mappings/real_portfolio_mapping.yaml`. Realiza coerciones de tipo, validaciones y produce un DataFrame directamente consumible por el pipeline (entrenamiento o inferencia).

No tiene CLI propio; se importa como módulo desde `coordinator_inference`, `train_subagents` o en scripts de análisis ad hoc.

---

**3.3. Motores financieros (cerebro financiero)**

Los motores no dependen del RL: son lógica financiera determinista que el pipeline usa en cada decisión.

**optimizer/restructure_optimizer.py — Motor de reestructuración**

**Qué aporta:** Dado un préstamo en default, encuentra la combinación óptima de (plazo adicional, nueva tasa, quita sobre EAD) que maximiza el EVA post-reestructuración. Modela flujos de caja (sistema francés), asequibilidad vía PTI/DSCR, dinámica PD/LGD tras reestructuración y capital/provisiones a lo largo del horizonte.

Función principal: `optimize_restructure(loan_dict)` -> diccionario con `plazo_optimo, tasa_nueva, quita, EVA_post, EVA_gain, RWA_post, RORWA_post, PTI, DSCR`.

**optimizer/price_simulator.py — Motor de pricing NPL**

**Qué aporta:** Simula el precio de mercado para una venta NPL (bid price), calcula el P&L realizado, el capital liberado y los percentiles de precio. El parámetro `BID_HAIRCUT_GLOBAL` amplifica haircuts en el escenario `pricing_crunch`. Genera `price_ratio_ead` (bid/EAD), columna canónica para el KPI `avg_bid_pct_ead` del PC10.

Función principal: `simulate_npl_price(loan_dict)` -> `{precio_optimo, pnl, capital_liberado, price_ratio_ead, p5, p50, p95, ...}`.

**engines/stress_engine.py — Motor de estrés multi-escenario**

**Qué aporta:** Orquesta las 12 ejecuciones (4 escenarios × 3 posturas). Por cada combinación: aplica shocks macro a la cartera, inyecta `BID_HAIRCUT_GLOBAL` si corresponde, lanza `coordinator_inference` para las 3 posturas, lee los CSVs de auditoría y calcula los KPIs de ventas PC10. Consolida todo en `stress_summary_<tag>.csv`.

Función orquestadora: `run_stress_pipeline(tag, scenarios, postures)`.

**engines/sensitivities_engine.py — Motor de sensitividades**

**Qué aporta:** Calcula derivadas parciales de NI, EVA, RWA y capital respecto a PD, LGD, rate, RW y EAD. Genera un `global_sensitivity_score(loan)` que identifica los préstamos "palanca" (alta sensibilidad a mejoras). Se usa para ranking en PortfolioEnv y para explicabilidad en la inferencia.

---

**3.4. Entornos de aprendizaje por refuerzo (RL)**

Los entornos son el "mundo simulado" donde los agentes RL aprenden a optimizar decisiones.

**env/loan_env.py — Entorno micro (nivel préstamo)**

**Qué aporta:** Simula la evolución de un préstamo individual. El agente puede elegir MANTENER, REESTRUCTURAR o VENDER. La observación es un vector de 10 features (EAD, PD, LGD, RW, DPD, spread, segmento, ingreso, cashflow, LTV). La recompensa combina ΔEVA, capital liberado, estabilidad y prudencia regulatoria.

Se usa para entrenar el agente micro (PPO loan) y para inferencia loan-level en `policy_inference.py`.

**env/portfolio_env.py — Entorno macro (nivel cartera)**

**Qué aporta:** Simula decisiones agregadas sobre una cartera NPL completa. El estado es un vector de indicadores de cartera (EVA total, RWA, HHI, volatilidad EVA, capital liberado acumulado). El espacio de acciones (0–11) incluye: mantener todos, vender/reestructurar top-K por EVA/RORWA/PTI, combinaciones, y heurística STD.

Internamente implementa `_apply_maintain` (drift NPL), `_apply_restructure` y `_apply_sell` (llama a `simulate_npl_price` y calcula `pnl_realized`, `price_ratio_ead`, capital liberado).

Se usa para entrenar el agente macro (PPO portfolio) y para inferencia macro en `policy_inference_portfolio.py` y `coordinator_inference.py`.

---

**3.5. agent/train_subagents.py — Entrenamiento de agentes RL**

**Qué aporta:** Entrena el subagente micro (en LoanEnv) y el subagente macro (en PortfolioEnv) usando PPO de Stable-Baselines3. Guarda los modelos entrenados y los normalizadores VecNormalize. Sin estos modelos, no hay inferencia.

Funciones clave:

* `train_loan_agent(...)` -> entrena en LoanEnv -> guarda `models/best_model.zip`, `models/vecnormalize_final.pkl`.
* `train_portfolio_agent(...)` -> entrena en PortfolioEnv -> guarda `models/best_model_portfolio.zip`.
* Genera checkpoints periódicos en `models/checkpoints/` y logs para TensorBoard en `logs/tb/` y `logs/tensorboard/`.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`python main.py train --agent both --portfolio data/portfolio_synth.xlsx --total-steps 500000 --top-k 5 --scenario baseline`

O directamente:

&nbsp;&nbsp;&nbsp;&nbsp;`python -m agent.train_subagents --agent loan --portfolio data/portfolio_synth.xlsx`
&nbsp;&nbsp;&nbsp;&nbsp;`python -m agent.train_subagents --agent portfolio --portfolio data/portfolio_synth.xlsx`

**agent/train_agent.py — Entrenamiento avanzado (alternativo)**

**Qué aporta:** Versión avanzada con `BusinessEvalCallback` (early stopping basado en EVA + capital liberado), integración completa de VecNormalize en train y eval, y logs estructurados para TensorBoard.

---

**3.6. agent/policy_inference.py — Inferencia micro (loan-level)**

**Qué aporta:** Aplica el agente PPO micro préstamo a préstamo sobre la cartera de entrada. Para cada préstamo reconstruye la observación en el orden canónico de `FEATURE_COLUMNS`, obtiene la acción del agente y calcula métricas post-decisión (EVA_post, RWA_post, capital_liberado, pnl). Genera un resumen con trazabilidad completa (Explain_Steps, ΔEVA).

Flujo:

1. Carga `best_model.zip` + `vecnormalize_final.pkl`.
2. Carga cartera desde Excel o CSV.
3. Para cada préstamo: predice acción (MANTENER/REESTRUCTURAR/VENDER) + calcula métricas.
4. Exporta `summary.csv` y `decisiones_explicadas.xlsx`.

Log: `logs/policy_inference.log`.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`python main.py infer --model models/best_model.zip --vecnorm models/vecnormalize_final.pkl --portfolio data/portfolio_synth.xlsx`

---

**3.7. agent/policy_inference_portfolio.py — Inferencia macro (cartera completa)**

**Qué aporta:** Aplica el agente PPO macro sobre la cartera completa durante n_steps. En cada paso predice la acción macro (qué palanca aplicar a qué subgrupo de préstamos), aplica `env.step(action)` y recoge métricas agregadas (EVA total, RWA, capital, reward). Produce un historial de trayectoria y un resumen final.

Flujo:

1. Carga `best_model_portfolio.zip`.
2. Crea `PortfolioEnv(loans_df=df, top_k)`.
3. Ejecuta n_steps de inferencia y registra métricas por paso.
4. Exporta `trajectory.csv`, `summary_portfolio.csv`, `portfolio_final.xlsx`.

Log: `logs/policy_inference_portfolio.log`.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`python -m agent.policy_inference_portfolio --portfolio data/portfolio_synth.xlsx --tag mi_run`

---

**3.8. Inferencia coordinada multi-agente**

Este es el modo de producción del pipeline. Combina los tres agentes y los overrides prudenciales en una sola ejecución por postura.

**multiagent/coordinator_agent.py — Agente coordinador**

**Qué aporta:** Implementa la jerarquía de decisión completa en el método `decide(portfolio_obs, loan_dict)`:

1. Obtiene `act_macro` del PPO macro (qué acción aplicar a la cartera).
2. Calcula sugerencia de reestructuración micro (RestructAgent).
3. Calcula sugerencia de pricing micro (PricingAgent).
4. Aplica overrides prudenciales Banco L1.5:
&nbsp;&nbsp;&nbsp;&nbsp;· Evita vender préstamos con EVA_post alto (buenos préstamos).
&nbsp;&nbsp;&nbsp;&nbsp;· Fuerza reestructuración si EVA_gain muy alto y PTI aceptable.
&nbsp;&nbsp;&nbsp;&nbsp;· Bloquea fire-sales si precio < floor prudencial.
5. Devuelve: `accion_final` (MANTENER/REESTRUCTURAR/VENDER) + `razon` (lista explicativa).

**multiagent/restruct_agent.py — Agente de reestructuración**

**Qué aporta:** Wrapper de `optimize_restructure`. Normaliza entradas del préstamo y devuelve diccionario homogéneo con EVA_post, EVA_gain, RWA_post, PTI, DSCR, plazo_optimo, tasa_nueva, quita, ok.

**multiagent/pricing_agent.py — Agente de pricing**

**Qué aporta:** Wrapper de `simulate_npl_price`. Construye argumentos robustos desde el préstamo y devuelve precio_optimo, pnl, capital_liberado, p5, p50, p95, price_ratio_ead, ok.

**agent/coordinator_inference.py — Script central de inferencia por postura**

**Qué aporta:** Ejecuta la inferencia coordinada completa para una postura concreta. Es el script de producción que llama el stress engine para cada una de las 12 combinaciones (escenario × postura).

Flujo:

1. Carga configuración de postura + modelos (PPO macro, PPO micro, VecNormalize).
2. Instancia `CoordinatorAgent`.
3. Para cada préstamo: decide vía `coordinator_agent.decide(...)`, registra métricas completas.
4. Exporta cuatro ficheros en `reports/coordinated_inference_<tag>_<timestamp>_<postura>/`:
&nbsp;&nbsp;&nbsp;&nbsp;· `decisiones_finales_<postura>.xlsx` — Excel con todas las decisiones y métricas.
&nbsp;&nbsp;&nbsp;&nbsp;· `overrides_log_<postura>.csv` — log de overrides prudenciales aplicados.
&nbsp;&nbsp;&nbsp;&nbsp;· `portfolio_kpis_<postura>.json` — KPIs agregados de la cartera.
&nbsp;&nbsp;&nbsp;&nbsp;· `decisiones_audit_<postura>.csv` — CSV loan-by-loan para cálculo de KPIs PC10.

Log: `logs/coordinator_inference.log`.

Invocación directa (una postura):

&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python -m agent.coordinator_inference --posture prudencial --tag mi_run`
&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python -m agent.coordinator_inference --posture balanceado --tag mi_run`
&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python -m agent.coordinator_inference --posture desinversion --tag mi_run`

---

**3.9. engines/stress_engine.py — Estrés multi-escenario**

**Qué aporta:** Orquesta la ejecución completa de las 12 combinaciones (4 escenarios × 3 posturas) de forma automática. Por cada combinación aplica shocks macro a la cartera, llama a `coordinator_inference` para las 3 posturas y calcula los KPIs de ventas PC10 desde los CSVs de auditoría. Consolida todo en un único `stress_summary_<tag>.csv` de 12 filas.

Flujo interno:

1. Lee `configs/stress_scenarios.yaml` -> obtiene los 4 escenarios con sus shocks.
2. Por cada escenario: aplica `stress_portfolio(df, scenario)` -> cartera shockeada.
3. Si escenario == `pricing_crunch`: inyecta `BID_HAIRCUT_GLOBAL = bid_haircut_mult` en config.
4. Ejecuta `coordinator_inference` para cada una de las 3 posturas.
5. Lee `decisiones_audit_<postura>.csv` -> calcula KPIs ventas (n_sales, sale_pnl_total, avg_bid_pct_ead, sell_blocked_count).
6. Construye `stress_summary_<tag>.csv` (12 filas: escenario × postura).

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python -m engines.stress_engine --tag pc10_final_clean`

O vía bat:

&nbsp;&nbsp;&nbsp;&nbsp;`.
un_inference_only.bat`

---

**3.10. Comparativa de posturas y backtesting**

**reports/compare_postures.py — Comparativa de posturas**

**Qué aporta:** Lee los KPIs de las 3 posturas para cada escenario y genera una tabla comparativa en una sola fila por escenario, con las 3 posturas como columnas. Verifica automáticamente la monotonía esperada de negocio:
&nbsp;&nbsp;· Ventas: Prudencial <= Balanceado <= Desinversión.
&nbsp;&nbsp;· Capital liberado: Prudencial <= Balanceado <= Desinversión.
&nbsp;&nbsp;· RWA final: Desinversión <= Balanceado <= Prudencial.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python -m reports.compare_postures --tag mi_run`

**reports/backtesting_light.py — Backtesting ligero**

**Qué aporta:** Compara la estrategia del coordinador RL contra tres baselines de referencia (HoldAll, SellAll, RuleBasedNPL) para el escenario baseline. Calcula la ganancia incremental en EVA, capital liberado y P&L de ventas que aporta el agente RL sobre cada baseline.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python -m reports.backtesting_light --tag mi_run`

**baselines/baseline_policies.py — Políticas de referencia**

**Qué aporta:** Define las tres estrategias de referencia para evaluación comparativa:
&nbsp;&nbsp;· `HoldAll`: mantiene toda la cartera sin tomar ninguna acción.
&nbsp;&nbsp;· `SellAll`: vende toda la cartera indiscriminadamente.
&nbsp;&nbsp;· `RuleBasedNPL`: reglas simples basadas en umbrales de PD y PTI.

---

**3.11. reports/make_committee_pack.py — Committee pack**

**Qué aporta:** Genera un paquete completo y auditable para presentación a comité. Copia todos los artefactos relevantes (CSVs de stress, Excels de decisiones, ficheros de configuración, logs de CI, modelos) en una carpeta estructurada. Genera `MANIFEST.json` con trazabilidad completa: timestamp, tag, commit git (HEAD), estado del repo (clean/dirty), checksums SHA-256 de todos los modelos y lista de artefactos incluidos.

Invocación:

&nbsp;&nbsp;&nbsp;&nbsp;`.\.venv\Scripts\python reports\make_committee_pack.py --tag mi_run`



**======================================================================**
**4. RESULTADOS: DÓNDE ENCONTRARLOS Y QUÉ SE VE EN CADA UNO**
**======================================================================**



Una vez completado el pipeline (o cualquier subpaso), los resultados se encuentran en las carpetas `reports/`, `logs/` y `models/`. A continuación se describe cada tipo de resultado y qué información contiene.

---

**4.1. reports/stress_summary_<tag>.csv — Tabla maestra de KPIs (12 filas)**

**Dónde:** `reports/stress_summary_<tag>.csv`
**Cuándo se genera:** al final del stress engine completo.
**Qué se ve:** Una fila por cada combinación (escenario × postura). Columnas principales:

* `scenario` / `posture`: identificador de la combinación.
* `total_ead`: EAD total de la cartera (€).
* `total_eva_pre` / `total_eva_post`: EVA antes/después de las decisiones (solo MANTENER+REESTRUCTURAR).
* `total_rwa_pre` / `total_rwa_post`: RWA antes/después.
* `capital_liberado`: capital liberado total por ventas (€).
* `n_sales` / `n_restruct` / `n_mantener`: número de préstamos por acción final.
* `sale_pnl_total`: P&L total realizado en ventas (negativo = descuento NPL, en euros).
* `avg_sale_pnl`: P&L medio por operación de venta.
* `avg_bid_pct_ead`: precio bid medio como fracción del EAD (ej: 0.13 = 13% del EAD).
* `avg_bid_pct_ead_available`: True si el KPI está calculado; False si no hay ventas o falta columna.
* `sell_blocked_count`: número de ventas bloqueadas por overrides prudenciales o guardas fire-sale.

**Cómo interpretar:** Compara filas para ver el impacto por escenario (baseline vs. severe) y por postura (prudencial vs. desinversion). En `pricing_crunch`, `avg_bid_pct_ead` baja (~10% vs ~13% en baseline) y `sell_blocked_count` sube. Desinversión siempre muestra más ventas y más capital liberado que Prudencial.

---

**4.2. reports/stress_<tag>_<timestamp>/ — Carpetas del stress engine**

**Dónde:** `reports/stress_<tag>_<timestamp>/`
**Qué contiene:** Una subcarpeta por escenario, y dentro una por postura. Cada postura contiene:

* **`decisiones_finales_<postura>.xlsx`** — Excel con decisiones loan-by-loan. Columnas: `loan_id, EAD, PD, LGD, RW, DPD, segmento, Accion_final, EVA_pre, EVA_post, RWA_pre, RWA_post, capital_liberado, pnl_realized, price_ratio_ead, PTI, DSCR, reason_code, macro_action_id`. Permite revisar qué se decidió para cada préstamo y por qué.

* **`overrides_log_<postura>.csv`** — Registro de todos los overrides prudenciales aplicados. Columnas garantizadas: `loan_id, level, from_action, to_action, portfolio_context, posture, run_id, macro_action_used, macro_rationales_short, pti_actual, dscr_actual, pnl`. Permite auditar cuántas y qué tipo de correcciones aplicó el coordinador sobre la recomendación del PPO.

* **`portfolio_kpis_<postura>.json`** — KPIs agregados en formato JSON: `n_loans, n_sales, n_restruct, n_mantener, total_ead, total_eva_pre, total_eva_post, total_rwa_pre, total_rwa_post, capital_liberado, sale_pnl_total, avg_sale_pnl, avg_bid_pct_ead, avg_bid_pct_ead_available, sell_blocked_count`. Diseñado para consumo por herramientas BI o scripts de validación.

* **`decisiones_audit_<postura>.csv`** — CSV loan-by-loan completo; insumo del stress engine para calcular los KPIs PC10 de `stress_summary`.

* **`portfolio_<escenario>.xlsx`** — Cartera final agregada con las tres posturas del escenario.

---

**4.3. reports/compare_postures_<tag>.csv — Comparativa de posturas**

**Dónde:** `reports/compare_postures_<tag>.csv`
**Cuándo se genera:** al ejecutar `reports/compare_postures.py`.
**Qué se ve:** Una fila por escenario con las 3 posturas como columnas para comparación directa (side-by-side). Permite verificar la monotonía de negocio de un vistazo: si Desinversión vende más que Balanceado, y Balanceado más que Prudencial. Columnas representativas: `scenario, n_sales_prudencial, n_sales_balanceado, n_sales_desinversion, capital_liberado_prudencial, ..., rwa_post_prudencial, ...`

---

**4.4. reports/backtesting_light_<tag>.csv y .md — Backtesting vs. baselines**

**Dónde:** `reports/backtesting_light_<tag>.csv` y `reports/backtesting_light_<tag>.md`
**Cuándo se genera:** al ejecutar `reports/backtesting_light.py`.
**Qué se ve:** Tabla comparativa del agente coordinador RL vs. las tres estrategias de referencia (HoldAll, SellAll, RuleBasedNPL) para el escenario baseline. Métricas: EVA total, capital liberado, P&L de ventas, n_sales, n_restruct. El fichero `.md` es la versión legible para el committee pack.

---

**4.5. reports/coordinated_inference_<tag>_<timestamp>_<postura>/ — Inferencia individual**

**Dónde:** `reports/coordinated_inference_<tag>_<timestamp>_<postura>/`
**Cuándo se genera:** al ejecutar `coordinator_inference.py` directamente (sin stress engine).
**Qué contiene:** Los mismos cuatro ficheros que la inferencia dentro del stress engine (decisiones_finales, overrides_log, portfolio_kpis, decisiones_audit).

---

**4.6. reports/inference_<timestamp>_<tag>/ — Inferencia micro (loan-level)**

**Dónde:** `reports/inference_<timestamp>_<tag>/`
**Cuándo se genera:** al ejecutar `main.py infer` o `policy_inference.py` directamente.
**Qué se ve:**

* **`summary.csv`** — Una fila por préstamo: acción recomendada por el PPO micro, métricas financieras calculadas (EVA_post, RWA_post, capital_liberado, pnl) y pasos de razonamiento (Explain_Steps).
* **`decisiones_explicadas.xlsx`** — Mismo contenido en formato Excel corporativo con formato visual para presentación.

---

**4.7. reports/inference_portfolio_<timestamp>_<tag>/ — Inferencia macro (cartera)**

**Dónde:** `reports/inference_portfolio_<timestamp>_<tag>/`
**Cuándo se genera:** al ejecutar `policy_inference_portfolio.py` directamente.
**Qué se ve:**

* **`trajectory.csv`** — Historial paso a paso del agente macro: acción macro aplicada en cada step, EVA total, RWA, capital liberado acumulado, reward. Permite visualizar cómo evoluciona la cartera a lo largo de los pasos de decisión.
* **`summary_portfolio.csv`** — Resumen final de la cartera: KPIs agregados al final de la trayectoria.
* **`portfolio_final.xlsx`** — Estado final de cada préstamo tras la estrategia del agente macro.

---

**4.8. reports/committee_pack_<tag>_<timestamp>/ — Pack para comité**

**Dónde:** `reports/committee_pack_<tag>_<timestamp>/`
**Cuándo se genera:** al ejecutar `make_committee_pack.py`.
**Qué contiene:**

```
committee_pack_<tag>_<timestamp>/
|-- MANIFEST.json                <- trazabilidad: timestamp, tag, commit git,
|                                   estado repo (clean/dirty), checksums SHA-256
|                                   de todos los modelos, lista de artefactos
|-- config.py                    <- configuración exacta usada
|-- stress_summary_<tag>.csv     <- tabla maestra de KPIs (12 filas)
|-- MEMO_COMMITTEE.md            <- memo ejecutivo para comité
|-- RUNBOOK_COMMITTEE.md         <- runbook operativo
|-- CHECKLIST_COMMITTEE.md      <- checklist de validación
|-- README.md                    <- este fichero
|-- evaluation_pc6.csv           <- evaluación de referencia PC6
|-- evaluation_report.md         <- informe de evaluación
|-- pip_freeze.txt               <- snapshot exacto de dependencias
|-- stress_scenarios.yaml        <- escenarios usados
|-- real_portfolio_mapping.yaml  <- mapeo de cartera real
|-- ingest_portfolio.py          <- script de ingestión
|-- ci_local_<tag>.log           <- log completo del CI que generó el pack
|-- run_prudencial/
|   `-- decisiones_finales_prudencial.xlsx
|-- run_balanceado/
|   `-- portfolio_final.xlsx
`-- run_desinversion/
    `-- decisiones_finales_desinversion.xlsx
```

El `MANIFEST.json` garantiza reproducibilidad completa: cualquier auditor puede verificar que los modelos no fueron modificados (checksums SHA-256) y que el pipeline se ejecutó desde el commit exacto registrado.

---

**4.9. models/ — Modelos entrenados y normalizadores**

**Dónde:** `models/`
**Qué contiene:**

* `best_model.zip` — PPO micro entrenado en LoanEnv.
* `best_model_loan.zip` — PPO micro especializado en loans.
* `best_model_portfolio.zip` — PPO macro entrenado en PortfolioEnv.
* `vecnormalize_final.pkl` — Normalizador de observaciones del agente micro.
* `vecnormalize_loan.pkl` — Normalizador del agente loan.
* `vecnormalize_portfolio.pkl` — Normalizador del agente macro.
* `feature_order.json` — Orden canónico de features para reconstruir observaciones.
* `obs_feature_order_loan.json` — Orden de features del entorno loan.
* `training_metadata.json` — Hiperparámetros y métricas del entrenamiento micro.
* `training_metadata_loan.json` — Métricas del entrenamiento loan.
* `checkpoints/` — Checkpoints periódicos guardados durante el entrenamiento.

---

**4.10. logs/ — Logs de ejecución y evidencias QA**

**Dónde:** `logs/`
**Qué registra cada fichero:**

* `coordinator_inference.log` — Cada decisión del coordinador, overrides aplicados, métricas por préstamo.
* `stress_engine_<tag>.log` — Progreso del stress engine: escenarios, posturas, KPIs calculados.
* `train_subagents.log` — Progreso del entrenamiento: episodios, rewards, métricas de convergencia.
* `policy_inference.log` — Inferencia micro préstamo a préstamo.
* `policy_inference_portfolio.log` — Inferencia macro paso a paso.
* `ci_local_<tag>_<timestamp>.log` — Log completo del CI local (todos los steps).
* `qa_checkpoint<N>_evidence.txt` — Evidencias de validación QA por checkpoint.
* `tb/` y `tensorboard/` — Logs TensorBoard para visualizar curvas de entrenamiento.

Para visualizar las curvas de entrenamiento en TensorBoard:

&nbsp;&nbsp;&nbsp;&nbsp;`tensorboard --logdir logs/tb`



**======================================================================**
**5. ESCENARIOS Y POSTURAS DE RIESGO**
**======================================================================**



**5.1. Los 4 escenarios macro**

* **baseline**: sin shocks. Estado de referencia. Shocks: ninguno.
* **mild**: deterioro económico moderado. PD×1.2, LGD+0.05, RW×1.1, colateral×0.95.
* **severe**: recesión severa. PD×1.5, LGD+0.15, RW×1.25, colateral×0.85, rates+200bps.
* **pricing_crunch**: crisis de liquidez. bid_haircut×1.3 -> precios NPL 30% más bajos.

**5.2. Las 3 posturas de riesgo**

* **prudencial**: prioriza estabilidad y contención de pérdidas. Menos ventas, más reestructuraciones.
* **balanceado**: mejor trade-off EVA/capital. Punto medio en todas las métricas.
* **desinversion**: prioriza liberación inmediata de capital y reducción de RWA. Más ventas, más capital liberado, RWA más bajo.

**Regla de monotonía esperada (validación banco-ready):**

&nbsp;&nbsp;· Nº ventas: Prudencial <= Balanceado <= Desinversión.
&nbsp;&nbsp;· Capital liberado: Prudencial <= Balanceado <= Desinversión.
&nbsp;&nbsp;· RWA final: Desinversión <= Balanceado <= Prudencial.

Se verifica automáticamente en `reports/compare_postures.py` y en `ci_local.bat` (step 3.5).

**5.3. KPIs de ventas — PC10**

* `sale_pnl_total`: P&L total de ventas (valor negativo para NPLs, reflejando el descuento).
* `avg_sale_pnl`: P&L medio por operación.
* `avg_bid_pct_ead`: precio bid medio como fracción del EAD (ej: 0.13 = 13%). Baja en `pricing_crunch`.
* `avg_bid_pct_ead_available`: flag de disponibilidad del KPI.
* `sell_blocked_count`: ventas rechazadas por guardas de fire-sale o floor prudencial.

**Nota:** cuando `Accion_final = VENDER`, `EVA_post = 0` (el préstamo sale del balance). El P&L de la venta vive en `pnl_realized` / `sale_pnl_total`, no en `EVA_post`. El `total_eva_post` solo suma MANTENER y REESTRUCTURAR.



**======================================================================**
**6. SCRIPTS DE AUTOMATIZACIÓN Y CI**
**======================================================================**



**6.1. Ficheros .bat**

* **`run_3_postures_executability.bat`**: Ejecuta las 3 posturas con el coordinador multi-agente. Recibe `--tag`.
* **`run_inference_only.bat`**: Solo inferencia coordinada (sin entrenamiento). Recibe `--tag`.
* **`run_pipeline.bat`**: Pipeline completo: train + inference + stress (para entornos con modelos validados).
* **`run_recalibrated_inference.bat`**: Inferencia con modelos recalibrados.
* **`run_repro_test.bat`**: Test de reproducibilidad entre dos ejecuciones consecutivas.
* **`smoke_test.bat`**: Smoke test rápido (cartera reducida, 1 escenario, 1 postura).
* **`ci_local.bat`**: CI completo local con steps enumerados y validaciones QA.

**ci_local.bat — Steps del CI completo:**

```
Step 1 : Smoke test (cartera reducida, 1 escenario, 1 postura)
Step 2 : pytest completo (suite de tests)
Step 3 : Inferencia coordinada 3 posturas
Step 3.5: Validación de monotonía (compare_postures)
Step 3.6: pytest pricing KPIs + validación columnas PC10 en stress_summary
Step 4 : Stress engine completo (4 escenarios x 3 posturas = 12 runs)
Step 5 : Backtesting ligero vs. baselines
Step 6 : Committee pack
-> Salida: logs/ci_local_<tag>_<timestamp>.log
```

**6.2. main.py — Subcomandos del orquestador central**

* **`generate`**: llama a `data/generate_portfolio.py` -> `data/portfolio_synth.xlsx`.
* **`train`**: llama a `agent/train_subagents.py` -> `models/best_model*.zip`, `models/vecnormalize*.pkl`.
* **`infer`**: llama a `agent/policy_inference.py` -> `reports/inference_<ts>/summary.csv`, `decisiones_explicadas.xlsx`.
* **`summary`**: llama a `reports/results_summary.py` -> DataFrame consolidado, Excel, JSON, gráficos, `executive_summary.txt`.



**======================================================================**
**7. ESTRUCTURA DE CARPETAS COMPLETA**
**======================================================================**



```
data/
|-- portfolio_synth.xlsx            <- cartera sintética
|-- portfolio_synth_smoke.xlsx      <- cartera reducida para smoke tests
|-- mappings/
|   `-- real_portfolio_mapping.yaml <- mapeo columnas cartera real -> esquema interno
`-- __init__.py

models/
|-- best_model.zip                  <- PPO micro (LoanEnv)
|-- best_model_loan.zip             <- PPO micro especializado
|-- best_model_portfolio.zip        <- PPO macro (PortfolioEnv)
|-- vecnormalize_final.pkl          <- VecNormalize agente micro
|-- vecnormalize_loan.pkl           <- VecNormalize agente loan
|-- vecnormalize_portfolio.pkl      <- VecNormalize agente macro
|-- feature_order.json              <- orden canónico de features
|-- obs_feature_order_loan.json     <- orden features entorno loan
|-- training_metadata.json          <- metadatos entrenamiento micro
|-- training_metadata_loan.json     <- metadatos entrenamiento loan
`-- checkpoints/                    <- checkpoints periódicos PPO

reports/
|-- stress_summary_<tag>.csv        <- tabla KPIs (4 escenarios x 3 posturas)
|-- compare_postures_<tag>.csv      <- comparativa side-by-side de posturas
|-- backtesting_light_<tag>.csv     <- backtesting vs. baselines
|-- backtesting_light_<tag>.md      <- informe backtesting en Markdown
|-- stress_<tag>_<timestamp>/
|   |-- <escenario>/
|   |   `-- <postura>/
|   |       `-- <timestamp>_<tag>_<escenario>_<postura>/
|   |           |-- decisiones_audit_<postura>.csv
|   |           |-- decisiones_finales_<postura>.xlsx
|   |           |-- overrides_log_<postura>.csv
|   |           `-- portfolio_kpis_<postura>.json
|   `-- portfolio_<escenario>.xlsx
|-- coordinated_inference_<tag>_<timestamp>_<postura>/
|   |-- decisiones_finales_<postura>.xlsx
|   |-- overrides_log_<postura>.csv
|   `-- portfolio_kpis_<postura>.json
|-- inference_<timestamp>_<tag>/
|   |-- summary.csv
|   `-- decisiones_explicadas.xlsx
|-- inference_portfolio_<timestamp>_<tag>/
|   |-- trajectory.csv
|   |-- summary_portfolio.csv
|   `-- portfolio_final.xlsx
`-- committee_pack_<tag>_<timestamp>/
    |-- MANIFEST.json
    |-- stress_summary_<tag>.csv
    |-- MEMO_COMMITTEE.md / RUNBOOK_COMMITTEE.md / CHECKLIST_COMMITTEE.md
    |-- config.py / stress_scenarios.yaml / real_portfolio_mapping.yaml
    |-- pip_freeze.txt / ci_local_<tag>.log
    |-- run_prudencial/ run_balanceado/ run_desinversion/

logs/
|-- coordinator_inference.log
|-- stress_engine_<tag>.log
|-- train_subagents.log
|-- policy_inference.log
|-- policy_inference_portfolio.log
|-- ci_local_<tag>_<timestamp>.log
|-- qa_checkpoint<N>_evidence.txt   <- evidencias QA por checkpoint
|-- tb/                             <- TensorBoard (entrenamiento micro)
`-- tensorboard/                    <- TensorBoard (entrenamiento portfolio)

configs/
`-- stress_scenarios.yaml           <- shocks por escenario

tests/                              <- suite pytest completa
```



**======================================================================**
**8. TRAZABILIDAD Y REPRODUCIBILIDAD**
**======================================================================**



El proyecto mantiene trazabilidad completa en todos los niveles:

* **Nivel código:** ramas Git por Punto de Control (PC). Rama activa: `feat/pc10-pricing-kpis-stress` (HEAD: ceb5dcf). `main` recibe merges aprobados por comité.
* **Nivel ejecución:** cada run genera timestamps únicos en nombres de carpetas y ficheros. `MANIFEST.json` registra commit exacto, estado del repo (clean/dirty), checksums SHA-256 de modelos.
* **Nivel decisión:** `overrides_log_<postura>.csv` registra cada override prudencial con su motivo. `decisiones_audit_<postura>.csv` es el registro loan-by-loan completo.
* **Nivel CI:** `ci_local.bat` ejecuta la suite completa con evidencias en `logs/qa_checkpoint<N>_evidence.txt`.

El pack definitivo de PC10 es: `reports/committee_pack_pc10_hardening_final_20260221_125131/` (commit=ceb5dcf, status=clean).

`config.py` es la pieza central que garantiza coherencia entre todos los componentes: entornos RL, motores financieros, engines y agentes importan de un único fichero de configuración, garantizando alineación regulatoria y técnica a lo largo de toda la cadena.
