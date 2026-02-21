POC -- OPTIMIZADOR DE CARTERAS EN DEFAULT
Banco L1.5 - Basilea III Metodo Estandar - PC10 (feat/pc10-pricing-kpis-stress)
CI smoke: https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml

================================================================================
INDICE
================================================================================

  1. Vision general y flujo del pipeline
  2. Configuracion global
  3. Flujo paso a paso -- que hace cada script
     3.1  Generacion de cartera sintetica
     3.2  Ingesta de cartera real
     3.3  Motores financieros (cerebro financiero)
     3.4  Entornos de aprendizaje por refuerzo
     3.5  Entrenamiento de agentes RL
     3.6  Inferencia micro (prestamo a prestamo)
     3.7  Inferencia macro (cartera completa)
     3.8  Inferencia coordinada multi-agente
     3.9  Motor de estres multi-escenario
     3.10 Comparativa de posturas y backtesting
     3.11 Committee pack
  4. Resultados: donde encontrarlos y que se ve en cada uno
  5. Escenarios y posturas de riesgo
  6. Scripts de automatizacion y CI
  7. Estructura de carpetas completa
  8. Trazabilidad y reproducibilidad


================================================================================
1. VISION GENERAL Y FLUJO DEL PIPELINE
================================================================================

El proyecto implementa un pipeline completo para optimizar carteras en default
(NPL) bajo Basilea III metodo estandar. Combina motores financieros deterministas
con agentes de aprendizaje por refuerzo (RL) y una capa de overrides
prudenciales, produciendo decisiones explicables y trazables para comite.

El flujo de extremo a extremo es el siguiente:

  [1] DATOS
      data/generate_portfolio.py   <- genera cartera sintetica (Excel)
      data/ingest_portfolio.py     <- ingiere cartera real del banco (Excel/CSV)
           |
           v
  [2] CONFIGURACION GLOBAL
      config.py                    <- parametros regulatorios, RL, motores, rutas
      configs/stress_scenarios.yaml<- definicion de los 4 escenarios macro
           |
           v
  [3] ENTRENAMIENTO (solo primera vez o cuando cambia la cartera)
      agent/train_subagents.py     <- entrena PPO micro (LoanEnv) y macro (PortfolioEnv)
        via main.py train
           |
           v
  [4] INFERENCIA COORDINADA (produccion)
      agent/coordinator_inference.py  <- por postura (prudencial/balanceado/desinversion)
      multiagent/coordinator_agent.py <- logica jerarquica macro + micro + overrides
      multiagent/restruct_agent.py    <- motor de reestructuracion
      multiagent/pricing_agent.py     <- motor de pricing NPL
           |
           v
  [5] ESTRES MULTI-ESCENARIO
      engines/stress_engine.py     <- 4 escenarios x 3 posturas = 12 ejecuciones
           |
           v
  [6] REPORTING Y VALIDACION
      reports/compare_postures.py    <- comparativa 3 posturas (monotonia)
      reports/backtesting_light.py   <- PPO vs baselines de referencia
      reports/make_committee_pack.py <- empaquetado auditable para comite

Orquestador principal: main.py centraliza los subcomandos generate/train/infer/summary.
Para una ejecucion completa desatendida, se utiliza ci_local.bat.


================================================================================
2. CONFIGURACION GLOBAL
================================================================================

2.1. config.py -- Nucleo comun del proyecto
--------------------------------------------
Todos los modulos importan de config.py. Es la unica fuente de verdad para
parametros regulatorios, RL, motores y rutas. Cambiar un valor aqui afecta
a toda la cadena automaticamente.

Bloques principales:
  - Regulacion: hurdle RAROC, CET1/Capital minimos, buffers Basilea III.
  - BaselSTDMapping: ponderaciones de riesgo (RW) por segmento, rating y estado
    (performing/default).
  - ReestructuraParams: rejillas de plazo, tasa y quita; umbrales PTI/DSCR max.
  - SensibilidadReestructura: reduccion PD/LGD tras reestructuracion, ventana de
    curacion, RW "cured".
  - PrecioVentaParams: haircuts por segmento, rango bid, coste legal, horizonte
    de recuperacion.
  - RewardParams: pesos del reward RL (EVA, capital, HHI, P&L ventas, penaliz.).
  - PPOParams: hiperparametros del agente (lr, batch_size, n_steps, gamma, etc.)
  - EnvParams: dimensiones de estado, top-k, semillas globales.
  - BID_HAIRCUT_GLOBAL: multiplicador inyectado en pricing en escenario
    pricing_crunch.
  - Rutas estandar: DATA_PATH, MODELS_PATH, REPORTS_PATH, LOGS_PATH,
    CHECKPOINTS_PATH.

2.2. configs/stress_scenarios.yaml -- Los 4 escenarios macro
-------------------------------------------------------------
  - baseline      : sin shocks. Estado de referencia.
  - mild          : deterioro moderado. PD x1.2, LGD+0.05, RW x1.1, col x0.95.
  - severe        : recesion severa. PD x1.5, LGD+0.15, RW x1.25, col x0.85,
                    rates +200bps.
  - pricing_crunch: crisis de liquidez. bid_haircut x1.3 -> precios NPL 30% mas
                    bajos, mas ventas bloqueadas.

2.3. data/mappings/real_portfolio_mapping.yaml
----------------------------------------------
Traduce columnas del fichero bruto del banco al esquema interno del pipeline.
Lo usa data/ingest_portfolio.py con tipado, renombrado y conversion de unidades.


================================================================================
3. FLUJO PASO A PASO -- QUE HACE CADA SCRIPT
================================================================================

3.1. data/generate_portfolio.py -- Generacion de cartera sintetica
-------------------------------------------------------------------
QUE APORTA: Crea una cartera NPL sintetica con distribuciones realistas de
segmentos, ratings, EAD, PD y LGD. Sirve para desarrollo, pruebas y CI cuando
no se dispone de cartera real.

Funcion principal: generate_portfolio(n) -> DataFrame
                   export_excel(df, out_path) -> Excel corporativo

Invocacion:
  python main.py generate --n 1000 --out data/portfolio_synth.xlsx

Salidas:
  data/portfolio_synth.xlsx        <- cartera completa para entrenamiento e inferencia
  data/portfolio_synth_smoke.xlsx  <- cartera reducida para smoke tests

----------------------------------------------------------------------

3.2. data/ingest_portfolio.py -- Ingesta de cartera real
---------------------------------------------------------
QUE APORTA: Transforma la cartera real del banco al esquema interno aplicando
el mapeo de data/mappings/real_portfolio_mapping.yaml. Realiza coerciones de
tipo, validaciones y produce un DataFrame consumible por el pipeline
(entrenamiento o inferencia). No tiene CLI propio; se importa como modulo.

----------------------------------------------------------------------

3.3. Motores financieros (cerebro financiero)
---------------------------------------------
Los motores son logica financiera determinista; no dependen del RL.

  optimizer/restructure_optimizer.py -- Motor de reestructuracion
  ................................................................
  QUE APORTA: Dado un prestamo en default, encuentra la combinacion optima de
  (plazo adicional, nueva tasa, quita sobre EAD) que maximiza el EVA post-
  reestructuracion. Modela flujos de caja (sistema frances), asequibilidad via
  PTI/DSCR, dinamica PD/LGD tras reestructuracion y capital/provisiones en el
  horizonte.

  Funcion: optimize_restructure(loan_dict)
  Retorna: plazo_optimo, tasa_nueva, quita, EVA_post, EVA_gain, RWA_post,
           RORWA_post, PTI, DSCR.

  optimizer/price_simulator.py -- Motor de pricing NPL
  .....................................................
  QUE APORTA: Simula el precio de mercado para una venta NPL (bid price),
  calcula el P&L realizado, el capital liberado y percentiles de precio.
  BID_HAIRCUT_GLOBAL amplifica haircuts en pricing_crunch. Genera price_ratio_ead
  (bid/EAD), columna canonica para el KPI avg_bid_pct_ead del PC10.

  Funcion: simulate_npl_price(loan_dict)
  Retorna: precio_optimo, pnl, capital_liberado, price_ratio_ead, p5, p50, p95.

  engines/stress_engine.py -- Motor de estres multi-escenario
  ............................................................
  QUE APORTA: Orquesta las 12 ejecuciones (4 escenarios x 3 posturas). Por
  cada combinacion: aplica shocks macro, inyecta BID_HAIRCUT_GLOBAL si
  corresponde, lanza coordinator_inference para las 3 posturas, lee los CSVs
  de auditoria y calcula KPIs de ventas PC10. Consolida en
  stress_summary_<tag>.csv.

  Funcion orquestadora: run_stress_pipeline(tag, scenarios, postures)

  engines/sensitivities_engine.py -- Motor de sensitividades
  ..........................................................
  QUE APORTA: Calcula derivadas parciales de NI, EVA, RWA y capital respecto a
  PD, LGD, rate, RW y EAD. Genera global_sensitivity_score(loan) que identifica
  prestamos "palanca". Se usa para ranking en PortfolioEnv y explicabilidad.

----------------------------------------------------------------------

3.4. Entornos de aprendizaje por refuerzo (RL)
-----------------------------------------------
Los entornos son el "mundo simulado" donde los agentes RL aprenden.

  env/loan_env.py -- Entorno micro (nivel prestamo)
  .................................................
  QUE APORTA: Simula la evolucion de un prestamo individual. Acciones:
  MANTENER, REESTRUCTURAR o VENDER. Observacion: vector de 10 features
  (EAD, PD, LGD, RW, DPD, spread, segmento, ingreso, cashflow, LTV).
  Recompensa: combinacion de ΔEVA, capital liberado, estabilidad y prudencia.
  Se usa para entrenar el agente micro y para inferencia loan-level.

  env/portfolio_env.py -- Entorno macro (nivel cartera)
  .....................................................
  QUE APORTA: Simula decisiones agregadas sobre una cartera NPL completa.
  Estado: vector de indicadores de cartera (EVA total, RWA, HHI, volatilidad
  EVA, capital liberado acumulado). Acciones (0-11): mantener todos, vender/
  reestructurar top-K por EVA/RORWA/PTI, combinaciones, heuristica STD.
  Implementa internamente _apply_maintain (drift NPL), _apply_restructure y
  _apply_sell (llama a simulate_npl_price, calcula pnl_realized,
  price_ratio_ead, capital liberado).

----------------------------------------------------------------------

3.5. agent/train_subagents.py -- Entrenamiento de agentes RL
-------------------------------------------------------------
QUE APORTA: Entrena el subagente micro (LoanEnv) y el macro (PortfolioEnv)
usando PPO de Stable-Baselines3. Sin estos modelos no hay inferencia.

  train_loan_agent(...) -> models/best_model.zip + models/vecnormalize_final.pkl
  train_portfolio_agent(...) -> models/best_model_portfolio.zip

Genera checkpoints en models/checkpoints/ y logs TensorBoard en logs/tb/ y
logs/tensorboard/.

Invocacion:
  python main.py train --agent both --portfolio data/portfolio_synth.xlsx     --total-steps 500000 --top-k 5 --scenario baseline

  python -m agent.train_subagents --agent loan --portfolio data/portfolio_synth.xlsx
  python -m agent.train_subagents --agent portfolio --portfolio data/portfolio_synth.xlsx

agent/train_agent.py -- Entrenamiento avanzado (alternativo)
QUE APORTA: Version avanzada con BusinessEvalCallback (early stopping basado
en EVA + capital liberado), VecNormalize en train y eval, logs para TensorBoard.

----------------------------------------------------------------------

3.6. agent/policy_inference.py -- Inferencia micro (loan-level)
---------------------------------------------------------------
QUE APORTA: Aplica el agente PPO micro prestamo a prestamo. Para cada prestamo
reconstruye la observacion en el orden canonico de FEATURE_COLUMNS, obtiene la
accion y calcula metricas post-decision (EVA_post, RWA_post, capital_liberado,
pnl). Genera resumen con trazabilidad completa (Explain_Steps, ΔEVA).

Flujo:
  1. Carga best_model.zip + vecnormalize_final.pkl
  2. Carga cartera desde Excel o CSV
  3. Por cada prestamo: predice accion + calcula metricas
  4. Exporta summary.csv y decisiones_explicadas.xlsx

Log: logs/policy_inference.log

Invocacion:
  python main.py infer --model models/best_model.zip     --vecnorm models/vecnormalize_final.pkl     --portfolio data/portfolio_synth.xlsx

----------------------------------------------------------------------

3.7. agent/policy_inference_portfolio.py -- Inferencia macro (cartera completa)
--------------------------------------------------------------------------------
QUE APORTA: Aplica el agente PPO macro sobre la cartera completa durante
n_steps. En cada paso predice la accion macro, aplica env.step(action) y
recoge metricas agregadas (EVA total, RWA, capital, reward). Produce historial
de trayectoria y resumen final.

Flujo:
  1. Carga best_model_portfolio.zip
  2. Crea PortfolioEnv(loans_df=df, top_k)
  3. Ejecuta n_steps de inferencia y registra metricas por paso
  4. Exporta trajectory.csv, summary_portfolio.csv, portfolio_final.xlsx

Log: logs/policy_inference_portfolio.log

Invocacion:
  python -m agent.policy_inference_portfolio --portfolio data/portfolio_synth.xlsx     --tag mi_run

----------------------------------------------------------------------

3.8. Inferencia coordinada multi-agente
----------------------------------------
Este es el modo de produccion del pipeline. Combina los tres agentes y los
overrides prudenciales en una sola ejecucion por postura.

  multiagent/coordinator_agent.py -- Agente coordinador
  ......................................................
  QUE APORTA: Implementa la jerarquia de decision completa en decide():

    1. Obtiene act_macro del PPO macro.
    2. Calcula sugerencia de reestructuracion micro (RestructAgent).
    3. Calcula sugerencia de pricing micro (PricingAgent).
    4. Aplica overrides prudenciales Banco L1.5:
       - Evita vender prestamos con EVA_post alto (buenos prestamos).
       - Fuerza reestructuracion si EVA_gain muy alto y PTI aceptable.
       - Bloquea fire-sales si precio < floor prudencial.
    5. Devuelve: accion_final (MANTENER/REESTRUCTURAR/VENDER) + razon.

  multiagent/restruct_agent.py -- Agente de reestructuracion
  ...........................................................
  QUE APORTA: Wrapper de optimize_restructure. Normaliza entradas y devuelve
  diccionario homogeneo con EVA_post, EVA_gain, RWA_post, PTI, DSCR,
  plazo_optimo, tasa_nueva, quita, ok.

  multiagent/pricing_agent.py -- Agente de pricing
  .................................................
  QUE APORTA: Wrapper de simulate_npl_price. Devuelve precio_optimo, pnl,
  capital_liberado, p5, p50, p95, price_ratio_ead, ok.

  agent/coordinator_inference.py -- Script central de inferencia por postura
  ................................................................................
  QUE APORTA: Ejecuta la inferencia coordinada completa para una postura
  concreta. Es el script de produccion llamado por el stress engine para cada
  una de las 12 combinaciones (escenario x postura).

  Flujo:
    1. Carga configuracion de postura + modelos (PPO macro, micro, VecNormalize).
    2. Instancia CoordinatorAgent.
    3. Por cada prestamo: decide y registra metricas completas.
    4. Exporta en reports/coordinated_inference_<tag>_<timestamp>_<postura>/:
       - decisiones_finales_<postura>.xlsx  <- Excel con decisiones y metricas
       - overrides_log_<postura>.csv        <- log de overrides aplicados
       - portfolio_kpis_<postura>.json      <- KPIs agregados de la cartera
       - decisiones_audit_<postura>.csv     <- CSV loan-by-loan para KPIs PC10

  Log: logs/coordinator_inference.log

  Invocacion (una postura):
    .\.venv\Scripts\python -m agent.coordinator_inference --posture prudencial --tag mi_run
    .\.venv\Scripts\python -m agent.coordinator_inference --posture balanceado --tag mi_run
    .\.venv\Scripts\python -m agent.coordinator_inference --posture desinversion --tag mi_run

----------------------------------------------------------------------

3.9. engines/stress_engine.py -- Estres multi-escenario
-------------------------------------------------------
QUE APORTA: Orquesta la ejecucion completa de las 12 combinaciones (4
escenarios x 3 posturas) de forma automatica. Por cada combinacion aplica
shocks macro a la cartera, llama a coordinator_inference para las 3 posturas
y calcula los KPIs de ventas PC10 desde los CSVs de auditoria. Consolida en
un unico stress_summary_<tag>.csv de 12 filas.

Flujo interno:
  1. Lee configs/stress_scenarios.yaml -> 4 escenarios con sus shocks.
  2. Por cada escenario: aplica stress_portfolio(df, scenario).
  3. Si pricing_crunch: inyecta BID_HAIRCUT_GLOBAL = bid_haircut_mult.
  4. Ejecuta coordinator_inference para cada una de las 3 posturas.
  5. Lee decisiones_audit_<postura>.csv -> calcula KPIs ventas.
  6. Construye stress_summary_<tag>.csv (12 filas: escenario x postura).

Invocacion:
  .\.venv\Scripts\python -m engines.stress_engine --tag pc10_final_clean
  o via bat: .
un_inference_only.bat

----------------------------------------------------------------------

3.10. Comparativa de posturas y backtesting
--------------------------------------------

  reports/compare_postures.py -- Comparativa de posturas
  .......................................................
  QUE APORTA: Lee los KPIs de las 3 posturas por escenario y genera tabla
  comparativa (una fila por escenario, 3 posturas como columnas). Verifica
  automaticamente la monotonia esperada:
    - Ventas:          Prudencial <= Balanceado <= Desinversion
    - Capital liberado: Prudencial <= Balanceado <= Desinversion
    - RWA final:       Desinversion <= Balanceado <= Prudencial

  Invocacion: .\.venv\Scripts\python -m reports.compare_postures --tag mi_run

  reports/backtesting_light.py -- Backtesting ligero
  ...................................................
  QUE APORTA: Compara la estrategia del coordinador RL contra tres baselines
  (HoldAll, SellAll, RuleBasedNPL) para el escenario baseline. Calcula la
  ganancia incremental en EVA, capital liberado y P&L de ventas.

  Invocacion: .\.venv\Scripts\python -m reports.backtesting_light --tag mi_run

  baselines/baseline_policies.py -- Politicas de referencia
  ..........................................................
  QUE APORTA: Define HoldAll (mantiene todo), SellAll (vende todo) y
  RuleBasedNPL (reglas simples por umbrales de PD y PTI).

----------------------------------------------------------------------

3.11. reports/make_committee_pack.py -- Committee pack
------------------------------------------------------
QUE APORTA: Genera un paquete completo y auditable para comite. Copia
artefactos relevantes en una carpeta estructurada y genera MANIFEST.json
con: timestamp, tag, commit git (HEAD), estado repo (clean/dirty), checksums
SHA-256 de todos los modelos y lista de artefactos.

Invocacion: .\.venv\Scripts\python reports\make_committee_pack.py --tag mi_run


================================================================================
4. RESULTADOS: DONDE ENCONTRARLOS Y QUE SE VE EN CADA UNO
================================================================================

Una vez completado el pipeline (o cualquier subpaso), los resultados se
encuentran en reports/, logs/ y models/. A continuacion se describe cada tipo
de resultado y que informacion contiene.

----------------------------------------------------------------------

4.1. reports/stress_summary_<tag>.csv -- Tabla maestra de KPIs (12 filas)
--------------------------------------------------------------------------
Donde: reports/stress_summary_<tag>.csv
Cuando se genera: al final del stress engine completo.
Que se ve: Una fila por combinacion (escenario x postura). Columnas:

  Columna                   Que indica
  ------------------------  -------------------------------------------------
  scenario / posture        Identificador de la combinacion
  total_ead                 EAD total de la cartera (euros)
  total_eva_pre/post        EVA antes/despues (solo MANTENER+REESTRUCTURAR)
  total_rwa_pre/post        RWA antes/despues
  capital_liberado          Capital liberado total por ventas (euros)
  n_sales/n_restruct/n_man  Numero de prestamos por accion final
  sale_pnl_total            P&L total de ventas (negativo = descuento NPL)
  avg_sale_pnl              P&L medio por operacion de venta
  avg_bid_pct_ead           Precio bid medio como % del EAD (ej: 0.13 = 13%)
  avg_bid_pct_ead_available True si KPI calculado; False si no hay ventas
  sell_blocked_count        Ventas bloqueadas por overrides/fire-sale

Como interpretar: En pricing_crunch, avg_bid_pct_ead baja (~10% vs ~13% en
baseline) y sell_blocked_count sube. Desinversion siempre muestra mas ventas y
mas capital liberado que Prudencial.

----------------------------------------------------------------------

4.2. reports/stress_<tag>_<timestamp>/ -- Carpetas del stress engine
---------------------------------------------------------------------
Donde: reports/stress_<tag>_<timestamp>/
Que contiene: Una subcarpeta por escenario, y dentro una por postura. Cada
postura contiene:

  decisiones_finales_<postura>.xlsx
    Excel con decisiones loan-by-loan. Columnas: loan_id, EAD, PD, LGD, RW,
    DPD, segmento, Accion_final, EVA_pre, EVA_post, RWA_pre, RWA_post,
    capital_liberado, pnl_realized, price_ratio_ead, PTI, DSCR, reason_code,
    macro_action_id. Permite revisar que se decidio para cada prestamo y por
    que.

  overrides_log_<postura>.csv
    Registro de todos los overrides prudenciales aplicados. Columnas
    garantizadas: loan_id, level, from_action, to_action, portfolio_context,
    posture, run_id, macro_action_used, macro_rationales_short, pti_actual,
    dscr_actual, pnl. Permite auditar cuantas y que correcciones aplico el
    coordinador sobre la recomendacion del PPO.

  portfolio_kpis_<postura>.json
    KPIs agregados en JSON: n_loans, n_sales, n_restruct, n_mantener, total_ead,
    total_eva_pre, total_eva_post, total_rwa_pre, total_rwa_post,
    capital_liberado, sale_pnl_total, avg_sale_pnl, avg_bid_pct_ead,
    avg_bid_pct_ead_available, sell_blocked_count. Disenado para consumo por
    herramientas BI o scripts de validacion.

  decisiones_audit_<postura>.csv
    CSV loan-by-loan completo; insumo del stress engine para calcular los KPIs
    PC10 del stress_summary.

  portfolio_<escenario>.xlsx
    Cartera final agregada con las tres posturas del escenario.

----------------------------------------------------------------------

4.3. reports/compare_postures_<tag>.csv -- Comparativa de posturas
-------------------------------------------------------------------
Donde: reports/compare_postures_<tag>.csv
Cuando se genera: al ejecutar reports/compare_postures.py
Que se ve: Una fila por escenario con las 3 posturas como columnas para
comparacion directa (side-by-side). Columnas representativas:
  scenario, n_sales_prudencial, n_sales_balanceado, n_sales_desinversion,
  capital_liberado_prudencial, ..., rwa_post_prudencial, ...
Permite verificar la monotonia de negocio de un vistazo.

----------------------------------------------------------------------

4.4. reports/backtesting_light_<tag>.csv y .md -- Backtesting vs. baselines
---------------------------------------------------------------------------
Donde: reports/backtesting_light_<tag>.csv y .md
Cuando se genera: al ejecutar reports/backtesting_light.py
Que se ve: Tabla comparativa del coordinador RL vs. HoldAll, SellAll y
RuleBasedNPL para el escenario baseline. Metricas: EVA total, capital
liberado, P&L de ventas, n_sales, n_restruct. El .md es la version legible
para el committee pack.

----------------------------------------------------------------------

4.5. reports/coordinated_inference_<tag>_<timestamp>_<postura>/
----------------------------------------------------------------
Donde: reports/coordinated_inference_<tag>_<timestamp>_<postura>/
Cuando se genera: al ejecutar coordinator_inference.py directamente.
Que contiene: Los mismos cuatro ficheros que la inferencia dentro del stress
engine (decisiones_finales, overrides_log, portfolio_kpis, decisiones_audit).

----------------------------------------------------------------------

4.6. reports/inference_<timestamp>_<tag>/ -- Inferencia micro
--------------------------------------------------------------
Donde: reports/inference_<timestamp>_<tag>/
Cuando se genera: al ejecutar main.py infer o policy_inference.py.
Que se ve:

  summary.csv
    Una fila por prestamo: accion recomendada por el PPO micro, metricas
    financieras (EVA_post, RWA_post, capital_liberado, pnl) y pasos de
    razonamiento (Explain_Steps, ΔEVA).

  decisiones_explicadas.xlsx
    Mismo contenido en formato Excel corporativo con formato visual.

----------------------------------------------------------------------

4.7. reports/inference_portfolio_<timestamp>_<tag>/ -- Inferencia macro
-----------------------------------------------------------------------
Donde: reports/inference_portfolio_<timestamp>_<tag>/
Cuando se genera: al ejecutar policy_inference_portfolio.py.
Que se ve:

  trajectory.csv
    Historial paso a paso del agente macro: accion macro aplicada en cada
    step, EVA total, RWA, capital liberado acumulado, reward. Permite
    visualizar como evoluciona la cartera a lo largo de los pasos.

  summary_portfolio.csv
    Resumen final: KPIs agregados al terminar la trayectoria.

  portfolio_final.xlsx
    Estado final de cada prestamo tras la estrategia del agente macro.

----------------------------------------------------------------------

4.8. reports/committee_pack_<tag>_<timestamp>/ -- Pack para comite
------------------------------------------------------------------
Donde: reports/committee_pack_<tag>_<timestamp>/
Cuando se genera: al ejecutar make_committee_pack.py
Que contiene:

  MANIFEST.json                <- trazabilidad: timestamp, tag, commit git,
                                  estado repo (clean/dirty), checksums SHA-256
                                  de todos los modelos, lista de artefactos
  config.py                    <- configuracion exacta usada
  stress_summary_<tag>.csv     <- tabla maestra de KPIs (12 filas)
  MEMO_COMMITTEE.md            <- memo ejecutivo para comite
  RUNBOOK_COMMITTEE.md         <- runbook operativo
  CHECKLIST_COMMITTEE.md       <- checklist de validacion
  README.md / README.txt       <- documentacion del proyecto
  evaluation_pc6.csv           <- evaluacion de referencia PC6
  evaluation_report.md         <- informe de evaluacion
  pip_freeze.txt               <- snapshot exacto de dependencias
  stress_scenarios.yaml        <- escenarios usados
  real_portfolio_mapping.yaml  <- mapeo de cartera real
  ingest_portfolio.py          <- script de ingestion
  ci_local_<tag>.log           <- log completo del CI que genero el pack
  run_prudencial/              |
  run_balanceado/              +-- Excels de decisiones por postura
  run_desinversion/            |

El MANIFEST.json garantiza reproducibilidad: any auditor puede verificar que
los modelos no fueron modificados (checksums SHA-256) y que el pipeline se
ejecuto desde el commit exacto registrado.

----------------------------------------------------------------------

4.9. models/ -- Modelos entrenados y normalizadores
---------------------------------------------------
  best_model.zip              <- PPO micro entrenado en LoanEnv
  best_model_loan.zip         <- PPO micro especializado en loans
  best_model_portfolio.zip    <- PPO macro entrenado en PortfolioEnv
  vecnormalize_final.pkl      <- Normalizador agente micro
  vecnormalize_loan.pkl       <- Normalizador agente loan
  vecnormalize_portfolio.pkl  <- Normalizador agente macro
  feature_order.json          <- Orden canonico de features
  obs_feature_order_loan.json <- Orden features entorno loan
  training_metadata.json      <- Hiperparametros y metricas entrenamiento micro
  training_metadata_loan.json <- Metricas entrenamiento loan
  checkpoints/                <- Checkpoints periodicos del entrenamiento

----------------------------------------------------------------------

4.10. logs/ -- Logs de ejecucion y evidencias QA
-------------------------------------------------
  coordinator_inference.log      <- Cada decision, override, metricas por prestamo
  stress_engine_<tag>.log        <- Progreso stress engine: escenarios, posturas
  train_subagents.log            <- Progreso entrenamiento: episodios, rewards
  policy_inference.log           <- Inferencia micro prestamo a prestamo
  policy_inference_portfolio.log <- Inferencia macro paso a paso
  ci_local_<tag>_<timestamp>.log <- Log completo del CI local (todos los steps)
  qa_checkpoint<N>_evidence.txt  <- Evidencias de validacion QA por checkpoint
  tb/                            <- Logs TensorBoard (entrenamiento micro)
  tensorboard/                   <- Logs TensorBoard (entrenamiento portfolio)

Para visualizar curvas de entrenamiento:
  tensorboard --logdir logs/tb


================================================================================
5. ESCENARIOS Y POSTURAS DE RIESGO
================================================================================

5.1. Los 4 escenarios macro
----------------------------
  Escenario       Descripcion                  Shocks principales
  --------------- ---------------------------- ----------------------------------
  baseline        Sin shocks. Referencia.      Ninguno
  mild            Deterioro moderado           PD x1.2, LGD+0.05, RW x1.1, col x0.95
  severe          Recesion severa              PD x1.5, LGD+0.15, RW x1.25, col x0.85, rates+200bps
  pricing_crunch  Crisis de liquidez           bid_haircut x1.3 (precios NPL 30% mas bajos)

5.2. Las 3 posturas de riesgo
------------------------------
  prudencial  : estabilidad y contencion de perdidas. Menos ventas, mas
                reestructuraciones.
  balanceado  : mejor trade-off EVA/capital. Punto medio en todas las metricas.
  desinversion: liberacion inmediata de capital y RWA. Mas ventas, mas capital
                liberado, RWA mas bajo.

Regla de monotonia esperada (validacion banco-ready):
  - N ventas:        Prudencial <= Balanceado <= Desinversion
  - Capital liberado: Prudencial <= Balanceado <= Desinversion
  - RWA final:       Desinversion <= Balanceado <= Prudencial

Se verifica automaticamente en reports/compare_postures.py y ci_local.bat
(step 3.5).

5.3. KPIs de ventas -- PC10
-----------------------------
  sale_pnl_total         : P&L total de ventas (negativo para NPLs).
  avg_sale_pnl           : P&L medio por operacion.
  avg_bid_pct_ead        : precio bid medio como fraccion del EAD (0.13 = 13%).
                           Baja en pricing_crunch (~0.10 vs ~0.13 en baseline).
  avg_bid_pct_ead_avail. : flag True si KPI calculado, False si no hay ventas.
  sell_blocked_count     : ventas rechazadas por fire-sale o floor prudencial.

NOTA: cuando Accion_final = VENDER, EVA_post = 0 (prestamo sale del balance).
El P&L de la venta vive en pnl_realized/sale_pnl_total, NO en EVA_post.
El total_eva_post solo suma MANTENER y REESTRUCTURAR.


================================================================================
6. SCRIPTS DE AUTOMATIZACION Y CI
================================================================================

6.1. Ficheros .bat
------------------
  run_3_postures_executability.bat : Ejecuta las 3 posturas con el coordinador.
                                     Recibe --tag.
  run_inference_only.bat           : Solo inferencia coordinada (sin train).
                                     Recibe --tag.
  run_pipeline.bat                 : Pipeline completo: train + inference + stress.
  run_recalibrated_inference.bat   : Inferencia con modelos recalibrados.
  run_repro_test.bat               : Test de reproducibilidad entre dos runs.
  smoke_test.bat                   : Smoke test rapido (cartera reducida, 1 escenario,
                                     1 postura).
  ci_local.bat                     : CI completo local con steps enumerados y
                                     validaciones QA.

ci_local.bat -- Steps del CI completo:
  Step 1   : Smoke test (cartera reducida, 1 escenario, 1 postura)
  Step 2   : pytest completo (suite de tests)
  Step 3   : Inferencia coordinada 3 posturas
  Step 3.5 : Validacion de monotonia (compare_postures)
  Step 3.6 : pytest pricing KPIs + validacion columnas PC10 en stress_summary
  Step 4   : Stress engine completo (4 escenarios x 3 posturas = 12 runs)
  Step 5   : Backtesting ligero vs. baselines
  Step 6   : Committee pack
  Salida   : logs/ci_local_<tag>_<timestamp>.log

6.2. main.py -- Subcomandos del orquestador central
----------------------------------------------------
  generate : data/generate_portfolio.py      -> data/portfolio_synth.xlsx
  train    : agent/train_subagents.py        -> models/best_model*.zip,
                                               models/vecnormalize*.pkl
  infer    : agent/policy_inference.py       -> reports/inference_<ts>/summary.csv,
                                               decisiones_explicadas.xlsx
  summary  : reports/results_summary.py      -> DataFrame consolidado, Excel,
                                               JSON, graficos, executive_summary.txt


================================================================================
7. ESTRUCTURA DE CARPETAS COMPLETA
================================================================================

  data/
  |-- portfolio_synth.xlsx            <- cartera sintetica
  |-- portfolio_synth_smoke.xlsx      <- cartera reducida para smoke tests
  |-- mappings/
  |   `-- real_portfolio_mapping.yaml <- mapeo columnas cartera real -> interno
  `-- __init__.py

  models/
  |-- best_model.zip                  <- PPO micro (LoanEnv)
  |-- best_model_loan.zip             <- PPO micro especializado
  |-- best_model_portfolio.zip        <- PPO macro (PortfolioEnv)
  |-- vecnormalize_final.pkl          <- VecNormalize agente micro
  |-- vecnormalize_loan.pkl           <- VecNormalize agente loan
  |-- vecnormalize_portfolio.pkl      <- VecNormalize agente macro
  |-- feature_order.json              <- orden canonico de features
  |-- obs_feature_order_loan.json     <- orden features entorno loan
  |-- training_metadata.json          <- metadatos entrenamiento micro
  |-- training_metadata_loan.json     <- metadatos entrenamiento loan
  `-- checkpoints/                    <- checkpoints periodicos PPO

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
      `-- run_prudencial/ run_balanceado/ run_desinversion/

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


================================================================================
8. TRAZABILIDAD Y REPRODUCIBILIDAD
================================================================================

El proyecto mantiene trazabilidad completa en todos los niveles:

  Nivel codigo    : Ramas Git por Punto de Control (PC). Rama activa:
                    feat/pc10-pricing-kpis-stress (HEAD: e51b40d).
                    main recibe merges aprobados por comite.

  Nivel ejecucion : Cada run genera timestamps unicos en nombres de carpetas.
                    MANIFEST.json registra commit exacto, estado del repo
                    (clean/dirty), checksums SHA-256 de modelos.

  Nivel decision  : overrides_log_<postura>.csv registra cada override
                    prudencial con su motivo. decisiones_audit_<postura>.csv
                    es el registro loan-by-loan completo.

  Nivel CI        : ci_local.bat ejecuta la suite completa con evidencias en
                    logs/qa_checkpoint<N>_evidence.txt.

Pack definitivo PC10: reports/committee_pack_pc10_hardening_final_20260221_202331/
  (commit=e51b40d, status=clean)

config.py es la pieza central que garantiza coherencia entre todos los
componentes: entornos RL, motores financieros, engines y agentes importan de un
unico fichero de configuracion, garantizando alineacion regulatoria y tecnica a
lo largo de toda la cadena.
