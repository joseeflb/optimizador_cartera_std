**POC — OPTIMIZADOR DE CARTERAS EN DEFAULT**
[![CI smoke](https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml)
Banco L1.5 · Basilea III Método Estándar · PC10 (feat/pc10-pricing-kpis-stress)



**======================================================================**



**1. VISIÓN GENERAL DEL PIPELINE**

**======================================================================**



El proyecto implementa un pipeline completo para optimizar carteras en default (NPL) bajo Basilea III método estándar, combinando:



* Motores financieros (reestructuración, pricing NPL, estrés, sensitividades).
* Entornos de aprendizaje por refuerzo (LoanEnv a nivel préstamo y PortfolioEnv a nivel cartera).
* Agentes PPO (micro y macro).
* Arquitectura multi-agente (coordinador macro ↔ micro) con overrides prudenciales explícitos.
* Orquestación de estrés multi-escenario (4) y multi-postura (3) = 12 ejecuciones por tag.
* Generación automática de committee packs con trazabilidad de commit git.
* Backtesting ligero de estrategias vs. baselines.
* Orquestación central mediante main.py.



El pipeline opera bajo 4 escenarios macro × 3 posturas de riesgo = 12 ejecuciones por tag.

Cada ejecución produce outputs estructurados en reports/ con trazabilidad completa:

KPIs de ventas (PC10), audit CSV loan-by-loan, JSON de KPIs de cartera, Excels de decisiones y overrides log.



El usuario final interactúa principalmente con:



* main.py (generate / train / infer / summary)
* data/generate\_portfolio.py (carteras sintéticas)
* data/ingest\_portfolio.py (ingestión y mapeo de cartera real)
* agent/train\_subagents.py (entrenamiento PPO micro y macro)
* agent/coordinator\_inference.py (inferencia coordinada multi-agente por postura)
* engines/stress\_engine.py (estrés multi-escenario completo)
* reports/make\_committee\_pack.py (empaquetado para comité con MANIFEST.json)
* reports/compare\_postures.py (comparativa 3 posturas)
* reports/backtesting\_light.py (backtesting ligero vs. baselines)
* ci\_local.bat / run\_3\_postures\_executability\_venv.bat (automatización CI)



======================================================================

**2. CONFIGURACIÓN GLOBAL Y ESQUELETO**

**2.1. config.py — Configuración Banco L1.5 (núcleo común)**



Define la configuración global del proyecto. Todos los módulos (entornos, motores, agentes, main) importan de config.py, garantizando coherencia regulatoria y técnica en toda la cadena:



* **Regulacion**: hurdle rate (RAROC mínimo), ratios CET1 / Total Capital mínimos, buffers (conservation, countercyclical).
* **BaselSTDMapping**: RW por segmento (hipotecario, consumo, corporate, PYME, soberano), rating y estado (performing / default).
* **ReestructuraParams**: rejillas de plazo (plazo\_anios\_grid), tasa (tasa\_anual\_grid), quita (quita\_grid) y umbrales de esfuerzo PTI / DSCR máximos.
* **SensibilidadReestructura**: reducción de PD / LGD tras reestructuración, ventana de curación (cure\_window), RW "cured", horizonte de análisis.
* **PrecioVentaParams**: haircuts por segmento, rango de precios bid, coste legal estimado, horizonte de recuperación.
* **RewardParams**: pesos de EVA, capital liberado, estabilidad EVA, concentración HHI, P\&L de ventas, penalizaciones de fire-sale y costes explícitos de carry.
* **PPOParams**: learning\_rate, batch\_size, n\_steps, gamma, clip\_range, ent\_coef.
* **EnvParams**: portfolio\_state\_dim, loan\_state\_dim, normalización, top-k, semillas globales.
* **BID\_HAIRCUT\_GLOBAL**: multiplicador global inyectado en el motor de pricing antes de cada ejecución del escenario pricing\_crunch.
* Rutas estándar: DATA\_PATH, MODELS\_PATH, REPORTS\_PATH, LOGS\_PATH, CHECKPOINTS\_PATH.



**2.2. configs/stress\_scenarios.yaml — Escenarios macro**



Define los 4 escenarios de estrés aplicados por engines/stress\_engine.py:



* **baseline**: sin shocks. Estado de referencia (shocks: {}).
* **mild**: deterioro económico moderado. PD\_mult=1.2, LGD\_add=0.05, RW\_mult=1.1, collateral\_mult=0.95.
* **severe**: recesión severa. PD\_mult=1.5, LGD\_add=0.15, RW\_mult=1.25, collateral\_mult=0.85, interest\_rate\_add=0.02 (+200 bps).
* **pricing\_crunch**: crisis de liquidez. bid\_haircut\_mult=1.3 (30% de descuento adicional sobre precios NPL). Se inyecta en BID\_HAIRCUT\_GLOBAL antes de cada run de coordinator\_inference, lo que reduce ventas, aumenta sell\_blocked\_count y activa guardas de fire-sale en el coordinador.



**2.3. data/mappings/real\_portfolio\_mapping.yaml — Mapeo de cartera real**



Define la traducción de columnas entre el fichero bruto del banco y el esquema interno del pipeline.

Se usa en data/ingest\_portfolio.py para transformar la cartera real a formato ingestible con tipado, renombrado y conversión de unidades.



======================================================================

**3. MOTORES FINANCIEROS (CEREBRO FINANCIERO)**

**3.1. optimizer/restructure\_optimizer.py**



Función principal: optimize\_restructure(...)



Decide sobre tres palancas:



* Plazo adicional (rejilla plazo\_anios\_grid).
* Nueva tasa anual (rejilla tasa\_anual\_grid).
* Quita sobre EAD (quita\_grid, según política interna).



Modeliza:



* Flujo de caja explícito (sistema francés).
* Asequibilidad del cliente mediante PTI / DSCR y esfuerzo máximo.
* Dinámica de riesgo PD / LGD tras reestructuración.
* Capital y provisiones a lo largo del horizonte (RWA/EL medios).



Objetivo: maximizar EVA post reestructuración y obtener métricas ΔEVA, ΔRORWA, etc.



Devuelve un diccionario estandarizado con, entre otros:

plazo\_optimo, tasa\_nueva, quita, EVA\_post, EVA\_gain, RWA\_post, RORWA\_post, PTI, DSCR.



Se usa directamente y también a través del wrapper multiagent/restruct\_agent.py.



**3.2. optimizer/price\_simulator.py**



Función principal: simulate\_npl\_price(...)



Entradas: EAD, LGD, PD o DPD, segmento, secured / unsecured.



Simula el precio de mercado para una venta NPL, coste legal, horizonte de recuperación y capital liberado.



El parámetro BID\_HAIRCUT\_GLOBAL de config.py amplifica los haircuts en el escenario pricing\_crunch.



Devuelve, entre otros campos:



* precio\_optimo (bid price)
* pnl (precio menos recovery ajustado por costes)
* capital\_liberado (RWA × capital\_ratio)
* price\_ratio\_ead (bid price / EAD): columna canónica para avg\_bid\_pct\_ead (PC10)
* percentiles de precio: p5, p50, p95
* indicadores de simulación y banderas de calidad



Se integra en PortfolioEnv.\_apply\_sell(...) y en el agente multi-agente de pricing.



**3.3. engines/stress\_engine.py — Motor de estrés multi-escenario**



Clase StressEngine y función orquestadora run\_stress\_pipeline(...).



Aplica shocks macro definidos en configs/stress\_scenarios.yaml:

PD y LGD aumentan en escenarios adversos; RW se deteriora; el DPD crece; collateral se deprecia.



Recalcula RWA, EL, NI, RORWA y EVA tras los shocks.



La función run\_stress\_pipeline(tag, scenarios, postures) orquesta el pipeline completo:



1. Lee configs/stress\_scenarios.yaml y obtiene los shocks por escenario.
2. Para cada (escenario × postura):

&nbsp;&nbsp; a. Inyecta BID\_HAIRCUT\_GLOBAL en config si escenario == pricing\_crunch.

&nbsp;&nbsp; b. Llama a agent/coordinator\_inference.py para ejecutar la inferencia completa.

&nbsp;&nbsp; c. Lee el fichero decisiones\_audit\_\<postura\>.csv generado.

&nbsp;&nbsp; d. Calcula los KPIs de ventas y cartera.

3. Genera stress\_summary\_\<tag\>.csv con una fila por (escenario, postura).



**KPIs exportados en stress\_summary\_\<tag\>.csv (PC10):**



* scenario, posture: identificadores de la ejecución.
* total\_ead: EAD total de la cartera.
* total\_eva\_post, total\_eva\_pre: EVA agregado tras/antes de las decisiones.
* total\_rwa\_post, total\_rwa\_pre: RWA agregado tras/antes.
* capital\_liberado: suma de capital liberado por ventas.
* n\_sales, n\_restruct, n\_mantener: nº de préstamos por acción final.
* sale\_pnl\_total: P\&L total realizado en ventas (suma de pnl\_realized).
* avg\_sale\_pnl: P\&L medio por venta (sale\_pnl\_total / n\_sales o NaN si n\_sales=0).
* avg\_bid\_pct\_ead: precio bid medio como % del EAD (NaN si no hay ventas o columna no disponible). Mapeado sobre price\_ratio\_ead del fichero de auditoría (columna canónica).
* avg\_bid\_pct\_ead\_available: booleano True si avg\_bid\_pct\_ead es un valor real calculado; False si es NaN.
* sell\_blocked\_count: nº de ventas bloqueadas por overrides prudenciales o guardas de fire-sale.



**Outputs en reports/ por ejecución del stress engine:**



```
reports/stress_<tag>_<timestamp>/
  baseline/
    prudencial/        ← carpeta de inferencia individual
    balanceado/
    desinversion/
    portfolio_baseline.xlsx     ← cartera final agregada (baseline × 3 posturas)
  mild/
    …
  severe/
    …
  pricing_crunch/
    …
  portfolio_mild.xlsx
  portfolio_severe.xlsx
  portfolio_pricing_crunch.xlsx

reports/stress_summary_<tag>.csv  ← KPIs agregados (12 filas: 4 escenarios × 3 posturas)
```



**3.4. engines/sensitivities\_engine.py**



Clase SensitivityEngine.



Calcula sensitividades de las principales métricas:



* Derivadas de NI respecto a PD, LGD y rate.
* Derivadas de EVA respecto a PD, LGD, rate y EAD.
* Derivadas de RWA y capital respecto a RW.
* Sensibilidad a la curación (EVA\_cured − EVA\_actual).
* global\_sensitivity\_score(loan): score compuesto para identificar préstamos "palanca".



Se puede usar para ranking en PortfolioEnv y para explicabilidad en policy\_inference.



======================================================================

**4. ENTORNOS RL (MICRO Y MACRO)**

**4.1. env/loan\_env.py — Entorno micro (nivel préstamo)**



Simula la evolución de un préstamo individual.



Acciones: MANTENER (0), REESTRUCTURAR (1), VENDER (2).



Observación: vector de 10 características definidas en FEATURE\_COLUMNS de config.py.

Las features incluyen: EAD, PD, LGD, RW, DPD, spread, segmento codificado, ingreso\_mensual, cashflow\_operativo\_mensual, loan\_to\_value.



Recompensa: combinación de ΔEVA, capital liberado, estabilidad y prudencia regulatoria.



Guarda logs en logs/portfolio\_env.log.



Uso en el pipeline:



* Entrenamiento del subagente micro mediante agent/train\_subagents.py.
* Inferencia loan-level con agent/policy\_inference.py.
* Evaluación contra baselines en reports/evaluate\_against\_baselines.py.



**4.2. env/portfolio\_env.py — Entorno macro (nivel cartera, NPL-ready)**



PortfolioEnv simula decisiones agregadas sobre una cartera de NPL.



Principales elementos:



Estado: vector de dimensión portfolio\_state\_dim con indicadores agregados (EVA total, RWA, riesgo esperado, HHI, volatilidad EVA, capital liberado acumulado, etc.).



Acciones (0–11): mantener todos, vender o reestructurar top-1 o top-K según EVA, RORWA o PTI, combinaciones vender+reestructurar, heurística baseline STD y no-op.



Dinámicas micro internas:



* \_apply\_maintain: drift NPL (empeoramiento gradual de PD/LGD/DPD, o mejora ligera si el préstamo está cured).
* \_apply\_restructure: reestructuración local con rejilla coherente con restructure\_optimizer.
* \_apply\_sell: venta NPL llamando a simulate\_npl\_price, cálculo de P\&L real, price\_ratio\_ead, capital liberado y update de pnl\_realized.



Recompensa macro:



Basada en: ΔEVA, capital liberado, P\&L de ventas, penalización de riesgo residual, volatilidad EVA, concentración HHI, coste de carry de capital y bonus prudenciales (evitar vender préstamos buenos, primar liberación de capital en carteras malas).



Guarda logs en logs/portfolio\_env.log.



Uso en el pipeline:



* Entrenamiento PPO macro mediante agent/train\_subagents.py.
* Inferencia macro mediante agent/policy\_inference\_portfolio.py.
* Inferencia coordinada mediante agent/coordinator\_inference.py.



======================================================================

**5. ENTRENAMIENTO DE AGENTES**

**5.1. agent/train\_subagents.py**



Entrena los subagentes micro (LoanEnv) y macro (PortfolioEnv).



Funciones principales:



* train\_loan\_agent(portfolio\_path, total\_timesteps, device):

&nbsp;&nbsp; Carga cartera (opcional), construye loan\_pool, crea LoanEnv con VecNormalize.

&nbsp;&nbsp; Entrena PPO con parámetros de CONFIG.ppo.

&nbsp;&nbsp; Guarda: models/best\_model.zip, models/vecnormalize\_final.pkl.

* train\_portfolio\_agent(portfolio\_path, total\_timesteps, device, top\_k, scenario):

&nbsp;&nbsp; Carga cartera desde Excel o CSV, crea PortfolioEnv(loans\_df=df, top\_k=top\_k, scenario=scenario).

&nbsp;&nbsp; Entrena PPO macro.

&nbsp;&nbsp; Guarda: models/best\_model\_portfolio.zip.

* Genera logs en logs/train\_subagents.log y checkpoints periódicos en models/checkpoints/.



CLI típico:



* Entrenar solo micro:

&nbsp;	python -m agent.train\_subagents --agent loan --portfolio data/portfolio\_synth.xlsx

* Entrenar solo macro:

&nbsp;	python -m agent.train\_subagents --agent portfolio --portfolio data/portfolio\_synth.xlsx

* Entrenar ambos:

&nbsp;	python -m agent.train\_subagents --agent both --portfolio data/portfolio\_synth.xlsx



**5.2. agent/train\_agent.py**



Script alternativo de entrenamiento avanzado:



* Integra VecNormalize en train y eval.
* BusinessEvalCallback: early stopping con métricas EVA + capital liberado.
* Genera checkpoints periódicos y logs para TensorBoard (logs/tb/ y logs/tensorboard/).



======================================================================

**6. INFERENCIA (POLÍTICAS ENTRENADAS)**

**6.1. agent/policy\_inference.py — Inferencia a nivel préstamo (PPO micro)**



Flujo:



1. Carga best\_model.zip y vecnormalize\_final.pkl (o best\_model\_loan.zip / vecnormalize\_loan.pkl para el agente loan especializado).
2. Carga la cartera de entrada desde Excel o CSV.
3. Reconstruye la observación en el orden definido por FEATURE\_COLUMNS.
4. Para cada préstamo:

&nbsp;&nbsp; Obtiene la acción del agente (MANTENER, REESTRUCTURAR, VENDER).

&nbsp;&nbsp; Calcula métricas financieras post-decisión (EVA\_post, RWA\_post, capital\_liberado, pnl).

&nbsp;&nbsp; Construye un summary con trazabilidad completa (Explain\_Steps, ΔEVA, capital liberado).

5. Exporta:

&nbsp;&nbsp; summary.csv: resultados loan-level.

&nbsp;&nbsp; decisiones\_explicadas.xlsx: Excel formateado con decisiones y métricas.



Guarda log en logs/policy\_inference.log.

Se invoca desde coordinator\_inference.py o directamente para debug loan-level.



**6.2. agent/policy\_inference\_portfolio.py — Inferencia a nivel cartera (PPO macro)**



Flujo:



1. Carga best\_model\_portfolio.zip.
2. Carga la cartera de entrada y crea PortfolioEnv(loans\_df=df, top\_k).
3. Ejecuta n\_steps de inferencia.
4. En cada paso: predice la acción macro, aplica env.step(action), obtiene reward y métricas agregadas.
5. Exporta:

&nbsp;&nbsp; trajectory.csv: historial de pasos y métricas macro (EVA, RWA, capital, reward).

&nbsp;&nbsp; summary\_portfolio.csv: resumen final de la cartera.

&nbsp;&nbsp; portfolio\_final.xlsx: cartera resultante tras la estrategia RL macro.



Guarda log en logs/policy\_inference\_portfolio.log.



**6.3. agent/coordinator\_inference.py — Inferencia coordinada multi-agente por postura**



Script central del pipeline de producción. Ejecuta la inferencia completa para una postura dada.



Flujo:



1. Carga configuración de postura y modelos (best\_model.zip, best\_model\_portfolio.zip, vecnormalize\_loan.pkl).
2. Instancia CoordinatorAgent, que combina PPO macro + RestructAgent + PricingAgent + overrides prudenciales.
3. Para cada préstamo de la cartera:

&nbsp;&nbsp; Obtiene acción macro del PPO (act\_macro).

&nbsp;&nbsp; Calcula sugerencia micro de reestructuración (RestructAgent).

&nbsp;&nbsp; Calcula sugerencia micro de pricing (PricingAgent).

&nbsp;&nbsp; Aplica overrides prudenciales (evita fire-sales, fuerza reestructuraciones con EVA\_gain alto).

&nbsp;&nbsp; Registra accion\_final, razon explicativa y métricas (EVA\_post, RWA\_post, PTI, DSCR, pnl\_realized, price\_ratio\_ead).

4. Exporta en reports/coordinated\_inference\_\<tag\>\_\<timestamp\>\_\<postura\>/:

&nbsp;&nbsp; decisiones\_finales\_\<postura\>.xlsx: Excel con todas las decisiones y métricas.

&nbsp;&nbsp; overrides\_log\_\<postura\>.csv: log de todos los overrides aplicados (ver schema en sección 11.3).

&nbsp;&nbsp; portfolio\_kpis\_\<postura\>.json: KPIs agregados de la cartera (n\_sales, n\_restruct, capital\_liberado, total\_eva\_post, etc.).

5. Dentro del flujo de stress\_engine, también genera:

&nbsp;&nbsp; decisiones\_audit\_\<postura\>.csv: CSV loan-by-loan con todas las columnas necesarias para el cálculo de KPIs de ventas (PC10).



Guarda log en logs/coordinator\_inference.log.



Invocación directa por postura:



&nbsp;	.\.venv\Scripts\python -m agent.coordinator\_inference --posture prudencial --tag mi\_run

&nbsp;	.\.venv\Scripts\python -m agent.coordinator\_inference --posture balanceado --tag mi\_run

&nbsp;	.\.venv\Scripts\python -m agent.coordinator\_inference --posture desinversion --tag mi\_run



======================================================================

**7. ARQUITECTURA MULTI-AGENTE**

**7.1. multiagent/restruct\_agent.py**



Wrapper del motor de reestructuración.

Normaliza entradas del préstamo (EAD, rate, PD, LGD, RW, ingreso\_mensual, cashflow\_operativo\_mensual).

Llama a optimize\_restructure.

Devuelve diccionario homogéneo: EVA\_post, EVA\_gain, RWA\_post, PTI, DSCR, plazo\_optimo, tasa\_nueva, quita, ok.



**7.2. multiagent/pricing\_agent.py**



Wrapper del motor de pricing simulate\_npl\_price.

Construye argumentos robustos a partir del préstamo (EAD, LGD, PD o DPD, segmento, secured).

Devuelve: precio\_optimo, pnl, capital\_liberado, p5, p50, p95, price\_ratio\_ead, ok.



**7.3. multiagent/coordinator\_agent.py**



Agente jerárquico que coordina:



* PPO macro (PortfolioEnv): act\_macro(portfolio\_obs).
* Reestructuración micro (RestructAgent): act\_micro(loan\_dict).
* Pricing micro (PricingAgent): act\_micro(loan\_dict).
* Reglas prudenciales Banco L1.5 basadas en EVA, RORWA, PTI, DSCR.



Método decide(portfolio\_obs, loan\_dict):



1. Obtiene acción macro PPO.
2. Calcula sugerencias micro (restruct y pricing).
3. Aplica overrides prudenciales:

&nbsp;&nbsp; Evita venta de préstamos con EVA\_post alto (buenos préstamos).

&nbsp;&nbsp; Fuerza reestructuración si EVA\_gain es muy alto y PTI es aceptable.

&nbsp;&nbsp; Bloquea ventas fire-sale si precio < floor prudencial.

Devuelve: accion\_macro, accion\_macro\_desc, micro\_restruct, micro\_price, accion\_final (MANTENER / REESTRUCTURAR / VENDER), razon (lista de mensajes explicativos).



Esta capa sirve como pieza central para un sistema explicable a comité.



======================================================================

**8. INGESTA, GENERACIÓN DE CARTERA Y BASELINES**

**8.1. data/generate\_portfolio.py**



Genera carteras sintéticas a partir de CONFIG.simulacion.

Mezcla de segmentos, distribuciones de rating, rangos de EAD, PD y LGD.

Funciones: generate\_portfolio(n) → DataFrame; export\_excel(df, out\_path) → Excel corporativo.

Salidas: data/portfolio\_synth.xlsx, data/portfolio\_synth\_smoke.xlsx.



**8.2. data/ingest\_portfolio.py**



Ingiere la cartera real del banco aplicando el mapeo de data/mappings/real\_portfolio\_mapping.yaml.

Transforma columnas brutas al esquema interno, aplica coerciones de tipo y validaciones.

Salida: DataFrame listo para ser consumido por coordinator\_inference o train\_subagents.



**8.3. baselines/baseline\_policies.py**



Define estrategias de referencia para evaluación comparativa:



* HouldAll: mantiene toda la cartera.
* SellAll: vende toda la cartera.
* RuleBasedNPL: reglas simples (vender si PD > umbral, reestructurar si PTI es bajo).



Se usan en reports/evaluate\_against\_baselines.py para medir la ganancia del agente RL vs. baseline.



======================================================================

**9. ORQUESTACIÓN DEL PIPELINE CON main.py**



main.py actúa como orquestador principal. Define cuatro subcomandos: generate / train / infer / summary.



**9.1. Generación de cartera sintética**



Comando:



&nbsp;	python main.py generate --n 1000 --out data/portfolio\_synth.xlsx



Scripts implicados: main.py (cmd\_generate) → data/generate\_portfolio.py.

Salida: data/portfolio\_synth.xlsx.



**9.2. Entrenamiento de subagentes RL**



Comando:



&nbsp;	python main.py train --agent both --portfolio data/portfolio\_synth.xlsx --total-steps 500000 --top-k 5 --scenario baseline



Scripts implicados: main.py (cmd\_train) → agent/train\_subagents.py → env/loan\_env.py y env/portfolio\_env.py.

Salidas: models/best\_model.zip, models/vecnormalize\_final.pkl, models/best\_model\_portfolio.zip.



**9.3. Inferencia loan-level (PPO micro)**



Comando:



&nbsp;	python main.py infer --model models/best\_model.zip --vecnorm models/vecnormalize\_final.pkl --portfolio data/portfolio\_synth.xlsx



Scripts implicados: main.py (cmd\_infer) → agent/policy\_inference.py.

Salida: reports/inference\_\<timestamp\>\_\<tag\>/ con summary.csv y decisiones\_explicadas.xlsx.



**9.4. Inferencia coordinada multi-agente (PPO micro + macro + overrides)**



Ejecutado directamente con agent/coordinator\_inference.py por postura, o desde run\_3\_postures\_executability\_venv.bat / stress\_engine para las 3 posturas de forma automática.

Salidas por postura:



* reports/coordinated\_inference\_\<tag\>\_\<timestamp\>\_\<postura\>/decisiones\_finales\_\<postura\>.xlsx
* reports/coordinated\_inference\_\<tag\>\_\<timestamp\>\_\<postura\>/overrides\_log\_\<postura\>.csv
* reports/coordinated\_inference\_\<tag\>\_\<timestamp\>\_\<postura\>/portfolio\_kpis\_\<postura\>.json



**9.5. Inferencia macro de cartera (PPO macro solo)**



&nbsp;	python -m agent.policy\_inference\_portfolio --portfolio data/portfolio\_synth.xlsx --tag mi\_run

Salida: reports/inference\_portfolio\_\<timestamp\>\_coord\_portfolio\_\<tag\>\_\<escenario\>/ con:



* trajectory.csv: secuencia de pasos con métricas macro (EVA, RWA, capital, reward, n\_sales acumulado).
* summary\_portfolio.csv: resumen final.
* portfolio\_final.xlsx: cartera resultante.



**9.6. Consolidación y reporting final**



Comando:



&nbsp;	python main.py summary --source reports --charts --excel --json --executive



Scripts implicados: main.py (cmd\_summary) → reports/results\_summary.py.

Salidas: DataFrame consolidado, Excel y JSON con métricas agregadas, gráficos, executive\_summary.txt.



======================================================================

**10. ESTRÉS MULTI-ESCENARIO (run\_stress\_pipeline)**

**10.1. Ejecución completa del stress engine**



Comando:



&nbsp;	.\.venv\Scripts\python -m engines.stress\_engine --tag pc10\_final\_clean



O a través del bat wrapper:



&nbsp;	.\run\_inference\_only\_venv.bat



Flujo interno:



1. Lee configs/stress\_scenarios.yaml → 4 escenarios.
2. Por cada (escenario × postura) = 12 combinaciones:

&nbsp;&nbsp; a. Aplica shocks macro a la cartera vía stress\_portfolio(df, scenario).

&nbsp;&nbsp; b. Para pricing\_crunch: inyecta BID\_HAIRCUT\_GLOBAL = bid\_haircut\_mult en config.
&nbsp;&nbsp; c. Ejecuta coordinator\_inference para las 3 posturas.

&nbsp;&nbsp; d. Lee decisiones\_audit\_\<postura\>.csv y calcula KPIs de ventas (PC10).

3. Construye stress\_summary\_\<tag\>.csv (12 filas).

4. Genera las carpetas de inferencia individuales y los portfolios por escenario.



**10.2. Estructura de salidas en reports/**



Después de una ejecución completa con tag mi\_run:



```
reports/
├── stress_summary_mi_run.csv                   ← KPIs agregados (12 filas)
├── stress_mi_run_<timestamp>/
│   ├── baseline/
│   │   ├── prudencial/
│   │   │   └── <timestamp>_mi_run_baseline_prudencial/
│   │   │       ├── decisiones_audit_prudencial.csv      ← audit CSV loan-by-loan
│   │   │       ├── decisiones_finales_prudencial.xlsx   ← Excel decisiones
│   │   │       ├── overrides_log_prudencial.csv         ← overrides aplicados
│   │   │       └── portfolio_kpis_prudencial.json       ← KPIs JSON de cartera
│   │   ├── balanceado/   (misma estructura)
│   │   └── desinversion/ (misma estructura)
│   ├── mild/             (misma estructura por postura)
│   ├── severe/           (misma estructura por postura)
│   ├── pricing_crunch/   (misma estructura por postura)
│   ├── portfolio_baseline.xlsx     ← cartera final agregada (baseline × 3 posturas)
│   ├── portfolio_mild.xlsx
│   ├── portfolio_severe.xlsx
│   └── portfolio_pricing_crunch.xlsx
└── compare_postures_mi_run.csv                ← comparativa 3 posturas (generado aparte)
```



**10.3. Comparativa de posturas**



Genera una fila por escenario con las 3 posturas en columnas para análisis side-by-side:



&nbsp;	.\.venv\Scripts\python -m reports.compare\_postures --tag mi\_run

Salida: reports/compare\_postures\_mi\_run.csv.

Columnas: scenario, n\_sales\_prudencial, n\_sales\_balanceado, n\_sales\_desinversion, capital\_liberado\_prudencial, … (monotonía esperada: Prudencial ≤ Balanceado ≤ Desinversión).



**10.4. Backtesting ligero**



&nbsp;	.\.venv\Scripts\python -m reports.backtesting\_light --tag mi\_run

Compara la estrategia del coordinador vs. las baselines (HoldAll, SellAll, RuleBasedNPL) para el escenario baseline.

Salidas: reports/backtesting\_light\_mi\_run.csv, reports/backtesting\_light\_mi\_run.md.



======================================================================

**11. ESTRUCTURA DE CARPETAS DE OUTPUTS**

**11.1. Carpetas y ficheros generados durante el pipeline completo**



```
data/
├── portfolio_synth.xlsx            ← cartera sintética generada por generate_portfolio.py
├── portfolio_synth_smoke.xlsx      ← cartera reducida para smoke tests
├── portfolio_bad.csv / .xlsx       ← cartera con casos extremos para tests
├── mappings/
│   └── real_portfolio_mapping.yaml ← mapeo de columnas de cartera real
└── __init__.py

models/
├── best_model.zip                  ← PPO micro (LoanEnv) entrenado
├── best_model_loan.zip             ← PPO micro especializado en loans
├── best_model_loan_vecnormalize.pkl← VecNormalize del agente loan
├── best_model_portfolio.zip        ← PPO macro (PortfolioEnv) entrenado
├── vecnormalize.pkl                ← VecNormalize del agente micro
├── vecnormalize_final.pkl          ← VecNormalize final del agente micro
├── vecnormalize_running.pkl        ← VecNormalize running (entrenamiento)
├── vecnormalize_portfolio.pkl      ← VecNormalize del agente macro
├── vecnormalize_loan.pkl           ← VecNormalize del agente loan
├── vecnormalize_loan.meta.json     ← metadatos del VecNormalize loan
├── feature_order.json              ← orden canónico de features para observación
├── obs_feature_order_loan.json     ← orden de features del entorno loan
├── training_metadata.json          ← metadatos de entrenamiento micro
├── training_metadata_loan.json     ← metadatos de entrenamiento loan
└── checkpoints/                    ← checkpoints periódicos PPO

reports/
├── stress_summary_<tag>.csv        ← tabla de KPIs agregados (4×3 filas)
├── compare_postures_<tag>.csv      ← comparativa de las 3 posturas
├── backtesting_light_<tag>.csv     ← backtesting vs. baselines
├── backtesting_light_<tag>.md      ← informe de backtesting en Markdown
├── evaluation_pc6.csv              ← evaluación fija PC6 (referencia)
├── evaluation_report.md            ← informe de evaluación
│
├── stress_<tag>_<timestamp>/       ← run completo del stress engine
│   ├── <escenario>/
│   │   ├── <postura>/
│   │   │   └── <timestamp>_<tag>_<escenario>_<postura>/
│   │   │       ├── decisiones_audit_<postura>.csv
│   │   │       ├── decisiones_finales_<postura>.xlsx
│   │   │       ├── overrides_log_<postura>.csv
│   │   │       └── portfolio_kpis_<postura>.json
│   │   └── …
│   └── portfolio_<escenario>.xlsx
│
├── coordinated_inference_<tag>_<timestamp>_<postura>/  ← inferencia individual
│   ├── decisiones_finales_<postura>.xlsx
│   ├── overrides_log_<postura>.csv
│   └── portfolio_kpis_<postura>.json
│
├── inference_<timestamp>_coord_micro_<tag>_<escenario>/  ← inferencia micro
│   ├── decisiones_explicadas.xlsx
│   └── summary.csv
│
├── inference_portfolio_<timestamp>_coord_portfolio_<tag>_<escenario>/
│   ├── trajectory.csv
│   ├── summary_portfolio.csv
│   └── portfolio_final.xlsx
│
└── committee_pack_<tag>_<timestamp>/   ← pack para comité
    ├── MANIFEST.json                   ← metadatos, checksums de modelos, artefactos
    ├── config.py
    ├── stress_summary_<tag>.csv
    ├── MEMO_COMMITTEE.md
    ├── RUNBOOK_COMMITTEE.md
    ├── CHECKLIST_COMMITTEE.md
    ├── README.md
    ├── evaluation_pc6.csv
    ├── evaluation_report.md
    ├── pip_freeze.txt
    ├── stress_scenarios.yaml
    ├── real_portfolio_mapping.yaml
    ├── ingest_portfolio.py
    ├── ci_local_<tag>.log
    ├── run_prudencial/
    │   └── decisiones_finales_prudencial.xlsx
    ├── run_balanceado/
    │   └── portfolio_final.xlsx
    └── run_desinversion/
        └── decisiones_finales_desinversion.xlsx

logs/
├── ci_local_<tag>_<timestamp>.log      ← log completo del CI local
├── coordinator_inference.log
├── stress_engine_<tag>.log
├── train_subagents.log
├── policy_inference.log
├── policy_inference_portfolio.log
├── qa_checkpoint<N>_evidence.txt       ← evidencias QA por checkpoint
├── tb/                                 ← logs TensorBoard (entrenamiento micro)
└── tensorboard/                        ← logs TensorBoard (entrenamiento portfolio)

configs/
└── stress_scenarios.yaml               ← definición de shocks por escenario
```



**11.2. Ficheros por ejecución de coordinator\_inference (detalle columnas)**



decisiones\_audit\_\<postura\>.csv — columnas clave:

loan\_id, EAD, PD, LGD, RW, DPD, segmento, Accion\_final (MANTENER/REESTRUCTURAR/VENDER), EVA\_pre, EVA\_post, RWA\_pre, RWA\_post, capital\_liberado, pnl\_realized, pnl\_book, pnl\_ratio\_book, price\_ratio\_ead (bid/EAD ~0.10–0.15 para NPLs), Price\_to\_EAD, PTI, DSCR, reason\_code, macro\_action\_id.



overrides\_log\_\<postura\>.csv — schema garantizado:

loan\_id, level, from\_action, to\_action, portfolio\_context, posture, run\_id, macro\_action\_used, macro\_rationales\_short, pti\_actual, dscr\_actual, pnl



portfolio\_kpis\_\<postura\>.json — campos:

n\_loans, n\_sales, n\_restruct, n\_mantener, total\_ead, total\_eva\_pre, total\_eva\_post, total\_rwa\_pre, total\_rwa\_post, capital\_liberado, sale\_pnl\_total, avg\_sale\_pnl, avg\_bid\_pct\_ead, avg\_bid\_pct\_ead\_available, sell\_blocked\_count



======================================================================

**12. KPIs DE VENTAS — PC10 (PRICING IMPACT)**



PC10 añade impacto económico observable al escenario pricing\_crunch y a los KPIs de ventas en el stress\_summary.



Columnas añadidas en stress\_summary\_\<tag\>.csv respecto a PC9:



* **sale\_pnl\_total**: P\&L total realizado en ventas del escenario/postura. Suma de pnl\_realized de todos los préstamos vendidos. Es negativo para ventas NPL (la cartera se vende a descuento). Unidades: euros (misma escala que EAD).
* **avg\_sale\_pnl**: P\&L medio por operación de venta (sale\_pnl\_total / n\_sales). NaN si n\_sales = 0.
* **avg\_bid\_pct\_ead**: precio bid medio expresado como fracción del EAD (ej: 0.15 = 15% del EAD). Calculado como media de price\_ratio\_ead sobre los préstamos vendidos. NaN si n\_sales = 0 o la columna no está disponible en el audit CSV.
* **avg\_bid\_pct\_ead\_available**: True si avg\_bid\_pct\_ead contiene un valor real calculado; False si es NaN. Permite distinguir "no hay ventas" de "error de datos".
* **sell\_blocked\_count**: número de operaciones bloqueadas por overrides prudenciales (guardas de fire-sale, floor de precio ≥ precio mínimo regulatorio).



**Semántica de EVA\_post en ventas:**

Cuando Accion\_final = VENDER, EVA\_post = 0 (el préstamo sale del balance). El P\&L de la venta vive en pnl\_realized (sale\_pnl\_total) y NO en EVA\_post. Esto es correcto regulatoriamente bajo Basilea III (el capital se libera, la posición desaparece del balance). El total\_eva\_post es la suma de EVA únicamente sobre MANTENER y REESTRUCTURAR.



**Interpretación pricing\_crunch:**

BID\_HAIRCUT\_GLOBAL = 1.3 hace que simulate\_npl\_price calcule precios 30% más bajos, lo que resulta en:

avg\_bid\_pct\_ead inferior en pricing\_crunch vs. baseline (ej: 0.10 vs. 0.13),

sell\_blocked\_count superior (más ventas bloqueadas por floor prudencial),

n\_sales inferior (el coordinador rechaza más ventas por fire-sale).



======================================================================

**13. POSTURAS DE RIESGO Y VALIDACIÓN BANCO-READY**

**13.1. Las 3 Posturas**



El sistema opera bajo 3 posturas de riesgo diferenciadas, garantizando alineación con el apetito de riesgo del banco:



1. **PRUDENCIAL (Conservative)**: Prioriza estabilidad a largo plazo y contención de pérdidas.

&nbsp;&nbsp; Restringe ventas masivas (Fire Sales) y busca reestructuraciones sostenibles.

&nbsp;&nbsp; Objetivo: Maximizar valor recuperable (Recovery Rate) sin sacrificar capital innecesariamente.

2. **BALANCEADO (Balanced)**: Punto medio optimizado. Busca el mejor trade-off entre EVA y liberación de capital.

&nbsp;&nbsp; Objetivo: Maximizar rentabilidad ajustada al riesgo (RAROC / EVA).

3. **DESINVERSION (Aggressive / RWA Release)**: Prioriza la liberación inmediata de capital y reducción de RWA,

&nbsp;&nbsp; aceptando mayores descuentos en ventas (haircuts) si es necesario.

&nbsp;&nbsp; Objetivo: Maximizar Capital Release y minimizar RWA final.



**13.2. Lógica Monótona Esperada (Validation Checks)**



Para considerar la ejecución válida ("Bank-Ready"), se deben cumplir las siguientes reglas de negocio:



* **Ventas (Nº Operaciones)**: `Prudencial <= Balanceado <= Desinversion`
* **Liberación de Capital (€)**: `Prudencial <= Balanceado <= Desinversion`
* **RWA Final (€)**: `Desinversion <= Balanceado <= Prudencial` (Desinversión reduce más el RWA)



Se verifica automáticamente en reports/compare\_postures.py y en ci\_local.bat (step 3.5).



**13.3. Ejecución Comparativa**



Para generar y validar las 3 posturas en un solo paso:



1. Ejecución: `.\run\_3\_postures\_executability\_venv.bat --tag <nombre\_run>`
2. Comparativa: `.\.venv\Scripts\python -m reports.compare\_postures --tag <nombre\_run>`

&nbsp;&nbsp; Genera: reports/compare\_postures\_\<tag\>.csv

3. Evidencia QA: logs/qa\_checkpoint5\_evidence.txt (incluye resultados de tests automáticos).



**13.4. Consistencia de Outputs y Evidencia (Audit Ready)**



Todos los reportes, especialmente overrides\_log\_\*.csv, mantienen un schema estricto y consistente en todas las posturas, garantizando la integridad de datos para herramientas de BI/Auditoría incluso cuando los reportes están vacíos.

Columnas garantizadas en overrides\_log:

loan\_id, level, from\_action, to\_action, portfolio\_context, posture, run\_id, macro\_action\_used, macro\_rationales\_short, pti\_actual, dscr\_actual, pnl

La evidencia de validación de schema y consistencia monótona se encuentra en: logs/qa\_checkpoint5\_evidence.txt.



======================================================================

**14. SCRIPTS DE LÍNEA DE COMANDOS Y CI**

**14.1. Ficheros .bat**



* **run\_3\_postures\_executability\_venv.bat**: Ejecuta las 3 posturas (prudencial, balanceado, desinversión) con el coordinador multi-agente. Recibe --tag para nombrar los outputs.
* **run\_inference\_only\_venv.bat**: Ejecuta únicamente la inferencia coordinada (sin entrenamiento). Recibe --tag.
* **run\_pipeline\_venv.bat**: Pipeline completo: train + inference + stress. Para entornos con modelos validados.
* **smoke\_test\_venv.bat**: Smoke test rápido (cartera reducida, 1 escenario, 1 postura). Verifica que el pipeline arranca y produce outputs.
* **ci\_local.bat**: CI completo local con steps enumerados y validaciones QA:

&nbsp;&nbsp; Step 1: Smoke test.

&nbsp;&nbsp; Step 2: pytest completo (45 tests).

&nbsp;&nbsp; Step 3: Inferencia coordinada 3 posturas.

&nbsp;&nbsp; Step 3.5: Validación monotonía (compare\_postures).

&nbsp;&nbsp; Step 3.6: pytest pricing KPIs + validación columnas PC10 en stress\_summary.

&nbsp;&nbsp; Step 4: Stress engine completo (4 escenarios × 3 posturas).

&nbsp;&nbsp; Step 5: Backtesting ligero.

&nbsp;&nbsp; Step 6: Committee pack.

&nbsp;&nbsp; Salida: logs/ci\_local\_\<tag\>\_\<timestamp\>.log



**14.2. Scripts Python clave en reports/**



* **reports/make\_committee\_pack.py**: Genera el committee pack.

&nbsp;&nbsp; Copia artefactos relevantes (CSVs de stress, Excels de decisiones, ficheros de configuración, logs de CI).

&nbsp;&nbsp; Genera MANIFEST.json con: timestamp, tag, commit git (HEAD), status git (clean/dirty), checksums SHA-256 de todos los modelos, lista de artefactos incluidos.

&nbsp;&nbsp; Uso: `.\.venv\Scripts\python reports\make\_committee\_pack.py --tag <tag>`

&nbsp;&nbsp; Salida: reports/committee\_pack\_\<tag\>\_\<timestamp\>/ (ver estructura en sección 11.1).

* **reports/compare\_postures.py**: Genera compare\_postures\_\<tag\>.csv con la comparativa de las 3 posturas.
* **reports/backtesting\_light.py**: Backtesting ligero vs. HoldAll, SellAll y RuleBasedNPL.
* **reports/evaluate\_against\_baselines.py**: Evaluación completa del agente vs. baselines con métricas de negocio.
* **reports/export\_financial\_decisions.py**: Exporta las decisiones financieras en formato corporativo Excel.



======================================================================

**15. RAMAS GIT Y TRAZABILIDAD**



El repositorio mantiene ramas separadas por Punto de Control (PC) para disciplina y trazabilidad:



* **feat/pc10-pricing-kpis-stress** (activa): PC10 — KPIs de ventas en stress\_summary, fix avg\_bid\_pct\_ead, availability flag. HEAD: ceb5dcf.
* **feat/pc9-realdata-calibration** (congelada): PC9 — calibración con cartera real, scoring con datos reales. HEAD: da811b5.
* **main**: rama de integración. Recibe merges de ramas PC aprobadas por comité.



El MANIFEST.json del committee pack registra el commit exacto (git rev-parse HEAD) y el estado del repo (clean/dirty) en el momento de generación, garantizando la reproducibilidad del pack.



El pack definitivo de PC10 es:

reports/committee\_pack\_pc10\_hardening\_final\_20260221\_125131/

(commit=ceb5dcf, status=clean).



======================================================================

**16. NOTA FINAL**



config.py es la pieza central que garantiza coherencia entre todos los componentes:



* Entornos RL (LoanEnv y PortfolioEnv).
* Motores financieros (restructure\_optimizer, price\_simulator).
* Engines (StressEngine, SensitivityEngine).
* Agentes de entrenamiento e inferencia (train\_subagents, policy\_inference, policy\_inference\_portfolio, coordinator\_inference).
* Arquitectura multi-agente (RestructAgent, PricingAgent, CoordinatorAgent).
* El orquestador principal (main.py).



Los engines y los agentes están diseñados para:



* Enriquecer la inferencia con lógica financiera realista (reestructuración, pricing, estrés, sensitividades).
* Construir demos multi-agente explicables para negocio y para riesgo (overrides log, audit CSV, JSON de KPIs).
* Producir métricas y trazabilidad regulatoria centradas en EVA, capital, curación, P\&L de ventas y estabilidad de la cartera.
* Garantizar reproducibilidad completa mediante committee packs con MANIFEST.json (commit git, checksums de modelos, artefactos).
