**POC — OPTIMIZADOR DE CARTERAS EN DEFAULT**

Banco L1.5 · Basilea III Método Estándar





**======================================================================**



**VISIÓN GENERAL DEL PIPELINE**

**======================================================================**



El proyecto implementa un pipeline completo para optimizar carteras en default (NPL) bajo Basilea III método estándar, combinando:



* Motores financieros (reestructuración, pricing NPL, estrés, sensitividades).

* Entornos de aprendizaje por refuerzo (LoanEnv a nivel préstamo y PortfolioEnv a nivel cartera).
* Agentes PPO (micro y macro).
* Arquitectura multi-agente (coordinador macro ↔ micro).
* Orquestación central mediante main.py.



El usuario final interactúa principalmente con:



* main.py
* data/generate\_portfolio.py
* agent/train\_subagents.py
* agent/policy\_inference.py
* agent/policy\_inference\_portfolio.py
* reports/results\_summary.py



======================================================================

**2. CONFIGURACIÓN GLOBAL Y ESQUELETO**

**2.1. config.py — Configuración Banco L1.5 (núcleo común)**



Define la configuración global del proyecto:

* Parámetros regulatorios (clase Regulacion): hurdle rate, ratios mínimos, buffers.
* Mapeo Basilea III STD (BaselSTDMapping): RW por segmento, rating y estado (performing / default).
* Parámetros de reestructuración (ReestructuraParams): rejillas de plazo, tasa, quita y umbrales de esfuerzo PTI.
* Dinámica de riesgo tras reestructuración (SensibilidadReestructura): reducción de PD/LGD, ventana de curación, RW “cured”, horizonte.
* Parámetros de precio de venta NPL (PrecioVentaParams).
* Parámetros de recompensa RL (RewardParams): pesos de EVA, capital liberado, estabilidad, concentración, P\&L de ventas, penalizaciones de fire-sale y costes explícitos.
* Parámetros PPO (PPOParams): hiperparámetros estándar de entrenamiento.
* Parámetros de entorno para LoanEnv y PortfolioEnv (dimensiones de estado, número de acciones, normalización, etc.).
* Semillas globales y rutas estándar a data, models, reports y logs.
* Todos los módulos (entornos, optimizadores, agentes, main) leen de config.py, lo que garantiza coherencia regulatoria y técnica en toda la cadena.



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



Devuelve, entre otros campos:



* precio\_optimo
* pnl (precio menos recovery)
* capital\_liberado



percentiles de precio (p5, p50, p95)



indicadores de simulación y banderas de calidad.



Se integra en PortfolioEnv.\_apply\_sell(...) y en el agente multi-agente de pricing.



**3.3. engines/stress\_engine.py**



Clase StressEngine.



Aplica shocks macro multi-periodo (6–8 trimestres) inspirados en ejercicios BCE/EBA:

PD y LGD aumentan en escenarios adversos; RW aumenta; el margen se deteriora; el DPD crece.



Recalcula RWA, EL, NI, RORWA y EVA tras cada periodo.



Modeliza también el efecto “cure” cuando PD cae por debajo de un umbral.



Métodos clave:



* \_scenario\_shock(scenario)
* apply\_macro\_shock\_once(loan, scenario)
* apply\_stress\_path(loan, scenario)
* stress\_portfolio(df, scenario)



Está pensado para análisis de estrés sobre carteras y para integrarse con el coordinador multi-agente.



**3.4. engines/sensitivities\_engine.py**



Clase SensitivityEngine.



Calcula sensitividades de las principales métricas:



* Derivadas de NI respecto a PD, LGD y rate.
* Derivadas de EVA respecto a PD, LGD, rate y EAD.
* Derivadas de RWA y capital respecto a RW.



Sensibilidad a la curación (EVA\_cured − EVA\_actual).



global\_sensitivity\_score(loan): score compuesto para identificar préstamos “palanca”.



Se puede usar para ranking en PortfolioEnv y para explicabilidad en policy\_inference.



======================================================================

**4. ENTORNOS RL (MICRO Y MACRO)**

**4.1. env/loan\_env.py — Entorno micro (nivel préstamo)**



Simula la evolución de un préstamo individual.



Acciones: MANTENER, REESTRUCTURAR, VENDER.



Observación: vector de 10 características definidas en FEATURE\_COLUMNS de config.py.



Recompensa: combinación de ΔEVA, capital liberado, estabilidad y prudencia regulatoria.



Uso en el pipeline:



Entrenamiento del subagente micro mediante agent/train\_subagents.py.



Inferencia loan-level con agent/policy\_inference.py.



**4.2. env/portfolio\_env.py — Entorno macro (nivel cartera, NPL-ready)**



PortfolioEnv simula decisiones agregadas sobre una cartera de NPL.



Principales elementos:



Estado: vector de dimensión portfolio\_state\_dim con indicadores agregados (EVA total, RWA, riesgo esperado, HHI, volatilidad EVA, etc.).



Acciones (0–11): mantener todos, vender o reestructurar top-1 o top-K según EVA, RORWA o PTI, combinaciones vender+reestructurar, heurística baseline STD y no-op.



Dinámicas micro internas:



* \_apply\_maintain: drift NPL (empeoramiento gradual de PD/LGD/DPD, o mejora ligera si el préstamo está cured).
* \_apply\_restructure: reestructuración local con rejilla coherente con restructure\_optimizer.
* \_apply\_sell: venta NPL llamando a simulate\_npl\_price, cálculo de P\&L real y capital liberado.



Recompensa macro:



Basada en: ΔEVA, capital liberado, P\&L de ventas, penalización de riesgo residual, volatilidad EVA, concentración, coste de carry de capital y bonus prudenciales (evitar vender préstamos buenos, primar liberación de capital en carteras malas, etc.).



Uso en el pipeline:



* Entrenamiento PPO macro mediante agent/train\_subagents.py.
* Inferencia macro mediante agent/policy\_inference\_portfolio.py.



======================================================================

**5. ENTRENAMIENTO DE AGENTES**

**5.1. agent/train\_subagents.py**



Entrena los subagentes:



* Subagente micro (LoanEnv)
* Subagente macro (PortfolioEnv)



Funciones principales:



* train\_loan\_agent(portfolio\_path, total\_timesteps, device)
* Carga cartera (opcional) y construye loan\_pool.
* Crea LoanEnv y un VecNormalize consistente con policy\_inference.
* Entrena PPO con parámetros de CONFIG.ppo.



Guarda:



* models/best\_model.zip
* models/vecnormalize\_final.pkl
* train\_portfolio\_agent(portfolio\_path, total\_timesteps, device, top\_k, scenario)
* Carga cartera desde Excel o CSV.
* Crea PortfolioEnv(loans\_df=df, top\_k=top\_k, scenario=scenario).
* Entrena PPO macro.



Guarda:



* models/best\_model\_portfolio.zip



CLI típico:



* Entrenar solo micro:

&nbsp;	python -m agent.train\_subagents --agent loan --portfolio data/portfolio\_synth.xlsx

* Entrenar solo macro:

&nbsp;	python -m agent.train\_subagents --agent portfolio --portfolio data/portfolio\_synth.xlsx

* Entrenar ambos:

&nbsp;	python -m agent.train\_subagents --agent both --portfolio data/portfolio\_synth.xlsx



**5.2. agent/train\_agent.py**



Script alternativo para entrenar PPO sobre LoanEnv o PortfolioEnv en un único flujo:



* Integra VecNormalize en train y eval.
* Usa un callback de negocio (BusinessEvalCallback) para early stopping con métricas EVA + capital liberado.
* Genera checkpoints periódicos y logs para TensorBoard.
* Aunque main.py usa train\_subagents.py, este script queda como versión unificada de entrenamiento avanzado.



======================================================================

**6. INFERENCIA (POLÍTICAS ENTRENADAS)**

**6.1. agent/policy\_inference.py — Inferencia a nivel préstamo**



Flujo:



* Carga el modelo PPO micro (best\_model.zip) y el normalizador (vecnormalize\_final.pkl).
* Carga la cartera de entrada desde Excel o CSV.
* Reconstruye la observación en el orden definido por FEATURE\_COLUMNS.
* Para cada préstamo:

&nbsp;	Obtiene la acción del agente (MANTENER, REESTRUCTURAR, VENDER).

&nbsp;	Calcula métricas financieras post-decisión.

&nbsp;	Construye un summary con trazabilidad (Explain\_Steps, ΔEVA, capital liberado, etc.).

* Exporta:

&nbsp;	summary.csv

&nbsp;	Excels formateados con los resultados loan-level.

&nbsp;	Invocación desde main.py, comando “infer” (ver apartado 9).



**6.2. agent/policy\_inference\_portfolio.py — Inferencia a nivel cartera**



Flujo:



* Carga el modelo PPO macro (best\_model\_portfolio.zip).
* Carga la cartera de entrada y crea PortfolioEnv(loans\_df=df, top\_k).
* Ejecuta n\_steps de inferencia
* En cada paso:

&nbsp;	predice la acción macro con el agente PPO,

&nbsp;	aplica env.step(action),

&nbsp;	obtiene reward y métricas agregadas.

* Exporta:

&nbsp;	trajectory.csv con la secuencia de decisiones macro y métricas.

&nbsp;	portfolio\_final.xlsx con la cartera resultante tras la estrategia RL macro.



======================================================================

**7. ARQUITECTURA MULTI-AGENTE**

**7.1. multiagent/restruct\_agent.py**



Wrapper del motor de reestructuración.



Normaliza entradas del préstamo (EAD, rate, PD, LGD, RW, ingreso\_mensual, cashflow\_operativo\_mensual).



Llama a optimize\_restructure.



Devuelve un diccionario homogéneo con:

EVA\_post, EVA\_gain, RWA\_post, PTI, DSCR, plazo\_optimo, tasa\_nueva, quita, ok, etc.



Permite utilizar la lógica de reestructuración desde otros módulos (por ejemplo, el coordinador multi-agente).



**7.2. multiagent/pricing\_agent.py**



Wrapper del motor de pricing simulate\_npl\_price.



Construye argumentos robustos a partir del préstamo (EAD, LGD, PD o DPD, segmento, secured).



Devuelve un diccionario estandarizado con:

precio\_optimo, pnl, capital\_liberado, p5, p50, p95, ok.



Se usa para sugerencias micro de venta NPL.



**7.3. multiagent/coordinator\_agent.py**



Agente jerárquico que coordina:



* PPO macro (PortfolioEnv).
* Reestructuración micro (RestructAgent).
* Pricing micro (PricingAgent).
* Opcionalmente StressEngine y SensitivityEngine.



Métodos:



* act\_macro(portfolio\_obs): devuelve la acción sugerida por el PPO macro y su descripción.
* act\_micro(loan\_dict): devuelve sugerencias micro de reestructuración y pricing.
* decide(portfolio\_obs, loan\_dict):



Combina:



* acción macro PPO,
* reglas prudenciales Banco L1.5 basadas en EVA, RORWA, PTI, DSCR,
* sugerencias micro (restruct y pricing).



Aplica overrides cuando sea necesario:



* evita la venta de préstamos “buenos”,
* fuerza reestructuraciones muy beneficiosas, etc.



Devuelve:



* accion\_macro (id),
* accion\_macro\_desc,
* micro\_restruct,
* micro\_price,
* accion\_final (MANTENER, REESTRUCTURAR o VENDER),
* razon (lista de mensajes explicativos).



Esta capa sirve como pieza central para un sistema explicable a comité.



======================================================================

**8. GENERACIÓN DE CARTERA Y REPORTING**

**8.1. data/generate\_portfolio.py**



Genera carteras sintéticas a partir de CONFIG.simulacion:



* Mezcla de segmentos.
* Distribuciones de rating.
* Rangos de EAD, PD y LGD.



Funciones:



* generate\_portfolio(n) → DataFrame.
* export\_excel(df, out\_path) → Excel corporativo.



Se invoca desde main.py con el comando “generate”.



8.2. reports/results\_summary.py



Consolida resultados loan-level y genera informes.



Funciones principales:



consolidate\_summaries(source\_dir):



* lee todos los summary.csv que encuentre bajo source\_dir,
* concatena en un único DataFrame.



compute\_ratios(df):



* calcula métricas agregadas por escenario, agente, etc.



export\_outputs(df, save\_excel, save\_json):



* exporta los resultados consolidados a Excel y JSON.



generate\_executive\_summary(df, out\_txt):



* genera un resumen ejecutivo en texto plano con los principales indicadores.



plot\_metric\_evolution(...) y plot\_comparison\_bars(...):



* generan gráficos de evolución y comparación de métricas EVA, RWA, capital liberado, etc.



======================================================================

**9. ORQUESTACIÓN DEL PIPELINE CON main.py**



main.py actúa como orquestador principal. Define cuatro subcomandos:



* generate
* train
* infer
* summary



**9.1. Generación de cartera sintética**



Comando:



python main.py generate \\

&nbsp;   --n 1000 \\

&nbsp;   --out data/portfolio\_synth.xlsx





Scripts implicados:



* main.py (función cmd\_generate).
* data/generate\_portfolio.py.



Salida:



Fichero data/portfolio\_synth.xlsx con la cartera sintética.



**9.2. Entrenamiento de subagentes RL**



Comando típico:



python main.py train \\

&nbsp;   --agent both \\

&nbsp;   --portfolio data/portfolio\_synth.xlsx \\

&nbsp;   --total-steps 500000 \\

&nbsp;   --top-k 5 \\

&nbsp;   --scenario baseline





Scripts implicados:



* main.py (función cmd\_train).
* agent/train\_subagents.py.
* env/loan\_env.py y env/portfolio\_env.py.



Salida:



* models/best\_model.zip (subagente LoanEnv).
* models/vecnormalize\_final.pkl (normalizador micro).
* models/best\_model\_portfolio.zip (subagente PortfolioEnv).



**9.3. Inferencia loan-level (PPO micro)**



Comando:



python main.py infer \\

&nbsp;   --model models/best\_model.zip \\

&nbsp;   --vecnorm models/vecnormalize\_final.pkl \\

&nbsp;   --portfolio data/portfolio\_synth.xlsx





Scripts implicados:



* main.py (función cmd\_infer).
* agent/policy\_inference.py.



Salida:



Carpeta en reports/… con summary.csv y Excels formateados con las decisiones y métricas por préstamo.



**9.4. Inferencia macro de cartera (PPO macro)**



Esta parte se ejecuta directamente con agent/policy\_inference\_portfolio.py (no está aún colgada como subcomando de main.py, pero se puede integrar fácilmente).



Flujo:



* Carga best\_model\_portfolio.zip.
* Crea PortfolioEnv con la cartera de entrada.
* Ejecuta n\_steps de decisiones macro.



Salida:



reports/inference\_portfolio\_… con:



* trajectory.csv (historial de pasos y métricas macro).
* portfolio\_final.xlsx (cartera resultante).



9.5. Consolidación y reporting final



Comando:



python main.py summary \\

&nbsp;   --source reports \\

&nbsp;   --charts \\

&nbsp;   --excel \\

&nbsp;   --json \\

&nbsp;   --executive





Scripts implicados:



* main.py (función cmd\_summary).
* reports/results\_summary.py.



Salida:



* DataFrame consolidado de todos los summary.csv.
* Excel y JSON con métricas agregadas.
* Gráficos de evolución y comparación.
* executive\_summary.txt preparado para presentación a comité.



======================================================================

**10. NOTA FINAL**



config.py es la pieza central que garantiza coherencia entre:



* Entornos RL (LoanEnv y PortfolioEnv).
* Motores financieros (restructure\_optimizer, price\_simulator).
* Engines (StressEngine, SensitivityEngine).
* Agentes de entrenamiento e inferencia (train\_subagents, policy\_inference, policy\_inference\_portfolio).
* Arquitectura multi-agente (RestructAgent, PricingAgent, CoordinatorAgent).
* El orquestador principal (main.py).



Los engines y los agentes están diseñados para:



* Enriquecer la inferencia con lógica financiera realista (reestructuración, pricing, estrés, sensitividades).
* Construir demos multi-agente explicables para negocio y para riesgo.
* Producir métricas y trazabilidad regulatoria centradas en EVA, capital, curación y estabilidad de la cartera.





