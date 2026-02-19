**POC — OPTIMIZADOR DE CARTERAS EN DEFAULT**
[![CI smoke](https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/joseeflb/optimizador_cartera_std/actions/workflows/smoke.yml)
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





* * = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = * *  
  
 * * 1 1 .   P O S T U R A S   Y   E V O L U C I �  N   ( B A N K - R E A D Y ) * *  
  
 * * = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = * *  
  
 E l   s i s t e m a   o p e r a   b a j o   3   p o s t u r a s   d e   r i e s g o   d i f e r e n c i a d a s ,   g a r a n t i z a n d o   a l i n e a c i � � n   c o n   e l   a p e t i t o   d e   r i e s g o   d e l   b a n c o :  
  
 1 .     * * P R U D E N C I A L   ( C o n s e r v a t i v e ) * * :   P r i o r i z a   e s t a b i l i d a d   a   l a r g o   p l a z o   y   c o n t e n c i � � n   d e   p � � r d i d a s .   R e s t r i n g e   v e n t a s   m a s i v a s   ( F i r e   S a l e s )   y   b u s c a   r e e s t r u c t u r a c i o n e s   s o s t e n i b l e s .  
         *       * O b j e t i v o * :   M a x i m i z a r   v a l o r   r e c u p e r a b l e   ( R e c o v e r y   R a t e )   s i n   s a c r i f i c a r   c a p i t a l   i n n e c e s a r i a m e n t e .  
 2 .     * * B A L A N C E A D O   ( B a l a n c e d ) * * :   P u n t o   m e d i o   o p t i m i z a d o .   B u s c a   e l   m e j o r   t r a d e - o f f   e n t r e   E V A   y   l i b e r a c i � � n   d e   c a p i t a l .  
         *       * O b j e t i v o * :   M a x i m i z a r   r e n t a b i l i d a d   a j u s t a d a   a l   r i e s g o   ( R A R O C / E V A ) .  
 3 .     * * D E S I N V E R S I O N   ( A g g r e s s i v e   /   R W A   R e l e a s e ) * * :   P r i o r i z a   l a   l i b e r a c i � � n   i n m e d i a t a   d e   c a p i t a l   y   r e d u c c i � � n   d e   R W A ,   a c e p t a n d o   m a y o r e s   d e s c u e n t o s   e n   v e n t a s   ( h a i r c u t s )   s i   e s   n e c e s a r i o .  
         *       * O b j e t i v o * :   M a x i m i z a r   C a p i t a l   R e l e a s e   y   m i n i m i z a r   R W A   f i n a l .  
  
 * * 1 1 . 1 .   L � � g i c a   M o n � � t o n a   E s p e r a d a   ( V a l i d a t i o n   C h e c k s ) * *  
  
 P a r a   c o n s i d e r a r   l a   e j e c u c i � � n   v � � l i d a   ( " B a n k - R e a d y " ) ,   s e   d e b e n   c u m p l i r   l a s   s i g u i e n t e s   r e g l a s   d e   n e g o c i o :  
  
 *       * * V e n t a s   ( N � �   O p e r a c i o n e s ) * * :   ` P r u d e n c i a l   < =   B a l a n c e a d o   < =   D e s i n v e r s i o n `  
 *       * * L i b e r a c i � � n   d e   C a p i t a l   ( �  � ) * * :   ` P r u d e n c i a l   < =   B a l a n c e a d o   < =   D e s i n v e r s i o n `  
 *       * * R W A   F i n a l   ( �  � ) * * :   ` D e s i n v e r s i o n   < =   B a l a n c e a d o   < =   P r u d e n c i a l `   ( D e s i n s v e r s i � � n   r e d u c e   m � � s   e l   R W A ) .  
  
 * * 1 1 . 2 .   E j e c u c i � � n   C o m p a r a t i v a * *  
  
 P a r a   g e n e r a r   y   v a l i d a r   l a s   3   p o s t u r a s   e n   u n   s o l o   p a s o :  
  
 1 .     * * E j e c u c i � � n * * :   ` . \ r u n _ 3 _ p o s t u r e s _ e x e c u t a b i l i t y _ v e n v . b a t   - - t a g   < n o m b r e _ r u n > `  
 2 .     * * C o m p a r a t i v a * * :   ` . \ . v e n v \ S c r i p t s \ p y t h o n   - m   r e p o r t s . c o m p a r e _ p o s t u r e s   - - t a g   < n o m b r e _ r u n > `  
         *       G e n e r a :   ` r e p o r t s / c o m p a r e _ p o s t u r e s _ < t a g > . c s v `  
 3 .     * * E v i d e n c i a   Q A * * :   ` l o g s / q a _ c h e c k p o i n t 5 _ e v i d e n c e . t x t `   ( i n c l u y e   r e s u l t a d o s   d e   t e s t s   a u t o m � � t i c o s ) .  
 