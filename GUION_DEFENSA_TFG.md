# GUIÓN DE DEFENSA — TFG
## Gestión óptima de carteras en default con Reinforcement Learning y optimización matemática

**Duración objetivo:** 12–14 minutos · **Ritmo:** ~150 palabras/min · **Total:** ~1.900 palabras

> **Cómo usar este guión:**
> Memoriza las **frases en negrita** como anclas. El resto es relleno que puedes parafrasear. Los **[TIEMPOS]** te indican cuándo estar en cada bloque. Si te quedas en blanco, vuelve a la frase ancla más cercana.

---

## 0. APERTURA — [0:00 → 0:45] · *45 segundos*

> **Buenos días. Mi nombre es [NOMBRE] y voy a defender el Trabajo Fin de Grado titulado *"Gestión óptima de carteras en default con Reinforcement Learning y optimización matemática"*, dirigido por [TUTOR].**

El trabajo aborda un problema real de la banca: **cómo decidir, préstamo a préstamo, qué hacer con una cartera de créditos morosos —los llamados NPL, Non-Performing Loans— de forma que se maximice el valor económico cumpliendo en todo momento Basilea III en su método estándar.**

Voy a estructurar la defensa en cinco bloques: **motivación y problema, marco regulatorio, arquitectura técnica, resultados experimentales y conclusiones.**

---

## 1. MOTIVACIÓN Y PROBLEMA — [0:45 → 2:30] · *1 minuto 45 s*

> **El problema de partida es el siguiente: tras la pandemia y el endurecimiento de tipos, los bancos europeos acumulan carteras NPL que destruyen valor todos los trimestres.**

Hoy, la gestión típica de estas carteras se hace con **reglas heurísticas fijas** —"si la PD supera tal umbral, vender; si no, mantener"— o con **subastas en bloque** que sacrifican mucho valor por la prisa. Ambas alternativas tienen dos defectos:

1. **No optimizan**: dejan EVA —*Economic Value Added*— sobre la mesa.
2. **No se adaptan**: una regla calibrada en 2019 deja de funcionar en un escenario de crisis severa.

> **La hipótesis del trabajo es que un agente de Reinforcement Learning, combinado con optimización determinista y guardrails regulatorios, puede dominar a las heurísticas en términos de EVA, capital liberado y robustez ante estrés, manteniendo la trazabilidad que exige un Comité de Crédito.**

El objetivo es por tanto construir un **sistema end-to-end**, "bank-ready", que:

- Decida por préstamo entre **mantener, reestructurar o vender**.
- Lo haga bajo **tres posturas de riesgo simultáneas**: prudencial, balanceada y desinversión.
- Resista **cuatro escenarios de estrés**: baseline, moderado, severo y *pricing crunch*.
- Y entregue **evidencia auditable** —Excel, JSON y logs— para cada decisión.

---

## 2. MARCO TEÓRICO Y REGULATORIO — [2:30 → 4:15] · *1 minuto 45 s*

> **El sistema se apoya en tres pilares teóricos: Basilea III estándar, Reinforcement Learning con PPO y optimización financiera clásica.**

### 2.1 Basilea III — método estándar

El RWA —*Risk Weighted Assets*— se calcula como:

$$\text{RWA} = \text{EAD} \times \text{RW}$$

donde **EAD** es la exposición y **RW** la ponderación regulatoria por rating. La pérdida esperada es:

$$\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

Y la métrica que el agente realmente optimiza es el **EVA**:

$$\text{EVA} = \text{Ingresos netos} - \text{EL} - k \cdot \text{RWA}$$

donde $k$ es el coste de capital regulatorio. **El EVA es el lenguaje del Comité: positivo significa que se crea valor para el accionista.**

### 2.2 Reinforcement Learning — PPO

Modelizo cada préstamo como un **proceso de decisión de Markov**: estado $s_t$, acción $a_t$, recompensa $r_t$ y transición. Uso **PPO —Proximal Policy Optimization—** porque es estable, on-policy, y soporta espacios de acción discretos y multi-discretos, que es exactamente lo que necesito.

### 2.3 Optimización determinista

La parte de reestructuración no se aprende: **se resuelve como un problema de NPV máximo sujeto a PTI ≤ 0,45 y DSCR ≥ 1,10** —*Payment-to-Income* y *Debt-Service-Coverage*—. Esto garantiza que la cuota propuesta es legalmente sostenible para el deudor.

---

## 3. ARQUITECTURA DEL SISTEMA — [4:15 → 7:30] · *3 minutos 15 s*

> **La arquitectura tiene cuatro capas: datos, agentes, coordinador y reporting. Es un pipeline jerárquico micro-macro con guardrails duros.**

### 3.1 Capa de datos

Entran carteras reales en Excel/CSV, mapeadas mediante un **YAML flexible** —`real_portfolio_mapping.yaml`— al esquema interno. Hay también un **generador sintético** de 500 préstamos coherentes con la configuración de PD, LGD, ratings y segmentos. La ingesta es **estricta por defecto**: PD fuera de [0,1] aborta el pipeline con error accionable; no hay clipping silencioso.

### 3.2 Dos agentes PPO jerárquicos

> **Aquí está la primera contribución técnica: en lugar de un único agente, uso dos agentes PPO coordinados.**

- **Agente MICRO** —entorno `LoanEnv`—. Observa un vector de 10 *features* por préstamo: EAD, PD, LGD, RW, EVA, RONA, RORWA, rating, segmento y DPD. **Decide entre tres acciones: mantener, reestructurar o vender.** La recompensa es el **delta de EVA más bonificaciones regulatorias** por liberar capital sin violar restricciones.

- **Agente MACRO** —entorno `PortfolioEnv`—. Observa la cartera completa y elige un **top-K** de préstamos sobre los que actuar. **Resuelve el problema de asignación de presupuesto y concentración** que el agente micro no ve.

Ambos entrenan con **VecNormalize** para normalizar observaciones y recompensas, y se monitorizan en **TensorBoard**.

### 3.3 Coordinador, posturas y guardrails

> **Sobre los dos agentes hay una capa determinista que es la que hace al sistema bank-ready.**

- **El coordinador** resuelve conflictos entre micro y macro con prioridad `PRUDENCIAL_FIRST`: si discrepan, gana la opción más conservadora.
- **Las tres posturas** —prudencial, balanceada, desinversión— son **modulaciones del coordinador**: cambian los caps de venta, los floors de precio y los pesos del reward. **No se reentrenan los modelos: se reusan**, lo cual es clave para la trazabilidad.
- **Los guardrails** son restricciones duras: protección anti–*fire-sale* (no vender por debajo de un precio mínimo), capital liberado mínimo por venta, *acceptance score*, y *volume caps* por postura.

### 3.4 Estrés y backtesting

Las inferencias se repiten bajo cuatro **escenarios de estrés** definidos en `stress_scenarios.yaml`. Cada escenario aplica multiplicadores: `PD_mult`, `LGD_add`, `RW_mult`, `collateral_mult` y `bid_haircut_mult` para el *pricing crunch*. **Distinguo dos cosas que no son lo mismo:**

- **Stress engine**: el agente **re-decide** sabiendo el shock. Mide el *valor de la información*.
- **Backtesting light**: las decisiones se **congelan** en baseline y se recalculan KPIs bajo shock. Mide la **fragilidad residual** de la estrategia ya tomada.

---

## 4. RESULTADOS — [7:30 → 10:30] · *3 minutos*

> **Voy a mostrar los resultados sobre 500 préstamos NPL con EAD total de 4.077 millones de euros. El benchmark "no acción" tiene un EVA de -712 millones; cualquier estrategia razonable debería superarlo.**

### 4.1 Tabla principal — baseline

| Postura | EVA Post | RWA Post | Capital liberado | Ventas | Reestructs |
|---|---|---|---|---|---|
| **Prudencial** | +748,6 M € | 3.404,8 M | +283,0 M | 55 | 0 |
| **Balanceada** | +725,9 M € | 3.392,9 M | +283,1 M | 168 | 244 |
| **Desinversión** | +48,5 K € | 508,4 K | +641,1 M | 408 | 5 |

> **Observen tres cosas clave:**

1. **Las tres posturas generan EVA positivo** frente al benchmark negativo.
2. **Hay monotonía perfecta**: ventas Prudencial ≤ Balanceada ≤ Desinversión, capital liberado en el mismo orden, y RWA en el orden inverso. **Esto no es casualidad: es la validación de que el sistema entiende lo que significa cada postura.**
3. **La Desinversión libera más del doble de capital** que la Prudencial, a costa de sacrificar EVA. Es la decisión correcta cuando el banco necesita ratio de capital.

### 4.2 Robustez bajo estrés severo

> **El hallazgo más importante del trabajo está aquí.**

Bajo escenario **severe**, la postura Prudencial reduce las ventas de **55 a solo 1**. ¿Por qué? Porque los precios de mercado se desploman, los *fire-sale guardrails* bloquean automáticamente las ventas por debajo del *sale floor ratio*, **y el agente reconoce que vender a precio insultante destruye más valor que mantener.**

> **Esto es el comportamiento que querría ver un regulador: el sistema no liquida a cualquier precio en mitad de una crisis.**

### 4.3 Comparación contra baselines

He comparado el agente contra cuatro políticas: aleatoria, *keep-all*, *sell-all* y una **rule-based NPL** profesional. **El agente RL domina a las cuatro en EVA y en eficiencia respecto a la cota óptima teórica $V^*$**, que calculo por separado con un script de cota superior.

### 4.4 Reproducibilidad

> **Esto no es un experimento puntual: el pipeline está en CI.**

El comando `ci_local.bat pc9_final` ejecuta el pipeline completo —smoke test, tres posturas, doce runs de estrés, backtesting, comparativa, evaluación contra baselines, *committee pack* y 26 tests de `pytest`— y sale con **EXIT 0** en el log que está incluido como evidencia. Cada artefacto lleva *tag* y *checksum*.

---

## 5. CONCLUSIONES Y LÍNEAS FUTURAS — [10:30 → 12:30] · *2 minutos*

> **Las conclusiones del trabajo son cuatro:**

1. **Es factible combinar RL con optimización determinista y guardrails regulatorios en un único pipeline auditable.** La arquitectura micro-macro con coordinador resuelve la tensión clásica entre flexibilidad del agente y trazabilidad del Comité.

2. **El sistema bate a las heurísticas y baselines en EVA, capital liberado y robustez**, con las tres posturas mostrando monotonía coherente —que es el test que cualquier validador interno aplicaría.

3. **La separación entre stress engine y backtesting es metodológicamente importante**: una mide *el valor de la información* del agente, la otra la *fragilidad residual* de la estrategia. Confundirlas, como hace gran parte de la literatura aplicada, lleva a conclusiones erróneas.

4. **El pipeline es bank-ready en el sentido operativo del término**: ingesta estricta, sin fallbacks silenciosos, CI verde, evidencia versionada y *committee pack* automático para Comité de Crédito.

### Líneas futuras

> **Tres extensiones naturales:**

- **Migrar del método estándar al IRB** —*Internal Ratings-Based*—, lo que implicaría modelizar PD y LGD propias y validarlas con backtesting de un año.
- **Incorporar datos macro exógenos** —tipos de interés, paro, índice inmobiliario— como features del estado, en lugar de aplicarlos solo como shocks.
- **Pasar de PPO a un esquema *offline RL*** entrenado sobre histórico real de un banco, con técnicas tipo CQL para evitar extrapolación fuera de distribución.

---

## 6. CIERRE — [12:30 → 13:00] · *30 segundos*

> **En resumen: el trabajo demuestra que es posible construir un optimizador de carteras NPL con Reinforcement Learning que sea, a la vez, financieramente competitivo, regulatoriamente correcto y operativamente auditable. Todo el código, los modelos entrenados y la evidencia experimental están versionados en el repositorio y ejecutables con un único comando.**

> **Muchas gracias por su atención. Quedo a su disposición para las preguntas que el tribunal estime oportunas.**

---

# ANEXO A — RESPUESTAS PREPARADAS A PREGUNTAS PROBABLES

### A.1 — "¿Por qué PPO y no DQN, A3C o SAC?"
PPO es **on-policy, estable, soporta acciones discretas y multi-discretas, y tiene un único hiperparámetro crítico** —el *clip*— lo que facilita la reproducibilidad. DQN tiene problemas con espacios multi-discretos como el del agente macro. SAC es para acciones continuas. A3C es más inestable. PPO es el estándar de facto en RL aplicado y permite justificar las decisiones ante un Comité.

### A.2 — "¿Cómo evita el overfitting si entrena con cartera sintética?"
El generador sintético está **calibrado a distribuciones reales de PD, LGD, RW y segmentos**. Además, valido la generalización (a) sobre cartera real cuando está disponible vía el ingestor con mapping YAML, (b) bajo cuatro escenarios de estrés que mueven los parámetros fuera de la distribución de entrenamiento, y (c) contra baselines y la cota teórica $V^*$.

### A.3 — "¿Qué pasa si el banco cambia de método estándar a IRB?"
La arquitectura es agnóstica al cálculo de RW. Cambiar a IRB implica sustituir la función `compute_rw()` por una basada en la fórmula de Basilea para IRB —que depende de PD, LGD, M y correlación regulatoria—. El resto del pipeline —agentes, guardrails, posturas— no cambia.

### A.4 — "¿Cómo justifica la elección del coste de capital $k$?"
Es un parámetro calibrable en `config.py`. En el caso base uso el valor estándar de mercado para banca europea (entre el 8% y el 12% de RWA). He hecho análisis de sensibilidad —`sensitivities_engine.py`— y los rankings entre posturas son estables.

### A.5 — "¿La acción 'reestructurar' es realmente aprendida o es determinista?"
**Mixto.** El agente micro decide *si* reestructurar; pero los **términos óptimos** —haircut, plazo, step-up— se resuelven con el optimizador determinista de NPV bajo PTI/DSCR. Esto es deliberado: las restricciones legales de sostenibilidad de cuota no son negociables y no deben aprenderse.

### A.6 — "¿Por qué tres posturas y no una sola política óptima?"
Porque en banca real **no existe una única función objetivo**: hay trimestres en los que el banco prioriza EVA y otros en los que prioriza ratio de capital o reducción de exposición. Las tres posturas se ofrecen al Comité como **menú de decisión**, no como recomendación única. **El agente provee, el Comité decide.**

### A.7 — "¿Qué garantías de auditabilidad da el sistema?"
Cada decisión genera (a) **Excel canónico** `decisiones_finales_<postura>.xlsx`, (b) **JSON de auditoría** con el log de enforcement de guardrails, (c) **markdown** `POSTURE_ANALYSIS` y `CIB_GOVERNANCE`, y (d) **checksums** en el *committee pack*. El pipeline CI registra log completo con tag y fecha.

### A.8 — "¿Cuál es la principal limitación del trabajo?"
**Tres limitaciones honestas:**
1. La validación es sobre cartera sintética calibrada; con cartera real de un banco concreto los hiperparámetros tendrían que recalibrarse.
2. PPO es on-policy: cada cambio de configuración exige reentrenar. Un esquema offline sería más práctico en producción.
3. No modelo el riesgo legal/operacional de la reestructuración —solo el financiero—.

### A.9 — "¿Qué aporta esto frente a publicaciones académicas previas?"
La literatura tiende a separar (a) trabajos de RL en finanzas sin guardrails ni regulación, y (b) trabajos de gestión NPL puramente deterministas. **Mi contribución es la integración**: un único pipeline con RL + optimización + Basilea + posturas + estrés + CI, todo trazable. Es ingeniería más que ciencia, pero es la ingeniería que faltaba.

---

# ANEXO B — GLOSARIO RÁPIDO (por si el tribunal pregunta)

- **NPL**: *Non-Performing Loan*, préstamo con DPD ≥ 90 días.
- **EAD**: *Exposure At Default*. Exposición en el momento del default.
- **PD / LGD / RW**: Probabilidad de Default / Pérdida en caso de Default / Ponderación de Riesgo.
- **EVA**: *Economic Value Added*. Ingresos netos menos EL menos coste de capital.
- **RWA**: *Risk-Weighted Assets*. EAD × RW.
- **PTI / DSCR**: *Payment-to-Income* / *Debt-Service-Coverage Ratio*.
- **PPO**: *Proximal Policy Optimization*. Algoritmo de RL.
- **Guardrails**: restricciones duras no aprendidas, aplicadas tras la decisión del agente.
- **Fire-sale**: venta forzosa a precio destruido por urgencia o pánico.
- **Stress engine vs backtesting**: re-decidir bajo shock vs. evaluar decisiones congeladas bajo shock.

---

# ANEXO C — CHECKLIST PRE-DEFENSA

- [ ] Llevar **dos copias impresas** del Excel `decisiones_finales_balanceado.xlsx` por si preguntan por una decisión concreta.
- [ ] Tener abierto en pantalla **el log de CI** (`ci_local_pc9_final_*.log`) con EXIT 0 visible.
- [ ] Memorizar las **tres cifras ancla**: EVA Prudencial **+748,6 M €**, ventas Prudencial bajo severe **= 1**, EAD total **4.077 M €**.
- [ ] Practicar la apertura y el cierre **palabra por palabra**.
- [ ] El resto: parafrasear con seguridad usando las frases en negrita como ancla.

**¡Suerte!**
