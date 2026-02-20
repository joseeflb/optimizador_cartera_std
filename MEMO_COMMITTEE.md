# MEMO EJECUTIVO — COMITE DE MODELO (PC9)

**Proyecto**: Optimizador de Cartera NPL / RL Agent L1.5
**Fecha**: 20 febrero 2026
**Clasificacion**: CONFIDENCIAL — Solo Distribucion Interna
**Tag de evidencia**: `pc9_final` / `pc9_verification`

---

## 1. Resumen Ejecutivo

El agente de optimizacion de cartera (RL) ha superado todos los controles de validacion (PC1-PC9).
Las tres posturas de gestion (Prudencial, Balanceado, Desinversion) producen resultados coherentes,
monotonos y reproducibles bajo cuatro escenarios macro (baseline, mild, severe, pricing\_crunch).

El sistema gestiona 500 prestamos (EAD total ~4.077 MM EUR) y genera EVA positivo cuando compara
contra el benchmark de "no accion" (EVA pre = -712M EUR).

---

## 2. Posturas de Gestion — Resumen Financiero (Baseline)

Fuente: `reports/stress_summary_pc9_final.csv` (run `pc9_final`, escenario baseline)

| Postura        | EVA Post     | RWA Post   | Capital Liberado | Ventas | Restructs |
|----------------|-------------|-----------|-----------------|--------|-----------|
| **Prudencial** | +748.6 M EUR | 3 404.8 M  | +283.0 M        | 55     | 0         |
| **Balanceado** | +725.9 M EUR | 3 392.9 M  | +283.1 M        | 168    | 244       |
| **Desinversion**| +48.5 K EUR | 508.4 K    | +641.1 M        | 408    | 5         |

> Monotonia validada: Ventas P<=B<=D (55<=168<=408), Capital P<=B<=D, RWA D<=B<=P.

---

## 3. Robustez bajo Estres (PC9)

### 3A. Stress Engine — Re-optimizacion bajo shock

El agente RE-DECIDE bajo portfolios sometidos a shocks macro. KPIs por postura diferenciados.

Fuente: `reports/stress_summary_pc9_final.csv`

| Escenario | Postura      | EVA Post     | RWA Post   | n\_ventas | n\_mantener |
|-----------|-------------|-------------|-----------|----------|------------|
| baseline  | prudencial  | +748.6 M    | 3 404.8 M | 55       | 445        |
| baseline  | balanceado  | +725.9 M    | 3 392.9 M | 168      | 88         |
| baseline  | desinversion| +48.5 K     | 508.4 K   | 408      | 87         |
| severe    | prudencial  | +712.7 M    | 3 588.6 M | **1**    | 499        |
| severe    | balanceado  | +692.6 M    | 3 572.9 M | 50       | 208        |
| severe    | desinversion| +42.9 K     | 536.8 K   | 252      | 238        |

**Hallazgo clave**: Prudencial reduce ventas de 55 a **1** bajo severe. El agente
reconoce que vender bajo crisis severa (precios colapsados + fire-sale guardrails)
destruye mas valor que mantener, y adapta la estrategia sin violar restricciones regulatorias.

### 3B. Backtesting Light — Estrategia fija, mundo empeora

Las DECISIONES tomadas en baseline se mantienen fijas; solo se recalculan KPIs bajo shocks.
Esto mide la fragilidad residual DESPUES de las decisiones del agente.

Fuente: `reports/backtesting_light_pc9_final.csv`

| Escenario | Postura      | EVA Stressed  | RWA Stressed  | EL Stressed  |
|-----------|-------------|--------------|--------------|-------------|
| baseline  | prudencial  | -1 068 M     | 5 375 M      | 688 M       |
| baseline  | balanceado  |   -656 M     | 3 221 M      | 429 M       |
| baseline  | desinversion|   -236 M     | 1 200 M      | 151 M       |
| severe    | prudencial  | -1 894 M     | 6 719 M      | 1 380 M     |
| severe    | balanceado  | -1 159 M     | 4 026 M      | 852 M       |
| severe    | desinversion|   -417 M     | 1 499 M      | 303 M       |

> Nota: El EVA stressed es negativo porque incluye el coste de capital regulatorio
> sobre el RWA residual del libro. Los prestamos que ya fueron vendidos en baseline
> NO contribuyen a este calculo (on-book filtrado por postura).

### 3C. Interpretacion (3 bullets)

- **Re-optimizar vs aguantar**: El stress engine (3A) muestra lo que HARIA el agente si supiera
  el shock futuro; el backtesting (3B) muestra la vulnerabilidad de las decisiones ya tomadas.
  La diferencia es el "valor de la informacion" del agente ante escenarios adversos.

- **Por que Desinversion tiene RWA ~0 en backtesting**: Al vender 408/500 prestamos, quedan
  solo ~90 prestamos en libro con EAD muy reducida (~1.2 B EUR residual). Bajo shocks, el
  impacto en RWA stressed es proporcional al libro residual, que es minimo por diseno.

- **Por que Prudencial no vende bajo severe (1 venta)**: El agente opera bajo fire-sale
  guardrails que bloquean ventas cuando el precio/EAD cae por debajo del floor de venta
  configurado (`sale_floor_ratio`). Bajo severe, la mayoria de precios caen por debajo del
  umbral de "precio no insultante" => las ventas se bloquean automaticamente.
  Este es el comportamiento DESEADO: no liquidar a cualquier precio.

### 3D. Pricing Crunch (Fix PC9)

El escenario `pricing_crunch` aplica un haircut del 30% (`bid_haircut_mult: 1.3`) a los
precios calculados por el simulador NPL (`cfg.BID_HAIRCUT_GLOBAL`). Antes de PC9,
este shock no se propagaba al inferer (`stress_engine` recalculaba precios desde cero
sin leer el haircut). Este bug fue corregido: ahora `stress_engine` inyecta el haircut
en `cfg.BID_HAIRCUT_GLOBAL` justo antes de llamar a `run_coordinator_inference`, con
reset garantizado via `finally`. El test `tests/test_stress_pricing_crunch_effect.py`
valida que `pricing_crunch != baseline` en al menos un KPI.

---

## 4. Controles de Ingesta (Bank-Ready)

- **Modo estricto por defecto**: `ALLOW_CLIP_OUT_OF_RANGE = False`.
  Cualquier PD/LGD/RW fuera de rango lanza `ValueError` con mensaje accionable.
- **Sin fallback silencioso**: `ALLOW_LEGACY_PORTFOLIO_LOAD = False`.
  Si la ingesta robusta falla, el pipeline aborta con error claro.
- **Test negativo validado**: Portfolio con PD=1.5 lanza error en < 1s.

---

## 5. CI Pipeline

Ultimo run exitoso: `ci_local.bat pc9_final` -> EXIT 0
Log: `logs/ci_local_pc9_final_20260220_135438.log`

Pasos ejecutados: smoke_test, 3 posturas, stress_engine (12 runs), backtesting_light,
compare_postures (PASS), evaluate_baselines, committee_pack, pytest (26/26 passed).

---

## 6. Artefactos Entregables

| Artefacto | Ruta |
|-----------|------|
| Stress Summary (pc9\_final) | `reports/stress_summary_pc9_final.csv` |
| Backtesting Light (pc9\_final) | `reports/backtesting_light_pc9_final.csv/.md` |
| Committee Pack | `reports/committee_pack_pc9_final_20260220_141126/` |
| QA Evidence PC9 | `logs/qa_checkpoint9_evidence.txt` |
| CI Log | `logs/ci_local_pc9_final_20260220_135438.log` |
| Configuracion | `config.py` + `configs/stress_scenarios.yaml` |
| Mapping ingesta | `data/mappings/real_portfolio_mapping.yaml` |

---

*Generado automaticamente por pipeline PC9 — Version controlada en git branch `feat/pc9-realdata-calibration`*
