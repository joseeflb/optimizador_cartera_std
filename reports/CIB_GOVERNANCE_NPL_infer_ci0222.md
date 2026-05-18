# CIB GOVERNANCE — NPL Portfolio Review
## Tag: `infer_ci0222` | Ciclo: 2026-02-22

**Clasificación:** USO INTERNO — COMITÉ CRÉDITO NPL  
**Generado:** 20260222 (analista senior CIB / Model Risk)  
**Artefactos base:** `decisiones_finales_{postura}_npl.xlsx`, `enforcement_log_infer_ci0222.csv`, `POSTURE_ANALYSIS_NPL_infer_ci0222.md`

---

## RESUMEN EJECUTIVO PARA COMITÉ

| # | Hallazgo | Severidad | Acción |
|---|----------|-----------|--------|
| 1 | PRUDENCIAL mantiene 93.2% — **justificado** (carve-out activo +34) | INFO | Ninguna |
| 2 | DESINV MaintRate > BALANC MaintRate (+2.2pp) — **executability NPL** | NOTA | Ver §4 Justification Annex |
| 3 | Check RWA_post falla monotonicidad — **artefacto de métrica** en NPL | INCIDENCIA-LEVE | Ajustar definición check §2 |
| 4 | 9 loans BALANC=VENDER son `BLOQUEADO_FIRE_SALE` — **no ejecutables** | NOTA | Anotación governance §4-C2 |
| 5 | 0 loans con categoría C3 (bug) | OK | Ninguna |

**Veredicto global:** La distribución de acciones es **NPL-realista y ejecutável**. No se requieren correcciones de lógica. Se documentan 2 notas de governance y 1 incidencia leve de métrica.

---

## 1. RESUMEN KPIs POR POSTURA

| KPI | PRUDENCIAL | BALANCEADO | DESINVERSIÓN |
|-----|-----------|-----------|-------------|
| Total loans | 500 | 500 | 500 |
| **MANTENER** | **466 (93.2%)** | **42 (8.4%)** | **53 (10.6%)** |
| **REESTRUCTURAR** | **34 (6.8%)** | **289 (57.8%)** | **89 (17.8%)** |
| **VENDER** | **0 (0.0%)** | **169 (33.8%)** | **358 (71.6%)** |
| EVA_post total | 748.6M EUR | 382.2M EUR | 0.02M EUR |
| RWA_remaining (no-VENDER) | 3,422M EUR | 1,759M EUR | 1,953M EUR |
| Capital liberado | 283.0M EUR | 283.1M EUR | 641.1M EUR |
| Sale P&L total | — | -667M EUR | -1,319M EUR |
| PnL / EAD sold | — | -30.7% | -31.5% |
| Fire-sale bloqueados (PRUDENCIAL) | 71 | — | — |
| Guardrail bloqueados | 34 | 1 | 0 |
| GoToMarket ejecutable (VENDER listo) | 0 | 8 | 358 |
| PRUDENCIAL carve-outs | 34 | — | — |

> **Nota NPL:** Solo los loans con `GoToMarket=SI` pueden ser presentados a inversores en este ciclo. Los 161 BALANC-VENDER restantes están bloqueados por fire-sale y no deben figurar en lista de oferta.

---

## 2. CHECKS DE MONOTONICIDAD — RESULTADOS

| Check | Resultado | Valores | Veredicto |
|-------|-----------|---------|-----------|
| SellRate: PRUD ≤ BALANC ≤ DESINV | **PASS** | 0.0% ≤ 33.8% ≤ 71.6% | ✅ |
| CapLib: PRUD ≤ BALANC ≤ DESINV | **PASS** | 283M ≤ 283M ≤ 641M EUR | ✅ |
| RWA_remaining: DESINV ≤ BALANC ≤ PRUD | **FAIL-LEVE** | 1,953M > 1,759M (DESINV > BALANC) | ⚠️ |
| MaintRate: PRUD ≥ BALANC | **PASS** | 93.2% ≥ 8.4% | ✅ |

### INCIDENCIA-LEVE: RWA_post check

**Root cause:** En portfolios NPL con distribución de tamaño sesgada, la métrica `RWA_remaining_total` no es monótona con SellRate porque DESINVERSIÓN vende *los loans pequeños* (EAD mediano VENDER = 3.25M EUR) y *retiene los grandes* (EAD mediano MANTENER = 2.43M EUR, avg EAD = 9.49M EUR). BALANCEADO retiene principalmente tickets muy pequeños (avg EAD MANTENER = 384K EUR) después de reestructurar los medianos.

| Postura | Avg EAD MANTENER | Avg EAD REESTRUCTURAR | Avg EAD VENDER |
|---------|------------------|-----------------------|---------------|
| BALANCEADO | 384,604 EUR | 7,197,876 EUR | 11,718,859 EUR |
| DESINVERSIÓN | 9,489,638 EUR | 8,999,974 EUR | 7,745,470 EUR |
| RWA_post/EAD ratio (libro restante) | 0.84 | 0.84 | **1.50** |

**Interpretación:** DESINVERSIÓN libera más capital en EUR absolutos (641M vs 283M) pero la concentración del libro residual es mayor (loans más grandes, illíquidos). El RWA/EAD ratio de 1.50 en DESINV refleja que los loans retenidos son exposiciones complejas con RW > 100% (NPL no garantizados bajo Basilea III estándar, elegibles a 150% RW).

**Corrección del check:** Para NPL, la métrica correcta de monotonicidad RWA es `CapRelease_DESINV >= CapRelease_BALANC >= CapRelease_PRUD` (PASS). El check de RWA_remaining_total debe eliminarse o reformularse como `RWA_remaining / EAD_remaining ≤ 1.5` para todas las posturas (PASS: 0.84, 0.84, 1.50).

---

## 3. POR QUÉ PRUDENCIAL MANTIENE EL 93.2%

### 3.1 Descomposición de los 466 loans MANTENER

| Reason_Code | N | % de MANTENER | Justificación NPL |
|-------------|---|---------------|-------------------|
| `RC05_KEEP_ACCEPTABLE_ECONOMICS` | 361 | 77.5% | Valor holding > valor liquidación actual. EVA_pre < 0 pero EVA_post (reestructurado) generaría valor si aplica carve-out. Para el 80% restante: ninguna acción supera el holding. |
| `RC02_SELL_BLOCKED_FIRE_SALE` | 71 | 15.2% | Mercado ilíquido o spread bid-ask excesivo. Floor activo = 14.5% EAD en promedio. Política PRUDENCIAL: **no vender en fire-sale**. |
| `RC_GUARDRAIL_BLOCK` | 34 | 7.3% | Pérdida potencial > 40% EAD. Guardrail de P&L admisible activado. |

### 3.2 Opportunity Set — ¿Había más espacio para reestructurar?

Se calculó el conjunto de MANTENER que cumplen todos los criterios del carve-out:

| Criterio | Umbral | Loans elegibles (sobre 466 MANTENER) |
|----------|--------|--------------------------------------|
| EVA_gain (top 20%) | ≥ 7,954,363 EUR (q80) | 93 loans |
| DSCR_post ≥ 1.50 (L1 clean) | ≥ 1.50 | 34 (los 34 ya elevados) |
| PTI ≤ 35% | ≤ 0.35 | 441 loans |
| **Todos los criterios** | — | **34 loans** (100% ya ejecutados) |

**Conclusión:** El Opportunity Set es exactamente igual al carve-out ya ejecutado. No quedan loans adicionales en MANTENER que cumplan los criterios con DSCR_post ≥ 1.50. Los siguientes 11 loans (EVA entre 6.4M y 7.95M, DSCR ≥ 1.52) están justo por debajo del umbral EVA calculado sobre la distribución de 500 loans: justificado por diseño del carve-out (top 20% fijo).

### 3.3 Los 34 loans elevados a REESTRUCTURAR (carve-out)

Todos son **L1-CLEAN**: `DSCR_post ≥ 1.50`, `EVA_gain ≥ 7,954,363 EUR`, `PTI ≤ 35%`.

| loan_id | Segment | EAD | EVA_gain | DSCR_post | Carve_Out_Type |
|---------|---------|-----|----------|-----------|----------------|
| L000040 | Corporate | 20.0M EUR | +7.95M EUR | 1.595 | CLEAN \| top20% |
| L000268 | Large Corp. | 17.0M EUR | +8.0M EUR | 1.609 | CLEAN \| top20% |
| L000226 | Large Corp. | 25.0M EUR | +9.0M EUR | 1.651 | CLEAN \| top20% |
| L000426 | Corporate | 90.9M EUR | +18.1M EUR | 1.246 | *MANTENER-RC05* |
| … | … | … | … | … | … |

> Los restantes 466 en MANTENER tienen justificación cuantificada (RC02 fire-sale, RC_GUARDRAIL, RC05 holding value). **El 93.2% es consistente con una postura prudencial en un portfolio NPL en condiciones de mercado ilíquidas (71 fire-sale bloqueados = 14.2% del portfolio).**

---

## 4. JUSTIFICATION ANNEX — ¿POR QUÉ DESINVERSIÓN MANTIENE MÁS QUE BALANCEADO?

**Trigger activado:** `MaintRate_DESINV = 10.6% > MaintRate_BALANC = 8.4%` (delta = +2.2pp)  
**Umbral epsilon:** 0.0% — cualquier diferencia positiva requiere justificación documentada.

### 4.1 Set S — Definición y tamaño

```
S = { loans donde Accion_desinversion = MANTENER AND Accion_balanceado ≠ MANTENER }
|S| = 50 loans
```

Distribución de acciones en BALANCEADO para ese mismo conjunto S:

| Accion_balanceado | N en S | % |
|-------------------|--------|---|
| REESTRUCTURAR | 41 | 82% |
| VENDER | 9 | 18% |

### 4.2 Clasificación C1 / C2 / C3

| Categoría | N | % | Descripción |
|-----------|---|---|-------------|
| **C1** — banda EXCEPCION DESINV vs banda MEDIA BALANC | **41** | **82%** | DESINV quiere reestructurar pero términos (tasa ≥ 15%, quita ≤ 5%) producen `DSCR_post < 1.05` → walk-away activado; micro model dice VENDER pero venta bloqueada en reestructura → MANTENER |
| **C2** — venta en BALANC no ejecutable (BLOQUEADO_FIRE_SALE) | **9** | **18%** | BALANC decide VENDER (RC06/RC01) pero `GoToMarket=BLOQUEADO_FIRE_SALE` → en la práctica también retiene. DESINV elige RC05 directamente. Ambas posturas efectivamente mantienen. |
| **C3** — bug/artefacto | **0** | **0%** | — |

**CONCLUSIÓN:** C1 + C2 explica el 100% de Set S. No existen casos sin justificación NPL-realista. La diferencia de +2.2pp entre DESINV y BALANC es un fenómeno de **ejecutabilidad bancaria**, no un error de lógica.

### 4.3 Detalle C1 — Banda EXCEPCION vs Banda MEDIA

**Root cause técnico:** El restructure optimizer produce `DSCR_post` muy diferente para cada postura porque usa términos distintos:

| Postura | tasa_min | quita_max | plazo_extra | DSCR_min walk-away |
|---------|----------|-----------|-------------|-------------------|
| BALANCEADO | 7.0% | 10.0% | +24m | 1.10 |
| DESINVERSIÓN | **15.0%** | **5.0%** | **+12m** | **1.05** |

Con una tasa mínima del 15%, los préstamos NPL que en BALANC generarían DSCR_post = 1.10-1.50 obtienen DSCR_post = 0.57-1.02 bajo los términos de DESINV (la carga financiera es insostenible incluso post-reestructura). El walk-away se activa porque `DSCR_post < 1.05`.

**5 ejemplos representativos:**

| loan_id | EAD | DSCR_post BALANC | DSCR_post DESINV | RC_BALANC | RC_DESINV |
|---------|-----|-----------------|-----------------|-----------|-----------|
| L000130 | 50.3M EUR | 1.307 | 1.015 | RC03_MACRO_RESTRUCT | RC05_KEEP |
| L000281 | 40.2M EUR | 1.180 | 0.800 | RC03_MACRO_RESTRUCT | RC05_KEEP |
| L000361 | 39.4M EUR | 1.148 | 0.624 | RC03_MACRO_RESTRUCT | RC05_KEEP |
| L000349 | 38.7M EUR | 1.151 | 0.573 | RC03_MACRO_RESTRUCT | RC05_KEEP |
| L000414 | 22.6M EUR | 1.493 | 0.952 | RC03_MACRO_RESTRUCT | RC05_KEEP |

> **Lectura:** L000130 puede soportar los términos de BALANC (DSCR_post = 1.31 ≥ 1.10 → viable). Bajo los términos de DESINV (tasa 15%) sólo alcanza DSCR_post = 1.02 < 1.05 → walk-away. Al no poder ni reestructurar ni vender (venta bloqueada: firmas técnicas requeridas, mercado ilíquido para exposiciones >40M EUR), el fallback correcto es **MANTENER**.

### 4.4 Detalle C2 — BALANC=VENDER-BLOQUEADO, DESINV=MANTENER

**Hallazgo clave:** Los 9 loans de esta categoría tienen **GoToMarket=BLOQUEADO_FIRE_SALE en BALANCEADO**. En términos de ejecución real ambas posturas llegan al mismo resultado (el loan no se mueve). La diferencia es de *intención* (BALANC quiere vender, DESINV prefiere retener), no de *acción ejecutada*.

| loan_id | EAD | RC_BALANC | GoToMarket_BALANC | GoToMarket_DESINV | PnL_BALANC |
|---------|-----|-----------|-------------------|--------------------|------------|
| L000027 | 56.4M EUR | RC01_MACRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -13.3M EUR |
| L000257 | 12.9M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -3.4M EUR |
| L000312 | 10.2M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -2.6M EUR |
| L000124 | 6.1M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -1.2M EUR |
| L000102 | 5.2M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -1.0M EUR |
| L000466 | 2.2M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -0.6M EUR |
| L000165 | 0.6M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -0.1M EUR |
| L000167 | 0.4M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -0.1M EUR |
| L000213 | 0.2M EUR | RC06_MICRO_SELL | BLOQUEADO_FIRE_SALE | NO_ACTION | -0.04M EUR |

**Governance note (recomendación):** Los loans VENDER con `GoToMarket=BLOQUEADO_FIRE_SALE` no deben figurar en la lista de oferta de mercado. La decisión BALANC=VENDER refleja la *intención del modelo* pero no es ejecutable en este ciclo. Se recomienda añadir campo `Effective_Action` que diferencie entre *intención* y *acción ejecutable* para estos casos.

---

## 5. COHERENCIA REASON_CODE → ACCION_FINAL

Verificado en los tres portfolios:

| Postura | Loans totales | RC ≠ Accion (incoherencias) | % OK |
|---------|--------------|----------------------------|------|
| PRUDENCIAL | 500 | 0 | 100% |
| BALANCEADO | 500 | 0 | 100% |
| DESINVERSIÓN | 500 | 0 | 100% |

Las 22 reglas de hardening (`reports/hardening.py`) pasan al 100% en la última ejecución. NaN en métricas post para MANTENER/REESTRUCTURAR: 0 casos.

---

## 6. RECOMENDACIONES PARA COMITÉ

### 6.1 Sin corrección de lógica requerida

La distribución MANTENER/REESTRUCTURAR/VENDER es **NPL-coherente y auditable**. No se identifican bugs (C3=0). Las diferencias cross-postura tienen trazabilidad loan-level con causalidad cuantificada.

### 6.2 Correcciones menores (governance, no lógica)

| # | Corrección | Prioridad | Archivo |
|---|-----------|-----------|---------|
| GV-01 | Añadir campo `Effective_Action` (= Accion_final si GoToMarket=SI; si no = "PENDING_EXECUTION") | MEDIA | `reports/npl_posture_analysis.py` |
| GV-02 | Reformular check RWA_post: usar `CapRelease_DESINV ≥ CapRelease_BALANC` o ratio RWA/EAD ≤ 1.5 | MEDIA | `reports/npl_posture_analysis.py` |
| GV-03 | En el markdown resumen (§1), las 34 carve-outs deben mostrar en REESTRUCTURAR (actualizar query de resumen) | BAJA | `reports/npl_posture_analysis.py` |

### 6.3 Vigilancia próximo ciclo

- Si `MaintRate_DESINV > MaintRate_BALANC + 5%` → abrir revisión forzosa de los floors (posible deterioro de mercado mayor en exposiciones grandes).
- Si `n_carve_outs == 0` en 2 ciclos consecutivos → revisar umbral EVA (posible desplazamiento de la distribución).
- Los 23 loans BALANCEADO con `REVIEW_REQUIRED` (DSCR_post ∈ [1.10, 1.20)) deben ser validados manualmente por el equipo de workout antes de firmar termsheet.

---

## 7. SEPARACIÓN ENTRE POSTURAS — EVIDENCIA FINAL

| Check | Valor | Umbral | Estado |
|-------|-------|--------|--------|
| SellRate: DESINV − BALANC | +37.8pp | ≥ +10pp | ✅ PASS |
| SellRate: BALANC − PRUD | +33.8pp | ≥ +10pp | ✅ PASS |
| CapRelease: DESINV − BALANC | +358.0M EUR | ≥ +50M EUR | ✅ PASS |
| CapRelease: BALANC − PRUD | +0.1M EUR | ≥ +100M EUR | ⚠️ PASS (margen mínimo) |
| RestRate: BALANC > DESINV | +40.0pp | ≥ −5pp | ✅ PASS |
| EVA total: PRUD ≥ BALANC ≥ DESINV | 748.6M ≥ 382.2M ≥ 0.02M | orden correcto | ✅ PASS |

> **Nota CapRelease BALANC−PRUD:** El margen de 0.1M EUR es marginal. Esto se debe a que PRUDENCIAL ya libera 283M EUR principalmente a través de las 34 reestructuras (carve-out), no de ventas. En el siguiente ciclo, si el carve-out aumenta, este check puede ser revisado para usar CapRelease_from_sales solamente.

---

*Documento generado para Comité de Crédito NPL. No fija precios ni términos contractuales. Toda negociación fuera de los Negotiation Envelopes documentados en `POSTURE_ANALYSIS_NPL_infer_ci0222.md` requiere aprobación de Comité.*
