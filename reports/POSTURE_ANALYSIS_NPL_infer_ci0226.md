# POSTURE ANALYSIS NPL — Tag: `infer_ci0226`

**Generado:** 20260222_190347  |  **Status distance checks:** ✅ PASS

> **Principio NPL:** Todos los préstamos están en default. Este documento NO fija
> precios finales ni términos de contrato. Define un *Negotiation Envelope*
> (rango + walk-away) por postura. Si la negociación cae fuera del envelope:
> la decisión revierte a **MANTENER** (no cristalizar pérdida destructiva).

---

## 1. Comparación de Posturas — Resumen Ejecutivo

| KPI | PRUDENCIAL | BALANCEADO | DESINVERSION |
|-----|-----------|-----------|-------------|
| N loans | 500 | 500 | 500 |
| VENDER | 55 | 168 | 408 |
| REESTRUCTURAR | 71 | 244 | 5 |
| MANTENER | 374 | 88 | 87 |
| Sell rate | 11.0% | 33.6% | 81.6% |
| EVA_post total | 748,631,942 EUR | 725,865,394 EUR | 48,541 EUR |
| RWA_post total | 3,404,802,052 EUR | 3,392,921,675 EUR | 508,421 EUR |
| Capital liberado | 152,359,182 EUR | 411,429,723 EUR | 515,266,481 EUR |
| Sale P&L total | -166,437,430 EUR | -594,358,325 EUR | -1,022,628,036 EUR |
| Sale P&L / EAD | -34.1% | -30.9% | -31.2% |
| Fire-sale bloqueados | 18 | 0 | 0 |
| Guardrail bloqueados | 32 | 21 | 67 |
| Macro steered | 0 | 0 | 0 |

### Evidencia de separación

| Check | Valor | Umbral | Estado |
|-------|-------|--------|--------|
| SellRate_Desinv_minus_Balanc | 0.4800 | 0.1000 | ✅ PASS |
| SellRate_Balanc_minus_Prud | 0.2260 | 0.1000 | ✅ PASS |
| CapRelease_Desinv_minus_Balanc_EUR | 103836758.2617 | 50000000.0000 | ✅ PASS |
| CapRelease_Balanc_minus_Prud_EUR | 259070541.0672 | 100000000.0000 | ✅ PASS |
| RestRate_Balanc_higher_Desinv | 0.4780 | -0.0500 | ✅ PASS |
| EVA_Prud_highest | 22766547.7686 | 0.0000 | ✅ PASS |

---

## 2. Diseño de Negotiation Envelopes por Postura

### 2A. Reservation Floors (VENDER — walk-away price, **POR PRÉSTAMO**)

> **Principio NPL:** El floor NO es un porcentaje global fijo. Se deriva del precio
> propio del préstamo (`Price_to_EAD_i`) aplicando un multiplicador lognormal que
> aproxima el percentil objetivo del simulador de precios del propio instrumento.
> `floor_i = Price_to_EAD_i × mult_postura  (mínimo = floor_absoluto_postura)`

| Postura | Mult. lognormal | Percentil aprox. | Floor mínimo absoluto | Justificación |
|---------|-----------------|------------------|-----------------------|---------------|
| PRUDENCIAL | 0.945 | ≈ p40 precio propio | 8.0% EAD | Floor exigente: sólo vender si mercado paga ≥ p40 del préstamo propio. Fire-sale = nunca. |
| BALANCEADO | 0.873 | ≈ p27 precio propio | 5.5% EAD | Compromiso: precio razonable que libera capital sin destruir demasiado EVA. |
| DESINVERSION | 0.779 | ≈ p13 precio propio | 3.5% EAD | Velocidad > precio: floor bajo pero nunca insulto (<3.5% EAD). |

**Ejemplo cálculo** (loan con `Price_to_EAD = 0.130`):
```
  PRUDENCIAL   floor = max(0.130 × 0.945, 0.080) = 0.123
  BALANCEADO   floor = max(0.130 × 0.873, 0.055) = 0.113
  DESINVERSION floor = max(0.130 × 0.779, 0.035) = 0.101
```

**Regla de ejecución universal:**
```
Si best_offer >= floor_i (individual)  → EJECUTAR VENDER
Si best_offer <  floor_i (individual)  → MANTENER (o REESTRUCTURAR si viable y ΔEVA > 0)
```

### 2B. Bandas de Concesión (REESTRUCTURAR — workout NPL) + Margen de Seguridad

| Postura | Quita max | Plazo extra max | Tasa mín | DSCR_min (walk-away) | DSCR_clean (sin REVIEW) | PTI_post máx | Banda |
|---------|-----------|-----------------|----------|----------------------|--------------------------|-------------|-------|
| PRUDENCIAL | 3.0% | +12m | 10.0% | 1.30 | ≥ 1.50 (safety +20.0%) | 35.0% | ESTRECHA (prudencial: concesiones limitadas) |
| BALANCEADO | 10.0% | +24m | 7.0% | 1.10 | ≥ 1.20 (safety +10.0%) | 40.0% | MEDIA (balanceado: trade-off EVA vs viabilidad) |
| DESINVERSION | 5.0% | +12m | 15.0% | 1.05 | ≥ 1.20 (safety +15.0%) | 42.0% | EXCEPCION (desinversion: sólo si venta bloqueada y uplift alto) |

**Margen de seguridad NPL (`dscr_safety_band`)**:
> Si `DSCR_post ∈ [dscr_min_wa, dscr_clean)` → `REVIEW_REQUIRED` (reestructura frágil).
> Aplica especialmente a DESINVERSION: evitar reestructuras con DSCR pegado al umbral.

**Regla walk-away:**
```
Si deudor exige Quita > quita_max
   O Tasa < tasa_min
   O DSCR_post < dscr_min_wa
   O PTI_post > pti_max_wa
→ NO REESTRUCTURAR → MANTENER (o evaluar VENDER si best_offer >= floor_i)
```

### 2C. PRUDENCIAL Carve-Out (evitar 100% MANTENER)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| EVA_gain top X% | top 20% | Sólo préstamos con ΔEVA en el cuartil superior del portfolio |
| ΔEVA mínimo abs. | 15,000€ | Anti-trivial: excluye carve-outs de bajo valor |
| DSCR_post mínimo | ≥ 1.50 | dscr_min_wa + dscr_buffer (holgado, no pegado al umbral) |
| PTI máximo | ≤ 35.0% | Dentro del límite prudencial |
| Quita máxima | ≤ 3.0% | Banda estrecha prudencial |
| Tasa mínima | ≥ 10.0% | No por debajo del hurdle |

> Préstamos MANTENER en PRUDENCIAL que cumplen TODOS los criterios anteriores
> se elevan a **REESTRUCTURAR** con `Reason_Code_Final = RC03_PRUDENCIAL_CARVEOUT_RESTRUCT`.
> Los demás permanecen en MANTENER.

---

## 3.1 POSTURA: PRUDENCIAL

> **Objetivo:** Estabilidad del libro. Evitar cristalizar pérdidas. MANTENER como opción por defecto ante incertidumbre. No se ejecuta ninguna venta fire-sale ni por debajo del floor individual del préstamo (derivado su precio propio × 0.945). Se han aplicado **26 carve-outs** (top 20% EVA + DSCR≥1.50 + PTI≤35%) elevados a REESTRUCTURAR.

### Distribución Reason_Code_Final

| Reason_Code | Loans |
|-------------|-------|
| RC03_MICRO_RESTRUCT_VALUE_UPLIFT | 352 |
| RC03_PRUDENCIAL_CARVEOUT_CLEAN | 45 |
| RC_GUARDRAIL_BLOCK | 32 |
| RC05_KEEP_ACCEPTABLE_ECONOMICS | 27 |
| RC03_PRUDENCIAL_CARVEOUT_RESTRUCT | 26 |
| RC02_SELL_BLOCKED_FIRE_SALE | 18 |

### Convergencia

| Convergencia_Caso | Loans |
|-----------------|-------|
| MACRO_NOT_APPLIED | 498 |
| AGREE_MICRO_MACRO | 2 |

### Top 10 Loans Frontera (PRUDENCIAL)

| loan_id | segment | EAD | Accion_final | Reason_Code | EVA_pre | EVA_post | RORWA_pre | RORWA_post | pnl | Price_to_EAD | DSCR_post | PTI_post | GoToMarket | Indicative_Bid_Range | Reservation_Floor | Anchor_Terms | Concession_Band | WalkAway_Rule_Text | Fallback_Reason | Trigger_to_Action | Carve_Out_Type | Decision_Governance_Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L000054 | SME | 139465 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -24266.91 | 27436.35752737092 | -0.01600000000000001 | 0.2169947067698508 | -55773.90134395173 | 0.1583944034674588 | 4.358448386709995 |  | BLOQUEADO_FIRE_SALE | [+20,222 EUR, +24,182 EUR] | 15.0% EAD = +20,875 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 15.0% EAD = +20,875 EUR [floor propio] → MANTENE… |  |  | ENVELOPE=Postura=PRUDENCIAL | Floor_loan=15.0% EAD = +20,… |
| L000402 | SME | 771197 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -134188.278 | 152846.7329870281 | -0.016 | 0.2178679743015013 | -308239.8884537666 | 0.1618914344279624 | 2.010309690102036 |  | BLOQUEADO_FIRE_SALE | [+111,824 EUR, +136,418 EUR] | 15.3% EAD = +117,983 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 15.3% EAD = +117,983 EUR [floor propio] → MANTEN… |  |  | ENVELOPE=Postura=PRUDENCIAL | Floor_loan=15.3% EAD = +117… |
| L000058 | Large Corporate | 16224307 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -2887926.646 | 2570822.933121249 | -0.01866666666666666 | 0.1943448666630731 | -6469715.093295344 | 0.1552817584821383 | 1.151754058735836 |  | BLOQUEADO_FIRE_SALE | [+2,352,525 EUR, +2,762,704 EUR] | 14.7% EAD = +2,380,775 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 14.7% EAD = +2,380,775 EUR [floor propio] → MANT… |  |  | ENVELOPE=Postura=PRUDENCIAL | Floor_loan=14.7% EAD = +2,3… |
| L000237 | SME | 643338 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -110010.798 | 127464.3790839548 | -0.014 | 0.2178296590423877 | -256310.4700194735 | 0.1506949405976612 | 3.554625258393132 |  | BLOQUEADO_FIRE_SALE | [+93,284 EUR, +106,598 EUR] | 14.2% EAD = +91,616 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 14.2% EAD = +91,616 EUR [floor propio] → MANTENE… |  |  | ENVELOPE=Postura=PRUDENCIAL | Floor_loan=14.2% EAD = +91,… |
| L000035 | Large Corporate | 20425250 | REESTRUCTURAR | RC03_PRUDENCIAL_CARVEOUT_CLEAN | -3799096.5 | 4054536.319764335 | -0.024 | 0.2180534830561807 | -5682399.564514268 | 0.1379246902793333 | 1.267035993642793 | 0.0 | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=13.0% EAD = +2,662,203 EUR (derivado precio pr… | 13.0% EAD (backup si walk-away; floor propio >= min 8.0% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=3.0% | Plazo_extra_max=+12m | Tasa_min=10.0% | DSC… | WALK_AWAY ACTIVO: DSCR_post=1.27 < minimo=1.30 → Mantener o … | N/A (REESTRUCTURAR) | Si deudor fuera de banda [ESTRECHA (prudencial: concesiones … |  |  | WORKOUT_BAND=Quita_max=3.0% | Plazo_extra_max=+12m | Tasa… |
| L000286 | Large Corporate | 14873116 | REESTRUCTURAR | RC03_PRUDENCIAL_CARVEOUT_CLEAN | -2843627.730913102 | 2952336.004483769 | -0.02746164425858945 | 0.2180507657667397 | -4928915.868804462 | 0.09943134805679382 | 1.279048017087069 | 0.0 | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=9.4% EAD = +1,397,517 EUR (derivado precio pro… | 9.4% EAD (backup si walk-away; floor propio >= min 8.0% EAD) | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=3.0% | Plazo_extra_max=+12m | Tasa_min=10.0% | DSC… | WALK_AWAY ACTIVO: DSCR_post=1.28 < minimo=1.30 → Mantener o … | N/A (REESTRUCTURAR) | Si deudor fuera de banda [ESTRECHA (prudencial: concesiones … |  |  | WORKOUT_BAND=Quita_max=3.0% | Plazo_extra_max=+12m | Tasa… |
| L000455 | Large Corporate | 20496501 | REESTRUCTURAR | RC03_PRUDENCIAL_CARVEOUT_CLEAN | -3395271.162478475 | 4068680.948588933 | -0.01043417158465161 | 0.2180535083599204 | -8967739.22938473 | 0.1440430711729496 | 1.298860755889089 | 0.0 | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=13.6% EAD = +2,789,998 EUR (derivado precio pr… | 13.6% EAD (backup si walk-away; floor propio >= min 8.0% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=3.0% | Plazo_extra_max=+12m | Tasa_min=10.0% | DSC… | WALK_AWAY ACTIVO: DSCR_post=1.30 < minimo=1.30 → Mantener o … | N/A (REESTRUCTURAR) | Si deudor fuera de banda [ESTRECHA (prudencial: concesiones … |  |  | WORKOUT_BAND=Quita_max=3.0% | Plazo_extra_max=+12m | Tasa… |
| L000426 | Corporate | 90956907 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -15735544.911 | 18056362.38866975 | -0.01533333333333333 | 0.2180591275632561 | -23487326.88564626 | 0.1320288958184028 | 1.246288191395768 |  | NO_ACTION | [N/A — floor: 12.5% EAD = +11,348,448 EUR] | 12.5% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | MANTENER por decisión estratégica (RC03_MICRO_RESTRUCT_VALUE… | Monitorizar siguiente ciclo de valoración (90 días) |  |  | FALLBACK=MANTENER por decisión estratégica (RC03_MICRO_RE… |
| L000418 | Large Corporate | 82592045 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -14536199.92 | 9784880.300239842 | -0.01733333333333333 | 0.1706452982056079 | -23738214.19365321 | 0.1491158438141452 | 1.074166910669451 |  | NO_ACTION | [N/A — floor: 14.1% EAD = +11,638,414 EUR] | 14.1% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | MANTENER por decisión estratégica (RC03_MICRO_RESTRUCT_VALUE… | Monitorizar siguiente ciclo de valoración (90 días) |  |  | FALLBACK=MANTENER por decisión estratégica (RC03_MICRO_RE… |
| L000027 | Large Corporate | 56380099 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -9882503.844823735 | 11192232.54619808 | -0.01685569932307421 | 0.2180581250973894 | -13319638.19973983 | 0.07710492715367546 | 1.184640567746975 |  | NO_ACTION | [N/A — floor: 8.0% EAD = +4,510,408 EUR] | 8.0% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | MANTENER por decisión estratégica (RC03_MICRO_RESTRUCT_VALUE… | Monitorizar siguiente ciclo de valoración (90 días) |  |  | FALLBACK=MANTENER por decisión estratégica (RC03_MICRO_RE… |

### Análisis CIB — PRUDENCIAL

**VENTAS:** La postura prudencial no ejecuta ninguna venta en este ciclo.
- 18 loans bloqueados por fire-sale (RC02): mercado ilíquido / spread bid-ask excesivo.
- 32 loans bloqueados por guardrail P&L (RC_GUARDRAIL_BLOCK): pérdida potencial supera threshold EAD×40%.
- Remaining 324 MANTENER: economía holding > liquidación actual.

**FLOOR LATENTE:** 14.5% del EAD.
Si precio de mercado mejora hasta este nivel Y desaparece el fire-sale → re-evaluar VENDER.

**WALK-AWAY REESTRUCTURAR:** Banda estrecha — quita máx. 3.0%, DSCR mín. 1.30. Si el deudor exige condiciones más agresivas → MANTENER.

## 3.2 POSTURA: BALANCEADO

> **Objetivo:** Trade-off EVA vs capital. Activar ventas con precio razonable, priorizar reestructuras con ΔEVA claro, controlar concentración (HHI).

### Distribución Reason_Code_Final

| Reason_Code | Loans |
|-------------|-------|
| RC03_MICRO_RESTRUCT_VALUE_UPLIFT | 434 |
| RC05_KEEP_ACCEPTABLE_ECONOMICS | 45 |
| RC_GUARDRAIL_BLOCK | 21 |

### Convergencia

| Convergencia_Caso | Loans |
|-----------------|-------|
| MACRO_NOT_APPLIED | 496 |
| AGREE_MICRO_MACRO | 4 |

### Top 10 Loans Frontera (BALANCEADO)

| loan_id | segment | EAD | Accion_final | Reason_Code | EVA_pre | EVA_post | RORWA_pre | RORWA_post | pnl | Price_to_EAD | DSCR_post | PTI_post | GoToMarket | Indicative_Bid_Range | Reservation_Floor | Anchor_Terms | Concession_Band | WalkAway_Rule_Text | Fallback_Reason | Trigger_to_Action | Decision_Governance_Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L000058 | Large Corporate | 16224307 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -2887926.646 | 2570822.933121249 | -0.01866666666666666 | 0.1943448666630731 | -6469715.093295344 | 0.1552817584821383 | 1.151754058735836 |  | BLOQUEADO_FIRE_SALE | [+2,154,292 EUR, +2,762,704 EUR] | 13.6% EAD = +2,199,383 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 13.6% EAD = +2,199,383 EUR [floor propio] → MANT… |  | ENVELOPE=Postura=BALANCEADO | Floor_loan=13.6% EAD = +2,1… |
| L000374 | Large Corporate | 12008417 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -2138545.753991765 | 1033589.460286183 | -0.01872482187517665 | 0.1540262309903467 | -4762808.12978068 | 0.1314086855355099 | 1.1551906562587 |  | BLOQUEADO_FIRE_SALE | [+1,320,926 EUR, +1,758,137 EUR] | 11.5% EAD = +1,377,603 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 11.5% EAD = +1,377,603 EUR [floor propio] → MANT… |  | ENVELOPE=Postura=BALANCEADO | Floor_loan=11.5% EAD = +1,3… |
| L000189 | Corporate | 5449220 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -942715.06 | 1081520.000826732 | -0.01533333333333333 | 0.218033477990662 | -2147279.798483266 | 0.1547775433667532 | 1.33311422372491 |  | BLOQUEADO_FIRE_SALE | [+720,809 EUR, +925,155 EUR] | 13.5% EAD = +736,303 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 13.5% EAD = +736,303 EUR [floor propio] → MANTEN… |  | ENVELOPE=Postura=BALANCEADO | Floor_loan=13.5% EAD = +736… |
| L000247 | Corporate | 1772800 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -320876.8 | 351683.2780591774 | -0.02066666666666666 | 0.2179768962850774 | -698530.5826817746 | 0.1655266961480057 | 1.156539086608058 |  | BLOQUEADO_FIRE_SALE | [+253,558 EUR, +320,038 EUR] | 14.5% EAD = +256,178 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 14.5% EAD = +256,178 EUR [floor propio] → MANTEN… |  | ENVELOPE=Postura=BALANCEADO | Floor_loan=14.5% EAD = +256… |
| L000023 | Project Finance | 9769659 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -1641302.712 | -7923.376929357153 | -0.012 | 0.09951283226351174 | -3306170.428123887 | 0.117534236399294 | 1.131997942374777 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=10.3% EAD = +1,002,439 EUR (derivado precio pr… | 10.3% EAD (backup si walk-away; floor propio >= min 5.5% EAD… | Tasa_propuesta=9.0% | Plazo=240m | Quita=0.0% | DSCR_post=1.… | Quita_max=10.0% | Plazo_extra_max=+24m | Tasa_min=7.0% | DSC… | REVIEW_REQUIRED: DSCR_post=1.13 está en la zona de proximida… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [MEDIA (balanceado: trade-off EVA v… |  | WORKOUT_BAND=Quita_max=10.0% | Plazo_extra_max=+24m | Tas… |
| L000278 | Project Finance | 18527431 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -3139679.386384647 | -14801.98815897002 | -0.01297408641937988 | 0.09952009811545048 | -4906532.947679297 | 0.08736309608417098 | 1.144337332863889 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=7.6% EAD = +1,413,050 EUR (derivado precio pro… | 7.6% EAD (backup si walk-away; floor propio >= min 5.5% EAD) | Tasa_propuesta=9.0% | Plazo=240m | Quita=0.0% | DSCR_post=1.… | Quita_max=10.0% | Plazo_extra_max=+24m | Tasa_min=7.0% | DSC… | REVIEW_REQUIRED: DSCR_post=1.14 está en la zona de proximida… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [MEDIA (balanceado: trade-off EVA v… |  | WORKOUT_BAND=Quita_max=10.0% | Plazo_extra_max=+24m | Tas… |
| L000245 | Project Finance | 1884531 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -316601.208 | -1730.165960557912 | -0.01199999999999999 | 0.09944851705639285 | -651468.6454062512 | 0.1170168513201439 | 1.156948370861394 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=10.2% EAD = +192,516 EUR (derivado precio prop… | 10.2% EAD (backup si walk-away; floor propio >= min 5.5% EAD… | Tasa_propuesta=9.0% | Plazo=240m | Quita=0.0% | DSCR_post=1.… | Quita_max=10.0% | Plazo_extra_max=+24m | Tasa_min=7.0% | DSC… | REVIEW_REQUIRED: DSCR_post=1.16 está en la zona de proximida… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [MEDIA (balanceado: trade-off EVA v… |  | WORKOUT_BAND=Quita_max=10.0% | Plazo_extra_max=+24m | Tas… |
| L000027 | Large Corporate | 56380099 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -9882503.844823735 | 11192232.54619808 | -0.01685569932307421 | 0.2180581250973894 | -13319638.19973983 | 0.07710492715367546 | 1.184640567746975 |  | NO_ACTION | [N/A — floor: 6.7% EAD = +3,795,091 EUR] | 6.7% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | MANTENER por decisión estratégica (RC03_MICRO_RESTRUCT_VALUE… | Monitorizar siguiente ciclo de valoración (90 días) |  | FALLBACK=MANTENER por decisión estratégica (RC03_MICRO_RE… |
| L000130 | Large Corporate | 50307129 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -9340639.903314378 | 9986636.743122218 | -0.02378152744268615 | 0.2180578067571241 | -17135771.27777248 | 0.08046195522172979 | 1.306667656062364 |  | NO_ACTION | [N/A — floor: 7.0% EAD = +3,533,738 EUR] | 7.0% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | MANTENER por decisión estratégica (RC03_MICRO_RESTRUCT_VALUE… | Monitorizar siguiente ciclo de valoración (90 días) |  | FALLBACK=MANTENER por decisión estratégica (RC03_MICRO_RE… |
| L000281 | Large Corporate | 40226436 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -7511844.886333426 | 7985434.502314732 | -0.02449267419288526 | 0.2180570661390022 | -13479273.53156693 | 0.08138614609570433 | 1.179719568440087 |  | NO_ACTION | [N/A — floor: 7.1% EAD = +2,858,093 EUR] | 7.1% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | MANTENER por decisión estratégica (RC03_MICRO_RESTRUCT_VALUE… | Monitorizar siguiente ciclo de valoración (90 días) |  | FALLBACK=MANTENER por decisión estratégica (RC03_MICRO_RE… |

### Análisis CIB — BALANCEADO

**VENTAS (168 loans, 33.6%):**
- 168 loans con P&L medio = -30.9% del EAD.
- Floor activo: 11.0% EAD. Sólo ejecutar si la oferta supera este nivel.
- Fire-sale activo en la mayoría de loans VENDER: recomendación de GO-TO-MARKET sólo para los que no tienen este flag.

**REESTRUCTURAS (244 loans):**
- ΔEVA mediano = +3,711,875 EUR.
- DSCR_post p50 ≈ 2.07 (buena viabilidad post-workout).
- Banda media: quita máx 10.0%, tasa mín 7.0%.
- HHI segmento Large Corporate = 0.494 > umbral 0.30 → R2 macro aplicado (1 loan rotado a VENDER).

**DIFERENCIACIÓN vs PRUDENCIAL:**
- Sell rate: 33.6% vs 0.0% (prudencial). Gap = 33.6%.
- Capital liberado: +411,429,723 EUR vs +152,359,182 EUR (prud).

## 3.3 POSTURA: DESINVERSION

> **Objetivo:** Liberar capital y reducir RWA. Venta como acción principal con floor más bajo que balanceado. Reestructura sólo como excepción operada.

### Distribución Reason_Code_Final

| Reason_Code | Loans |
|-------------|-------|
| RC06_MICRO_SELL_VALUE_NEGATIVE | 403 |
| RC_GUARDRAIL_BLOCK | 67 |
| RC15_VOLUME_CAP_VOLUNTARY_SELL | 14 |
| RC03_MICRO_RESTRUCT_VALUE_UPLIFT | 8 |
| RC05_KEEP_ACCEPTABLE_ECONOMICS | 8 |

### Convergencia

| Convergencia_Caso | Loans |
|-----------------|-------|
| MACRO_NOT_APPLIED | 496 |
| GUARDRAIL_OVERRIDE | 4 |

### Top 10 Loans Frontera (DESINVERSION)

| loan_id | segment | EAD | Accion_final | Reason_Code | EVA_pre | EVA_post | RORWA_pre | RORWA_post | pnl | Price_to_EAD | DSCR_post | PTI_post | GoToMarket | Indicative_Bid_Range | Reservation_Floor | Anchor_Terms | Concession_Band | WalkAway_Rule_Text | Fallback_Reason | Trigger_to_Action | Decision_Governance_Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L000054 | SME | 139465 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -24266.91 | 0.0 | -0.01600000000000001 |  | -55773.90134395173 | 0.1583944034674588 | 1.022360170365771 |  | SI | [+18,953 EUR, +24,182 EUR] | 12.3% EAD = +17,208 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 12.3% EAD = +17,208 EUR [floor propio] → MANTENE… |  | ENVELOPE=Postura=DESINVERSION | Floor_loan=12.3% EAD = +1… |
| L000402 | SME | 771197 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -134188.278 | 0.0 | -0.016 |  | -308239.8884537666 | 0.1618914344279624 | 1.05287559650931 |  | SI | [+107,498 EUR, +136,418 EUR] | 12.6% EAD = +97,258 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 12.6% EAD = +97,258 EUR [floor propio] → MANTENE… |  | ENVELOPE=Postura=DESINVERSION | Floor_loan=12.6% EAD = +9… |
| L000058 | Large Corporate | 16224307 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -2887926.646 | 0.0 | -0.01866666666666666 |  | -6469715.093295344 | 0.1552817584821383 | 1.019371817931531 |  | SI | [+2,154,292 EUR, +2,762,704 EUR] | 12.1% EAD = +1,962,565 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 12.1% EAD = +1,962,565 EUR [floor propio] → MANT… |  | ENVELOPE=Postura=DESINVERSION | Floor_loan=12.1% EAD = +1… |
| L000103 | Mortgage | 448055 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -60911.99180183938 | 0.0 | -0.03594757742205617 |  | -178645.7904461331 | 0.1773845735550093 |  | 0.301174632045914 | SI | [+69,397 EUR, +86,199 EUR] | 13.8% EAD = +61,913 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 13.8% EAD = +61,913 EUR [floor propio] → MANTENE… |  | ENVELOPE=Postura=DESINVERSION | Floor_loan=13.8% EAD = +6… |
| L000342 | SME | 126466 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -19349.298 | 24855.81788302794 | -0.001999999999999995 | 0.2168851306470013 | -48071.97919402554 | 0.1409134128648626 | 2.036401365347867 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=11.0% EAD = +13,882 EUR (derivado precio propi… | 11.0% EAD (backup si walk-away; floor propio >= min 3.5% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=2… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | Dentro de banda DESINVERSION. Continuar negociación. Walk-aw… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… |  | WORKOUT_BAND=Quita_max=5.0% | Plazo_extra_max=+12m | Tasa… |
| L000324 | SME | 54812 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -9669.47270877269 | 10631.18616688961 | -0.01760773442278684 | 0.2153482646288899 | -21693.16971541694 | 0.1119398604722559 | 2.779918769816007 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=8.7% EAD = +4,780 EUR (derivado precio propio)… | 8.7% EAD (backup si walk-away; floor propio >= min 3.5% EAD) | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=2… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | Dentro de banda DESINVERSION. Continuar negociación. Walk-aw… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… |  | WORKOUT_BAND=Quita_max=5.0% | Plazo_extra_max=+12m | Tasa… |
| L000294 | SME | 34641 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | -6027.534000000001 | 6626.873130216587 | -0.016 | 0.2137688128955768 | -14367.82500184678 | 0.1571057316159347 | 3.080758727608118 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=12.2% EAD = +4,240 EUR (derivado precio propio… | 12.2% EAD (backup si walk-away; floor propio >= min 3.5% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=3… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | Dentro de banda DESINVERSION. Continuar negociación. Walk-aw… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… |  | WORKOUT_BAND=Quita_max=5.0% | Plazo_extra_max=+12m | Tasa… |
| L000426 | Corporate | 90956907 | MANTENER | RC05_KEEP_ACCEPTABLE_ECONOMICS | -15735544.911 | 0.0 | -0.01533333333333333 |  | -23487326.88564626 | 0.1320288958184028 | 1.070294087412586 |  | NO_ACTION | [N/A — floor: 10.3% EAD = +9,354,964 EUR] | 10.3% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | ECONOMICS_KEEP — loan genera valor holding > valor liquidaci… | Si RORWA_pre cae por debajo de hurdle (10.0%) O si aparece o… |  | FALLBACK=ECONOMICS_KEEP — loan genera valor holding > val… |
| L000468 | Large Corporate | 82382036 | MANTENER | RC05_KEEP_ACCEPTABLE_ECONOMICS | -14664002.408 | 0.0 | -0.01866666666666666 |  | -31438124.57034084 | 0.1585082219769753 | 1.051510000722116 |  | NO_ACTION | [N/A — floor: 12.3% EAD = +10,172,361 EUR] | 12.3% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | ECONOMICS_KEEP — loan genera valor holding > valor liquidaci… | Si RORWA_pre cae por debajo de hurdle (10.0%) O si aparece o… |  | FALLBACK=ECONOMICS_KEEP — loan genera valor holding > val… |
| L000074 | Large Corporate | 75162126 | MANTENER | RC05_KEEP_ACCEPTABLE_ECONOMICS | -12544228.55888524 | 0.0 | -0.01126373726119487 |  | -23798495.23879223 | 0.1215944057778369 | 1.009800191409938 |  | NO_ACTION | [N/A — floor: 9.5% EAD = +7,119,510 EUR] | 9.5% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | ECONOMICS_KEEP — loan genera valor holding > valor liquidaci… | Si RORWA_pre cae por debajo de hurdle (10.0%) O si aparece o… |  | FALLBACK=ECONOMICS_KEEP — loan genera valor holding > val… |

### Análisis CIB — DESINVERSION

**VENTAS (408 loans, 81.6%):**
- Floor bajo: 9.0% EAD. Acepta liquidación agresiva pero no precio insulto.
- P&L mediano estimado: -31.2% del EAD sold.
- Capital liberado total: +515,266,481 EUR — mayor de las 3 posturas.
- RWA liberado: ≈ +3,404,293,631 EUR.

**REESTRUCTURAS (5 loans — EXCEPCIÓN):**
- Activadas por R3 macro steering (capital_liberado=0 y ΔEVA>0).
- DSCR_post más bajo (p50 ≈ 1.08): mayor riesgo de recaída.
- Banda de excepción: quita máx 5.0%, tasa mín 15.0% (prima de riesgo elevada).

**DIFERENCIACIÓN vs BALANCEADO:**
- Sell rate: 81.6% vs 33.6%.
- Capital liberado gap: +103,836,758 EUR.
- EVA sacrificado: +725,816,852 EUR.

---

## 4. Governance y Trazabilidad

### Campos bank-ready añadidos (esta sesión)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `Envelope_Type` | SELL / WORKOUT / NO_ACTION | Tipo de envelope aplicado |
| `GoToMarket` | SI / BLOQUEADO_x | Si el loan está listo para mercado |
| `Indicative_Bid_Range` | rango [lo, hi] | Rango orientativo de oferta (no precio fijo) |
| `Reservation_Floor` | % EAD + EUR | Walk-away de venta por postura |
| `Execution_Rule_Text` | texto | Regla de ejecución completa con umbrales |
| `Anchor_Terms` | texto | Punto de partida de la reestructura |
| `Concession_Band` | texto | Límites máximos de concesión |
| `WalkAway_Rule_Text` | texto | Cuándo salir de la negociación |
| `Fallback_Reason` | texto | Por qué el loan está en MANTENER |
| `Trigger_to_Action` | texto | Qué cambio activaría una acción futura |
| `Reason_Code_Final` | RC code | 100% coherente con Accion_final (regla dura) |
| `Decision_Governance_Final` | texto enriquecido | Cada capa decisora + envelope |

### Coherencia Reason_Code → Accion_final

Validada al 100% por el módulo `reports/hardening.py` (22/22 PASS en última ejecución).

### Consistencia con committee pack

- Monotonía ventas: DESINVERSION (408) >= BALANCEADO (168) >= PRUDENCIAL (55) ✅
- Monotonía capital liberado: DESINV (+515,266,481 EUR) >= BALANC (+411,429,723 EUR) >= PRUD (+152,359,182 EUR) ✅
- EVA preservado mayor en PRUDENCIAL: +748,631,942 EUR ✅

---

## 5. Restricciones y Disclaimers

- Este documento es de uso **interno del banco**, generado automáticamente por el NPL Optimizer.
- **NO fija precio final** de venta ni términos contractuales de reestructuración.
- Los rangos indicativos se basan en el pricing model interno; no constituyen valoración de mercado.
- Toda nego por fuera del envelope debe ser aprobada por el Comité de Crédito NPL.
- Los datos de DSCR/PTI se calculan con supuestos del modelo; verificar con datos reales del deudor.
- Arquivos generados: `decisiones_finales_{postura}_npl.xlsx` + `enforcement_log_{tag}.csv`.
