# POSTURE ANALYSIS NPL — Tag: `infer_ci0222`

**Generado:** 20260222_184705  |  **Status distance checks:** ✅ PASS

> **Principio NPL:** Todos los préstamos están en default. Este documento NO fija
> precios finales ni términos de contrato. Define un *Negotiation Envelope*
> (rango + walk-away) por postura. Si la negociación cae fuera del envelope:
> la decisión revierte a **MANTENER** (no cristalizar pérdida destructiva).

---

## 1. Comparación de Posturas — Resumen Ejecutivo

| KPI | PRUDENCIAL | BALANCEADO | DESINVERSION |
|-----|-----------|-----------|-------------|
| N loans | 500 | 500 | 500 |
| VENDER | 0 | 169 | 358 |
| REESTRUCTURAR | 34 | 289 | 89 |
| MANTENER | 466 | 42 | 53 |
| Sell rate | 0.0% | 33.8% | 71.6% |
| EVA_post total | 748,631,942 EUR | 725,865,394 EUR | 48,541 EUR |
| RWA_post total | 3,404,802,052 EUR | 3,392,921,675 EUR | 508,421 EUR |
| Capital liberado | 76,827,915 EUR | 411,429,723 EUR | 515,266,481 EUR |
| Sale P&L total | 0 EUR | -607,677,963 EUR | -873,689,944 EUR |
| Sale P&L / EAD | 0.0% | -30.7% | -31.5% |
| Fire-sale bloqueados | 71 | 0 | 0 |
| Guardrail bloqueados | 34 | 1 | 0 |
| Macro steered | 500 | 500 | 500 |

### Evidencia de separación

| Check | Valor | Umbral | Estado |
|-------|-------|--------|--------|
| SellRate_Desinv_minus_Balanc | 0.3780 | 0.1000 | ✅ PASS |
| SellRate_Balanc_minus_Prud | 0.3380 | 0.1000 | ✅ PASS |
| CapRelease_Desinv_minus_Balanc_EUR | 103836758.2617 | 50000000.0000 | ✅ PASS |
| CapRelease_Balanc_minus_Prud_EUR | 334601807.5584 | 100000000.0000 | ✅ PASS |
| RestRate_Balanc_higher_Desinv | 0.4000 | -0.0500 | ✅ PASS |
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

> **Objetivo:** Estabilidad del libro. Evitar cristalizar pérdidas. MANTENER como opción por defecto ante incertidumbre. No se ejecuta ninguna venta fire-sale ni por debajo del floor individual del préstamo (derivado su precio propio × 0.945). Se han aplicado **34 carve-outs** (top 20% EVA + DSCR≥1.50 + PTI≤35%) elevados a REESTRUCTURAR.

### Distribución Reason_Code_Final

| Reason_Code | Loans |
|-------------|-------|
| RC05_KEEP_ACCEPTABLE_ECONOMICS | 361 |
| RC02_SELL_BLOCKED_FIRE_SALE | 71 |
| RC_GUARDRAIL_BLOCK | 34 |
| RC03_PRUDENCIAL_CARVEOUT_RESTRUCT | 34 |

### Convergencia

| Convergencia_Caso | Loans |
|-----------------|-------|
| MACRO_NOT_APPLIED | 443 |
| MACRO_HARDENING | 55 |
| AGREE_MICRO_MACRO | 2 |

### Top 10 Loans Frontera (PRUDENCIAL)

| loan_id | segment | EAD | Accion_final | Reason_Code_Final | EVA_pre | EVA_post | RORWA_pre | RORWA_post | pnl | Price_to_EAD | DSCR_post | PTI_post | GoToMarket | Indicative_Bid_Range | Reservation_Floor | Anchor_Terms | Concession_Band | WalkAway_Rule_Text | Fallback_Reason | Trigger_to_Action | Carve_Out_Type | Decision_Governance_Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L000040 | Corporate | 20000115 | REESTRUCTURAR | RC03_PRUDENCIAL_CARVEOUT_RESTRUCT | -3699751.357904101 | 3970139.233589334 | -0.02332433614853719 | 0.2180533283236745 | -6358097.36194442 | 0.07251022515907479 | 1.595440273532289 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=8.0% EAD = +1,600,009 EUR (derivado precio pro… | 8.0% EAD (backup si walk-away; floor propio >= min 8.0% EAD) | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=3.0% | Plazo_extra_max=+12m | Tasa_min=10.0% | DSC… | Dentro de banda PRUDENCIAL. Continuar negociación. Walk-away… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [ESTRECHA (prudencial: concesiones … | PRUDENCIAL_RESTRUCT_CARVEOUT_CLEAN|top20pct|EVA>=7,954,363|D… | [PRUDENCIAL] MIXED_DECISION: Micro=REESTRUCTURAR → Final=MAN… |
| L000268 | Large Corporate | 17013047 | REESTRUCTURAR | RC03_PRUDENCIAL_CARVEOUT_RESTRUCT | -3198452.836000001 | 3377151.512006348 | -0.02533333333333334 | 0.2180520231317463 | -5059431.752672512 | 0.136623614986609 | 1.608564669199853 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=12.9% EAD = +2,196,543 EUR (derivado precio pr… | 12.9% EAD (backup si walk-away; floor propio >= min 8.0% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=3.0% | Plazo_extra_max=+12m | Tasa_min=10.0% | DSC… | Dentro de banda PRUDENCIAL. Continuar negociación. Walk-away… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [ESTRECHA (prudencial: concesiones … | PRUDENCIAL_RESTRUCT_CARVEOUT_CLEAN|top20pct|EVA>=7,954,363|D… | [PRUDENCIAL] MIXED_DECISION: Micro=REESTRUCTURAR → Final=MAN… |
| L000226 | Large Corporate | 25043940 | REESTRUCTURAR | RC03_PRUDENCIAL_CARVEOUT_RESTRUCT | -4658172.84 | 4971430.900111323 | -0.024 | 0.2180548254927614 | -7646651.751677651 | 0.1363483122530012 | 1.651218226003044 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=12.9% EAD = +3,226,891 EUR (derivado precio pr… | 12.9% EAD (backup si walk-away; floor propio >= min 8.0% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=3.0% | Plazo_extra_max=+12m | Tasa_min=10.0% | DSC… | Dentro de banda PRUDENCIAL. Continuar negociación. Walk-away… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [ESTRECHA (prudencial: concesiones … | PRUDENCIAL_RESTRUCT_CARVEOUT_CLEAN|top20pct|EVA>=7,954,363|D… | [PRUDENCIAL] MIXED_DECISION: Micro=REESTRUCTURAR → Final=MAN… |
| L000426 | Corporate | 90956907 | MANTENER | RC05_KEEP_ACCEPTABLE_ECONOMICS | -15735544.911 | 18056362.38866975 | -0.01533333333333333 | 0.2180591275632561 | -23487326.88564626 | 0.1320288958184028 | 1.246288191395768 |  | NO_ACTION | [N/A — floor: 12.5% EAD = +11,348,448 EUR] | 12.5% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | ECONOMICS_KEEP — loan genera valor holding > valor liquidaci… | Si RORWA_pre cae por debajo de hurdle (10.0%) O si aparece o… |  | [PRUDENCIAL] MIXED_DECISION: Micro=REESTRUCTURAR → Final=MAN… |
| L000468 | Large Corporate | 82382036 | MANTENER | RC02_SELL_BLOCKED_FIRE_SALE | -14664002.408 | 16354093.40177638 | -0.01866666666666666 | 0.2180589574240547 | -31438124.57034084 | 0.1585082219769753 | 1.114133818852923 |  | NO_ACTION | [N/A — floor: 15.0% EAD = +12,340,027 EUR] | 15.0% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | FIRE_SALE_BLOCK — venta suspendida hasta condiciones de merc… | Si mercado_bid >= 15.0% EAD Y sin fire-sale → re-evaluar VEN… |  | [PRUDENCIAL] MACRO_HARDENING_STEERING: R1_PRUDENCIAL_FIRE_SA… |
| L000418 | Large Corporate | 82592045 | MANTENER | RC05_KEEP_ACCEPTABLE_ECONOMICS | -14536199.92 | 9784880.300239842 | -0.01733333333333333 | 0.1706452982056079 | -23738214.19365321 | 0.1491158438141452 | 1.074166910669451 |  | NO_ACTION | [N/A — floor: 14.1% EAD = +11,638,414 EUR] | 14.1% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | ECONOMICS_KEEP — loan genera valor holding > valor liquidaci… | Si RORWA_pre cae por debajo de hurdle (10.0%) O si aparece o… |  | [PRUDENCIAL] MIXED_DECISION: Micro=REESTRUCTURAR → Final=MAN… |

### Análisis CIB — PRUDENCIAL

**VENTAS:** La postura prudencial no ejecuta ninguna venta en este ciclo.
- 71 loans bloqueados por fire-sale (RC02): mercado ilíquido / spread bid-ask excesivo.
- 34 loans bloqueados por guardrail P&L (RC_GUARDRAIL_BLOCK): pérdida potencial supera threshold EAD×40%.
- Remaining 361 MANTENER: economía holding > liquidación actual.

**FLOOR LATENTE:** 14.5% del EAD.
Si precio de mercado mejora hasta este nivel Y desaparece el fire-sale → re-evaluar VENDER.

**WALK-AWAY REESTRUCTURAR:** Banda estrecha — quita máx. 3.0%, DSCR mín. 1.30. Si el deudor exige condiciones más agresivas → MANTENER.

## 3.2 POSTURA: BALANCEADO

> **Objetivo:** Trade-off EVA vs capital. Activar ventas con precio razonable, priorizar reestructuras con ΔEVA claro, controlar concentración (HHI).

### Distribución Reason_Code_Final

| Reason_Code | Loans |
|-------------|-------|
| RC03_MICRO_RESTRUCT_VALUE_UPLIFT | 244 |
| RC06_MICRO_SELL_VALUE_NEGATIVE | 168 |
| RC03_MACRO_RESTRUCT_STEERING | 45 |
| RC05_KEEP_ACCEPTABLE_ECONOMICS | 41 |
| RC01_MACRO_SELL_STEERING | 1 |
| RC_GUARDRAIL_BLOCK | 1 |

### Convergencia

| Convergencia_Caso | Loans |
|-----------------|-------|
| MACRO_NOT_APPLIED | 450 |
| MACRO_HARDENING | 46 |
| AGREE_MICRO_MACRO | 4 |

### Top 10 Loans Frontera (BALANCEADO)

| loan_id | segment | EAD | Accion_final | Reason_Code_Final | EVA_pre | EVA_post | RORWA_pre | RORWA_post | pnl | Price_to_EAD | DSCR_post | PTI_post | GoToMarket | Indicative_Bid_Range | Reservation_Floor | Anchor_Terms | Concession_Band | WalkAway_Rule_Text | Fallback_Reason | Trigger_to_Action | Decision_Governance_Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L000058 | Large Corporate | 16224307 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -2887926.646 | 0.0 | -0.01866666666666666 |  | -6469715.093295344 | 0.1552817584821383 | 1.151754058735836 |  | BLOQUEADO_FIRE_SALE | [+2,154,292 EUR, +2,762,704 EUR] | 13.6% EAD = +2,199,383 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 13.6% EAD = +2,199,383 EUR [floor propio] → MANT… | [BALANCEADO] MIXED_DECISION: Micro=REESTRUCTURAR → Final=VEN… |
| L000374 | Large Corporate | 12008417 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -2138545.753991765 | 0.0 | -0.01872482187517665 |  | -4762808.12978068 | 0.1314086855355099 | 1.1551906562587 |  | BLOQUEADO_FIRE_SALE | [+1,320,926 EUR, +1,758,137 EUR] | 11.5% EAD = +1,377,603 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 11.5% EAD = +1,377,603 EUR [floor propio] → MANT… | [BALANCEADO] MIXED_DECISION: Micro=REESTRUCTURAR → Final=VEN… |
| L000189 | Corporate | 5449220 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -942715.06 | 0.0 | -0.01533333333333333 |  | -2147279.798483266 | 0.1547775433667532 | 1.33311422372491 |  | BLOQUEADO_FIRE_SALE | [+720,809 EUR, +925,155 EUR] | 13.5% EAD = +736,303 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 13.5% EAD = +736,303 EUR [floor propio] → MANTEN… | [BALANCEADO] MIXED_DECISION: Micro=REESTRUCTURAR → Final=VEN… |
| L000247 | Corporate | 1772800 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -320876.8 | 0.0 | -0.02066666666666666 |  | -698530.5826817746 | 0.1655266961480057 | 1.156539086608058 |  | BLOQUEADO_FIRE_SALE | [+253,558 EUR, +320,038 EUR] | 14.5% EAD = +256,178 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 14.5% EAD = +256,178 EUR [floor propio] → MANTEN… | [BALANCEADO] MIXED_DECISION: Micro=REESTRUCTURAR → Final=VEN… |
| L000413 | Project Finance | 10162806 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -1530490.101975592 | 122932.5369323812 | -0.0003981316430122299 | 0.1076256431669709 | -3337148.686153578 | 0.08754493586701852 | 1.111035189022252 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=7.6% EAD = +776,710 EUR (derivado precio propi… | 7.6% EAD (backup si walk-away; floor propio >= min 5.5% EAD) | Tasa_propuesta=11.0% | Plazo=240m | Quita=5.0% | DSCR_post=1… | Quita_max=10.0% | Plazo_extra_max=+24m | Tasa_min=7.0% | DSC… | REVIEW_REQUIRED: DSCR_post=1.11 está en la zona de proximida… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [MEDIA (balanceado: trade-off EVA v… | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD… |
| L000472 | Large Corporate | 23030984 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -4099515.152 | 4571822.256345029 | -0.01866666666666667 | 0.2180543066161045 | -9350526.584396098 | 0.1524963330265564 | 1.111453940859887 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=13.3% EAD = +3,066,099 EUR (derivado precio pr… | 13.3% EAD (backup si walk-away; floor propio >= min 5.5% EAD… | Tasa_propuesta=19.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=10.0% | Plazo_extra_max=+24m | Tasa_min=7.0% | DSC… | REVIEW_REQUIRED: DSCR_post=1.11 está en la zona de proximida… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [MEDIA (balanceado: trade-off EVA v… | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD… |
| L000115 | Project Finance | 12779854 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -2159965.581803345 | 1003838.177971385 | -0.01267554813502277 | 0.1469285008171226 | -4483771.469711255 | 0.09855346980308742 | 1.123641861408986 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=8.6% EAD = +1,099,543 EUR (derivado precio pro… | 8.6% EAD (backup si walk-away; floor propio >= min 5.5% EAD) | Tasa_propuesta=13.0% | Plazo=240m | Quita=0.0% | DSCR_post=1… | Quita_max=10.0% | Plazo_extra_max=+24m | Tasa_min=7.0% | DSC… | REVIEW_REQUIRED: DSCR_post=1.12 está en la zona de proximida… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [MEDIA (balanceado: trade-off EVA v… | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD… |
| L000367 | MidCap | 266862 | MANTENER | RC_GUARDRAIL_BLOCK | -40655.57371440424 | 52726.99596649931 | -0.001564538261733893 | 0.2175036299507564 | -111786.7140102237 | 0.141986837289146 | 1.472874414599455 |  | NO_ACTION | [N/A — floor: 12.4% EAD = +33,079 EUR] | 12.4% EAD (latente, no activo) | N/A (MANTENER) | N/A (MANTENER) | N/A (MANTENER — evaluar si score_sell mejora) | GUARDRAIL_BLOCK — pérdida superaría umbral de P&L admisible | Si mercado_bid >= 12.4% EAD y pérdida < 40% EAD → re-evaluar | [BALANCEADO] GUARDRAIL_OVERRIDE: PNL_TOO_NEGATIVE_EAD40 [pos… |

### Análisis CIB — BALANCEADO

**VENTAS (169 loans, 33.8%):**
- 169 loans con P&L medio = -30.7% del EAD.
- Floor activo: 11.0% EAD. Sólo ejecutar si la oferta supera este nivel.
- Fire-sale activo en la mayoría de loans VENDER: recomendación de GO-TO-MARKET sólo para los que no tienen este flag.

**REESTRUCTURAS (289 loans):**
- ΔEVA mediano = +8,496,322 EUR.
- DSCR_post p50 ≈ 2.07 (buena viabilidad post-workout).
- Banda media: quita máx 10.0%, tasa mín 7.0%.
- HHI segmento Large Corporate = 0.494 > umbral 0.30 → R2 macro aplicado (1 loan rotado a VENDER).

**DIFERENCIACIÓN vs PRUDENCIAL:**
- Sell rate: 33.8% vs 0.0% (prudencial). Gap = 33.8%.
- Capital liberado: +411,429,723 EUR vs +76,827,915 EUR (prud).

## 3.3 POSTURA: DESINVERSION

> **Objetivo:** Liberar capital y reducir RWA. Venta como acción principal con floor más bajo que balanceado. Reestructura sólo como excepción operada.

### Distribución Reason_Code_Final

| Reason_Code | Loans |
|-------------|-------|
| RC06_MICRO_SELL_VALUE_NEGATIVE | 358 |
| RC03_MACRO_RESTRUCT_STEERING | 84 |
| RC05_KEEP_ACCEPTABLE_ECONOMICS | 53 |
| RC03_MICRO_RESTRUCT_VALUE_UPLIFT | 5 |

### Convergencia

| Convergencia_Caso | Loans |
|-----------------|-------|
| MACRO_NOT_APPLIED | 365 |
| MACRO_HARDENING | 134 |
| GUARDRAIL_OVERRIDE | 1 |

### Top 10 Loans Frontera (DESINVERSION)

| loan_id | segment | EAD | Accion_final | Reason_Code_Final | EVA_pre | EVA_post | RORWA_pre | RORWA_post | pnl | Price_to_EAD | DSCR_post | PTI_post | GoToMarket | Indicative_Bid_Range | Reservation_Floor | Anchor_Terms | Concession_Band | WalkAway_Rule_Text | Fallback_Reason | Trigger_to_Action | Decision_Governance_Final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L000054 | SME | 139465 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -24266.91 | 0.0 | -0.01600000000000001 |  | -55773.90134395173 | 0.1583944034674588 | 1.022360170365771 |  | SI | [+18,953 EUR, +24,182 EUR] | 12.3% EAD = +17,208 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 12.3% EAD = +17,208 EUR [floor propio] → MANTENE… | [DESINVERSION] MICRO_LED: micro-model sell decision. EVA_pre… |
| L000402 | SME | 771197 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -134188.278 | 0.0 | -0.016 |  | -308239.8884537666 | 0.1618914344279624 | 1.05287559650931 |  | SI | [+107,498 EUR, +136,418 EUR] | 12.6% EAD = +97,258 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 12.6% EAD = +97,258 EUR [floor propio] → MANTENE… | [DESINVERSION] MICRO_LED: micro-model sell decision. EVA_pre… |
| L000058 | Large Corporate | 16224307 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -2887926.646 | 0.0 | -0.01866666666666666 |  | -6469715.093295344 | 0.1552817584821383 | 1.019371817931531 |  | SI | [+2,154,292 EUR, +2,762,704 EUR] | 12.1% EAD = +1,962,565 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 12.1% EAD = +1,962,565 EUR [floor propio] → MANT… | [DESINVERSION] MICRO_LED: micro-model sell decision. EVA_pre… |
| L000103 | Mortgage | 448055 | VENDER | RC06_MICRO_SELL_VALUE_NEGATIVE | -60911.99180183938 | 0.0 | -0.03594757742205617 |  | -178645.7904461331 | 0.1773845735550093 |  | 0.301174632045914 | SI | [+69,397 EUR, +86,199 EUR] | 13.8% EAD = +61,913 EUR [derivado precio propio] | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | N/A (VENDER) | Si oferta < 13.8% EAD = +61,913 EUR [floor propio] → MANTENE… | [DESINVERSION] MICRO_LED: micro-model sell decision. EVA_pre… |
| L000002 | Project Finance | 13868974 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -2307339.017712519 | 0.0 | -0.01091130546559146 | 0.0 | -4000643.755997523 | 0.07644826775116861 | 1.08780585789082 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=6.0% EAD = +825,942 EUR (derivado precio propi… | 6.0% EAD (backup si walk-away; floor propio >= min 3.5% EAD) | Tasa_propuesta=0.0% | Plazo=0m | Quita=0.0% | DSCR_post=1.09… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | WALK_AWAY ACTIVO: Tasa=0.0% < min=15.0% → Mantener o evaluar… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RES… |
| L000009 | Corporate | 3768147 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -689570.901 | 0.0 | -0.02199999999999999 | 0.0 | -1570530.277915198 | 0.1673060643249398 | 1.239550910636251 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=13.0% EAD = +491,108 EUR (derivado precio prop… | 13.0% EAD (backup si walk-away; floor propio >= min 3.5% EAD… | Tasa_propuesta=0.0% | Plazo=0m | Quita=0.0% | DSCR_post=1.24… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | WALK_AWAY ACTIVO: Tasa=0.0% < min=15.0% → Mantener o evaluar… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RES… |
| L000011 | Mortgage | 150759 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -20933.80720446096 | 0.0 | -0.03885610281615664 | 0.0 | -62668.30325857137 | 0.1726965573839793 |  | 0.3250957860062318 | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=13.5% EAD = +20,282 EUR (derivado precio propi… | 13.5% EAD (backup si walk-away; floor propio >= min 3.5% EAD… | Tasa_propuesta=0.0% | Plazo=0m | Quita=0.0% | PTI_post=32.5%… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | WALK_AWAY ACTIVO: Tasa=0.0% < min=15.0% → Mantener o evaluar… | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RES… |
| L000480 | Corporate | 1247606 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -218708.9444727522 | 0.0 | -0.01686859712267722 | 0.0 | -332057.8704719685 | 0.08066192768539394 | 0.7842442825698045 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=6.3% EAD = +78,394 EUR (derivado precio propio… | 6.3% EAD (backup si walk-away; floor propio >= min 3.5% EAD) | Tasa_propuesta=0.0% | Plazo=0m | Quita=0.0% | DSCR_post=0.78… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | WALK_AWAY ACTIVO: DSCR_post=0.78 < minimo=1.05; Tasa=0.0% < … | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RES… |
| L000355 | MidCap | 641782 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -107819.376 | 0.0 | -0.012 | 0.0 | -275127.6953113185 | 0.1478311267282208 | 0.8187653006493978 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=11.5% EAD = +73,908 EUR (derivado precio propi… | 11.5% EAD (backup si walk-away; floor propio >= min 3.5% EAD… | Tasa_propuesta=0.0% | Plazo=0m | Quita=0.0% | DSCR_post=0.82… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | WALK_AWAY ACTIVO: DSCR_post=0.82 < minimo=1.05; Tasa=0.0% < … | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RES… |
| L000391 | Large Corporate | 27591648 | REESTRUCTURAR | RC03_MACRO_RESTRUCT_STEERING | -4485288.445530513 | 0.0 | -0.00837309525767877 | 0.0 | -11572443.92863474 | 0.1134125909153597 | 0.8328940915161545 |  | NO_ACTIVO_REESTRUCTURAR_PRIORITARIO | [Backup_floor=8.8% EAD = +2,437,678 EUR (derivado precio pro… | 8.8% EAD (backup si walk-away; floor propio >= min 3.5% EAD) | Tasa_propuesta=0.0% | Plazo=0m | Quita=0.0% | DSCR_post=0.83… | Quita_max=5.0% | Plazo_extra_max=+12m | Tasa_min=15.0% | DSC… | WALK_AWAY ACTIVO: DSCR_post=0.83 < minimo=1.05; Tasa=0.0% < … | N/A (REESTRUCTURAR) | Si deudor fuera de banda [EXCEPCION (desinversion: sólo si v… | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RES… |

### Análisis CIB — DESINVERSION

**VENTAS (358 loans, 71.6%):**
- Floor bajo: 9.0% EAD. Acepta liquidación agresiva pero no precio insulto.
- P&L mediano estimado: -31.5% del EAD sold.
- Capital liberado total: +515,266,481 EUR — mayor de las 3 posturas.
- RWA liberado: ≈ +3,404,293,631 EUR.

**REESTRUCTURAS (89 loans — EXCEPCIÓN):**
- Activadas por R3 macro steering (capital_liberado=0 y ΔEVA>0).
- DSCR_post más bajo (p50 ≈ 1.08): mayor riesgo de recaída.
- Banda de excepción: quita máx 5.0%, tasa mín 15.0% (prima de riesgo elevada).

**DIFERENCIACIÓN vs BALANCEADO:**
- Sell rate: 71.6% vs 33.8%.
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

- Monotonía ventas: DESINVERSION (358) >= BALANCEADO (169) >= PRUDENCIAL (0) ✅
- Monotonía capital liberado: DESINV (+515,266,481 EUR) >= BALANC (+411,429,723 EUR) >= PRUD (+76,827,915 EUR) ✅
- EVA preservado mayor en PRUDENCIAL: +748,631,942 EUR ✅

---

## 5. Restricciones y Disclaimers

- Este documento es de uso **interno del banco**, generado automáticamente por el NPL Optimizer.
- **NO fija precio final** de venta ni términos contractuales de reestructuración.
- Los rangos indicativos se basan en el pricing model interno; no constituyen valoración de mercado.
- Toda nego por fuera del envelope debe ser aprobada por el Comité de Crédito NPL.
- Los datos de DSCR/PTI se calculan con supuestos del modelo; verificar con datos reales del deudor.
- Arquivos generados: `decisiones_finales_{postura}_npl.xlsx` + `enforcement_log_{tag}.csv`.
