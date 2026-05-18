# BANK_READY_HARDENING — Tag: `infer_ci0222`

**Generado:** 20260222_162742  |  **Versión hardening:** 1.0.0  |  **Status:** ✅ PASS

---

## Resumen de Cambios

| Postura | RC incoherentes antes | RC incoherentes después | Flips Macro | Macro Rate |
|---------|----------------------|------------------------|-------------|------------|
| prudencial | 416 | 0 | 55 | 11.0% |
| balanceado | 189 | 0 | 46 | 9.2% |
| desinversion | 139 | 0 | 134 | 26.8% |

---

## Detalles por Postura

### PRUDENCIAL

**Distribución acciones hardened:**
- VENDER: 0 | REESTRUCTURAR: 0 | MANTENER: 500

**KPIs portfolio finales:**
- EVA_final: +748,631,942 € | RWA_final: 3,404,802,052 € | Capital liberado: 76,827,915 €

**Macro Steering:**
- Loans intervenidos: 55/500 (11.0%)
  - flips_r1: 55 flips
  - ⚠️ MACRO_INACTIVE_WARNING: macro_applied_rate=11.0% < 20% threshold. Portfolio steering is micro-dominated. Consider reviewing macro model coverage or hardening thresholds.

**Top 5 préstamos con governance modificada:**

| loan_id | Accion_final | RC_Before | RC_Final | Decision_Governance_Final |
|---------|-------------|-----------|----------|--------------------------|
| L000018 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC02_SELL_BLOCKED_FIRE_SALE | [PRUDENCIAL] MACRO_HARDENING_STEERING: R1_PRUDENCIAL_FIRE_SALE_BLOCK (umbral=|, variable=fire_sale=True|fire_sale/FireSa |
| L000038 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC02_SELL_BLOCKED_FIRE_SALE | [PRUDENCIAL] MACRO_HARDENING_STEERING: R1_PRUDENCIAL_FIRE_SALE_BLOCK (umbral=|, variable=fire_sale=True|fire_sale/FireSa |
| L000054 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC02_SELL_BLOCKED_FIRE_SALE | [PRUDENCIAL] MACRO_HARDENING_STEERING: R1_PRUDENCIAL_FIRE_SALE_BLOCK (umbral=|, variable=fire_sale=True|fire_sale/FireSa |
| L000058 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC02_SELL_BLOCKED_FIRE_SALE | [PRUDENCIAL] MACRO_HARDENING_STEERING: R1_PRUDENCIAL_FIRE_SALE_BLOCK (umbral=|, variable=fire_sale=True|fire_sale/FireSa |
| L000060 | MANTENER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC02_SELL_BLOCKED_FIRE_SALE | [PRUDENCIAL] MACRO_HARDENING_STEERING: R1_PRUDENCIAL_FIRE_SALE_BLOCK (umbral=|, variable=fire_sale=True|fire_sale/FireSa |


### BALANCEADO

**Distribución acciones hardened:**
- VENDER: 169 | REESTRUCTURAR: 289 | MANTENER: 42

**KPIs portfolio finales:**
- EVA_final: +725,865,394 € | RWA_final: 3,392,921,675 € | Capital liberado: 411,429,723 €

**Macro Steering:**
- Loans intervenidos: 46/500 (9.2%)
  - flips_r2: 1 flips
  - flips_r3: 45 flips
  - ⚠️ MACRO_INACTIVE_WARNING: macro_applied_rate=9.2% < 20% threshold. Portfolio steering is micro-dominated. Consider reviewing macro model coverage or hardening thresholds.

**Top 5 préstamos con governance modificada:**

| loan_id | Accion_final | RC_Before | RC_Final | Decision_Governance_Final |
|---------|-------------|-----------|----------|--------------------------|
| L000002 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC03_MACRO_RESTRUCT_STEERING | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD_RESTRUCT (umbral=|, variable=EAD>=+486,220€ & RORWA_pre<10. |
| L000019 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC03_MACRO_RESTRUCT_STEERING | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD_RESTRUCT (umbral=|, variable=EAD>=+486,220€ & RORWA_pre<10. |
| L000026 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC03_MACRO_RESTRUCT_STEERING | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD_RESTRUCT (umbral=|, variable=EAD>=+486,220€ & RORWA_pre<10. |
| L000027 | VENDER | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC01_MACRO_SELL_STEERING | [BALANCEADO] MACRO_HARDENING_STEERING: R2_BALANCEADO_HHI_CONCENTRATION (umbral=|, variable=HHI_segment>0.30|segment=Larg |
| L000053 | REESTRUCTURAR | RC03_MICRO_RESTRUCT_VALUE_UPLIFT | RC03_MACRO_RESTRUCT_STEERING | [BALANCEADO] MACRO_HARDENING_STEERING: R3_BALANCEADO_TOP_EAD_RESTRUCT (umbral=|, variable=EAD>=+486,220€ & RORWA_pre<10. |


### DESINVERSION

**Distribución acciones hardened:**
- VENDER: 358 | REESTRUCTURAR: 89 | MANTENER: 53

**KPIs portfolio finales:**
- EVA_final: +48,541 € | RWA_final: 508,421 € | Capital liberado: 515,266,481 €

**Macro Steering:**
- Loans intervenidos: 134/500 (26.8%)
  - flips_r1: 50 flips
  - flips_r3: 84 flips

**Top 5 préstamos con governance modificada:**

| loan_id | Accion_final | RC_Before | RC_Final | Decision_Governance_Final |
|---------|-------------|-----------|----------|--------------------------|
| L000001 | MANTENER | RC06_MICRO_SELL_VALUE_NEGATIVE | RC05_KEEP_ACCEPTABLE_ECONOMICS | [DESINVERSION] MACRO_HARDENING_STEERING: R1_DESINVERSION_INSULTING_PRICE (umbral=|, variable=price_ratio_ead<0.10|Price/ |
| L000002 | REESTRUCTURAR | RC15_VOLUME_CAP_VOLUNTARY_SELL | RC03_MACRO_RESTRUCT_STEERING | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RESTRUCT_FOR_CAP (umbral=|, variable=cap_release<=0 & ΔEVA>0|ca |
| L000009 | REESTRUCTURAR | RC_GUARDRAIL_BLOCK | RC03_MACRO_RESTRUCT_STEERING | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RESTRUCT_FOR_CAP (umbral=|, variable=cap_release<=0 & ΔEVA>0|ca |
| L000011 | REESTRUCTURAR | RC_GUARDRAIL_BLOCK | RC03_MACRO_RESTRUCT_STEERING | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RESTRUCT_FOR_CAP (umbral=|, variable=cap_release<=0 & ΔEVA>0|ca |
| L000012 | REESTRUCTURAR | RC_GUARDRAIL_BLOCK | RC03_MACRO_RESTRUCT_STEERING | [DESINVERSION] MACRO_HARDENING_STEERING: R3_DESINVERSION_RESTRUCT_FOR_CAP (umbral=|, variable=cap_release<=0 & ΔEVA>0|ca |

---

## Validaciones Automáticas

```
[PASS] prudencial: Accion_final vs Reason_Code_Final coherente al 100% (500 loans)
[PASS] balanceado: Accion_final vs Reason_Code_Final coherente al 100% (500 loans)
[PASS] desinversion: Accion_final vs Reason_Code_Final coherente al 100% (500 loans)
[PASS] prudencial: EVA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] prudencial: RWA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] prudencial: RORWA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] balanceado: EVA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] balanceado: RWA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] balanceado: RORWA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] desinversion: EVA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] desinversion: RWA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] desinversion: RORWA_post sin NaN para MANTENER/REESTRUCTURAR
[PASS] Monotonía ventas: desinversion(358) >= balanceado(169)
[PASS] Monotonía ventas: balanceado(169) >= prudencial(0)
[PASS] Monotonía capital_release: desinversion(515,266,481) >= balanceado(411,429,723)
[PASS] Monotonía capital_release: balanceado(411,429,723) >= prudencial(76,827,915)
[PASS] prudencial: ningún MANTENER con RC de VENDER
[PASS] balanceado: ningún MANTENER con RC de VENDER
[PASS] desinversion: ningún MANTENER con RC de VENDER
[PASS] prudencial: macro_steering registrado en KPIs (applied_rate=11.0%)
[PASS] balanceado: macro_steering registrado en KPIs (applied_rate=9.2%)
[PASS] desinversion: macro_steering registrado en KPIs (applied_rate=26.8%)
```

**Summary:** 22 PASS | 0 FAIL

---

## Reglas Macro Aplicadas (con umbrales y variables)

### PRUDENCIAL
| Regla | Umbral | Variable | Acción |
|-------|--------|----------|--------|
| R1_PRUDENCIAL_FIRE_SALE_BLOCK | fire_sale=True | fire_sale / FireSale_Triggered | VENDER → MANTENER |
| R2_PRUDENCIAL_PNL_AGG_LIMIT | sum(pnl_ventas)/total_EAD < -15% | pnl, EAD | VENDER → MANTENER (peor P&L) |

### BALANCEADO
| Regla | Umbral | Variable | Acción |
|-------|--------|----------|--------|
| R1_BALANCEADO_EVA_NEG_TO_RESTRUCT | EVA_post < 0 AND ΔEVA > 0 | EVA_post, ΔEVA | MANTENER → REESTRUCTURAR |
| R2_BALANCEADO_HHI_CONCENTRATION | HHI_segment > 0.30 | segment, EAD | MANTENER top-EAD → VENDER |
| R3_BALANCEADO_TOP_EAD_RESTRUCT | EAD >= p80 AND RORWA_pre < 10% AND ΔEVA > 0 | EAD, RORWA_pre | MANTENER → REESTRUCTURAR |

### DESINVERSION
| Regla | Umbral | Variable | Acción |
|-------|--------|----------|--------|
| R1_DESINVERSION_INSULTING_PRICE | price/EAD < 0.10 | Price_to_EAD | VENDER → MANTENER |
| R2_DESINVERSION_EXCESS_LOSS | pnl/EAD < -0.50 | pnl, EAD | VENDER → MANTENER |
| R3_DESINVERSION_RESTRUCT_FOR_CAP | cap_release=0 AND ΔEVA > 0 | capital_release_realized, ΔEVA | MANTENER → REESTRUCTURAR |

---

## Notas Técnicas

- `Reason_Code_Final` siempre coherente con `Accion_final` (regla dura, 100%).
- `EVA_post` para MANTENER = EVA_pre (sin drift). Para REESTRUCTURAR = EVA_pre + ΔEVA.
- `RORWA_post` para VENDER = NaN (diseño: activo fuera de libro).
- Guardrail `PNL_TOO_NEGATIVE_PRUDENCIAL` renombrado a `PNL_TOO_NEGATIVE_EAD40 [postura=X]`.
- `Macro_Steering_Applied` contiene `regla|umbral|variable` para trazabilidad completa.
- Ficheros generados en `reports/coordinated_inference_{tag}_*_{postura}/`.
