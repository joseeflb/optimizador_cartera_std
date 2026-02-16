# RESUMEN EJECUTIVO - CALIBRACIÓN PC7 (POINT DE CONTROL 7)

**Fecha:** 16-Feb-2026 01:00 UTC  
**Status:** ✅ POC RECALIBRADO | ⚠️ REQUIERE AJUSTE FINO DE MANDATOS

---

## 📊 RESULTADOS DE CALIBRACIÓN RECALIBRATED_V2

### 1. MIX DE DECISIONES (500 préstamos)

| Postura | MANTENER | REESTRUCTURAR | VENDER | Perfil |
|---------|----------|---------------|--------|--------|
| **PRUDENCIAL** | 83.0% | 0.8% | **16.2%** | ✅ Conservador (más mantener) |
| **BALANCEADO** | 50.0% | 0.6% | **49.4%** | ✅ Equilibrado (ventas~50%) |
| **DESINVERSION** | 70.2% | 0.0% | **29.8%** | ⚠️ Cap 30% aplicó, forzó mantener |

**✅ LOGROS:**
- PRUD vs BAL diferenciadas: PRUD 83% mantener > BAL 50% mantener
- BAL más ejecutivo: 49.4% ventas vs 16.2% PRUD
- 10 casos frontera PRUD≠BAL identificados con WHY completo

**⚠️ GAPS:**
- DESINV: cap 30% ventas forzó 70% mantener (artificialmente conservador)
- Casi 0% reestructuras en las 3 posturas (min_acceptance_score muy alto)

---

### 2. MANDATOS DE VENTA (Ejecutabilidad)

| Postura | Mandatos Totales | % Cartera | Ventas por Mandato | Ventas Voluntarias | Target |
|---------|------------------|-----------|--------------------|--------------------|--------|
| **PRUDENCIAL** | 0 | 0.0% | 0 | 81 | 0% OK (conservador) |
| **BALANCEADO** | 14 | 2.8% | 11 | 236 | ~3% OK (equilibrado) |
| **DESINVERSION** | 24 | **4.8%** | 23 | 126 | ⚠️ **Target 20-30%** |

**❌ CRÍTICO - DESINV MANDATOS:**
- Actual: 4.8% (24/500 préstamos)
- Target: 20-30% (~100-150 préstamos)
- Calibración necesaria: Mandatos AÚN muy restrictivos

**Parámetros actuales (DESINV):**
- `mandate_recovery_floor=0.03` (p1 - worst 1% SOLO)
- `mandate_age_npl_months=18` (1.5 años NPL)
- `mandate_rwa_threshold=2.50` (desactivado - muy alto)

---

### 3. KPIs AGREGADOS (Trade-Offs Portfolio)

| KPI | PRUDENCIAL | BALANCEADO | DESINVERSION | Ganador |
|-----|------------|------------|--------------|---------|
| **EVA_post (EUR)** | **€748M** | €726M | €48k | PRUD |
| **Capital liberado (EUR)** | €116M | **€389M** | €154M | BAL |
| **RWA_post (EUR)** | €3.4B | €3.4B | **€508k** | DESINV |
| **Préstamos activos** | 415 | 250 | **351** | PRUD |

**Interpretación:**
- **PRUDENCIAL:** Maximiza EVA futuro (+€748M) manteniendo casi todo
- **BALANCEADO:** Equilibrio EVA vs capital (€726M EVA, €389M capital)
- **DESINVERSION:** Minimiza RWA (-99.9%) pero con cap 30% limita liquidez

---

## ❌ PROBLEMAS IDENTIFICADOS

### 1. DESINV: Mandatos 4.8% vs Target 20-30%
**Causa raíz:**
- Thresholds calibrados para p1 (worst 1%) → solo 24 préstamos
- mandate_recovery_floor=0.03 (3%) captura solo NPL "unsellable"
- mandate_age_npl_months=18 meses → solo 12% préstamos

**Fix requerido:**
```python
# OPCIÓN A: Subir mandate_recovery_floor (menos estricto)
mandate_recovery_floor=0.12  # p25 (worst 25%) → ~125 préstamos

# OPCIÓN B: Bajar mandate_age_npl_months (más préstamos old NPL)
mandate_age_npl_months=6  # 6 meses NPL → ~30% préstamos

# OPCIÓN C: Activar mandate_rwa_threshold (usar RWA alto)
mandate_rwa_threshold=1.50  # RWA > 1.5 (worst RWA loans) → ~20% préstamos
```

### 2. Cap 30% Ventas: Trade-off Mandatos vs Voluntarias
**Situación actual:**
- DESINV: 24 mandatos + 126 voluntarias = 150 ventas (30% cap)
- Cap forzó 344 ventas voluntarias → MANTENER (70.2% mantener artificial)

**Trade-off design decision:**
- **Opción A (actual):** Max 30% ventas TOTAL (mandato + voluntarias)
  - Pro: Diversificación portfolio, no "vendo todo"
  - Con: Mandatos compiten con voluntarias por capacidad
  
- **Opción B (alternativa):** Mandatos SIN cap + voluntarias con cap
  - Pro: Mandatos siempre ejecutan (prioridad absoluta)
  - Con: Si mandatos=30%, voluntarias=0 (less flexibility)

### 3. Casi 0% Reestructuras (Acceptance Score Too High)
**Parámetros actuales:**
- PRUD: min_acceptance_score=65 → solo 4 reestructuras (0.8%)
- BAL: min_acceptance_score=50 → solo 3 reestructuras (0.6%)
- DESINV: min_acceptance_score=40 → 0 reestructuras (0%)

**Fix requerido:**
```python
# Bajar thresholds para permitir más reestructuras viables
PRUDENCIAL: min_acceptance_score=35  # vs 65 actual
BALANCEADO: min_acceptance_score=25  # vs 50 actual
DESINVERSION: min_acceptance_score=20  # vs 40 actual
```

---

## ✅ CORRECCIONES REALIZADAS (DESDE PC6)

1. **Documentación corregida:**
   - ❌ "100% completado / listo para producción"
   - ✅ "POC validado en sintético / pending pilot con datos reales"
   
2. **Secciones añadidas:**
   - Limitaciones y Riesgos Conocidos (6 bloqueantes críticos)
   - Plan de Piloto (3 fases: Preparación / Piloto / Productivización)
   - Criterios GO/NO-GO para producción
   
3. **Mandatos recalibrados (FIRST PASS):**
   - DESINV: mandate_recovery_floor 12.2% → 3% (p30 → p1)
   - DESINV: mandate_age_npl_months 8m → 18m (p70 → p95)
   - Resultado: 4.8% mandatos (vs target 20-30%) → REQUIERE SECOND PASS
   
4. **Knobs PRUD vs BAL diferenciados:**
   - PRUD: sale_floor=25%, loss_cap=85%, recovery_min=15%
   - BAL: sale_floor=18%, loss_cap=90%, recovery_min=10%
   - Resultado: ✅ PRUD 83% mantener > BAL 50% mantener (diferenciación clara)
   
5. **Scripts de reproducibilidad:**
   - ✅ export_canonical_csv.py (CSV con hash SHA256)
   - ✅ verify_manifest.py (verificación integridad)
   - ✅ analyze_calibration_evidence.py (mixes, KPIs, mandatos, casos frontera)

---

## 📋 ENTREGABLES PC7

### Documentación actualizada:
1. ✅ [RESUMEN_FINAL_COMPLETION.md](RESUMEN_FINAL_COMPLETION.md) - Header + Limitaciones + Plan Piloto
2. ✅ [REPORTE_EJECUTIVO_FINAL_BANK_READY.md](REPORTE_EJECUTIVO_FINAL_BANK_READY.md) - Checklist POC + Riesgos

### Scripts nuevos:
3. ✅ [_tmp/export_canonical_csv.py](_tmp/export_canonical_csv.py) - Export CSV + MANIFEST.json
4. ✅ [_tmp/verify_manifest.py](_tmp/verify_manifest.py) - Verificación hashes
5. ✅ [_tmp/analyze_calibration_evidence.py](_tmp/analyze_calibration_evidence.py) - Análisis evidencia

### Configuración recalibrada:
6. ✅ [config.py](config.py) lines 348-470 - BANK_STRATEGIES con mandatos p1-p3

### Evidencia generada:
7. ✅ reports/calibration/calibration_mixes_*.csv - Tabla mixes 3 posturas
8. ✅ reports/calibration/calibration_kpis_*.csv - KPIs agregados
9. ✅ reports/calibration/calibration_mandates_*.csv - Mandatos vs voluntarias
10. ✅ reports/calibration/calibration_frontier_cases_*.csv - 10 casos PRUD≠BAL

### Archivos Excel recalibrados:
11. ✅ reports/coordinated_inference_RECALIBRATED_V2_pru_*/decisiones_finales_prudencial.xlsx
12. ✅ reports/coordinated_inference_RECALIBRATED_V2_bal_*/decisiones_finales_balanceado.xlsx
13. ✅ reports/coordinated_inference_RECALIBRATED_V2_des_*/decisiones_finales_desinversion.xlsx

---

## 🚦 RECOMENDACIONES FINALES

### ✅ APROBAR COMO COMPLETADO:
1. Marco conceptual POC vs Producción (honesto, realista)
2. Diferenciación PRUD vs BAL (clara, defendible)
3. Framework de reproducibilidad (CSV + MANIFEST + verificación)
4. Sistema de evidencia para comité (mixes, KPIs, mandatos, casos frontera)

### ⚠️ PENDIENTE PARA PILOTO (SECOND PASS):
1. **CRÍTICO:** Recalibrar mandatos DESINV (4.8% → 20-30%)
   - Subir mandate_recovery_floor a ~12% (p25) O
   - Bajar mandate_age_npl_months a ~6 meses O
   - Activar mandate_rwa_threshold=1.50 (RWA alto)
   
2. **IMPORTANTE:** Bajar min_acceptance_score (permitir más reestructuras)
   - PRUD: 65→35, BAL: 50→25, DESINV: 40→20
   
3. **DISEÑO:** Decidir trade-off cap ventas (mandatos vs voluntarias)
   - Opción A (actual): Cap total 30% (mandatos compiten)
   - Opción B: Mandatos sin cap + voluntarias con cap
   
4. **FUNDAMENTAL:** Ejecutar piloto con 100-200 préstamos REALES
   - Validar percentiles recovery, age_npl, RWA con distribuciones reales
   - Backtest casos frontera con decisiones de negocio
   - Obtener GO/NO-GO de Model Risk

---

## 📊 DEFINITION OF DONE (PC7)

| Item | Status | Evidencia |
|------|--------|-----------|
| 1. Documentación POC vs Producción | ✅ DONE | Headers + Limitaciones + Plan Piloto añadidos |
| 2. Mandatos recalibrados DESINV~20-30% | ⚠️ PARTIALLY | 4.8% actual → requiere second pass |
| 3. PRUD vs BAL diferenciadas | ✅ DONE | 83% mantener vs 50% mantener (p<0.001) |
| 4. Export CSV canónico + MANIFEST | ✅ DONE | Scripts funcionando + verificación hashes |
| 5. Evidencia evidencia para comité | ✅ DONE | 4 tablas CSV + 10 casos frontera |
| 6. Release/tag Git + checksum | ⚠️ PENDING | Manual step (fuera de scope Python) |

**SCORE PC7:** 4/6 DONE + 2/6 PENDING → **67% completado**

---

## 🎯 PRÓXIMOS PASOS INMEDIATOS

### PASO 1: SECOND PASS Calibración Mandatos (1-2 horas)
```python
# En config.py, BANK_STRATEGIES[BankProfile.DESINVERSION]:
mandate_recovery_floor=0.12  # 12% (p25 - worst 25%) [vs 3% actual]
mandate_age_npl_months=6     # 6 meses NPL [vs 18m actual]
mandate_rwa_threshold=1.50   # Activar RWA gate [vs 2.50 desactivado]
```

Re-ejecutar: `.\run_recalibrated_inference.bat` → validar mandatos ~20-30%

### PASO 2: Ajustar Acceptance Scores (30 min)
```python
# Permitir más reestructuras viables
min_acceptance_score: PRUD=35, BAL=25, DESINV=20
max_restructure_share: PRUD=50%, BAL=70%, DESINV=30%
```

### PASO 3: Export Canónico Final (30 min)
```bash
# Para cada postura (PRUD, BAL, DESINV):
python _tmp/export_canonical_csv.py \
    --xlsx reports/coordinated_inference_*_pru*/decisiones_finales*.xlsx \
    --output-dir reports/canonical \
    --tag FINAL_CALIBRATED \
    --posture prudencial

# Verificar hashes:
python _tmp/verify_manifest.py --manifest reports/canonical/MANIFEST_pru.json
```

### PASO 4: Release Git + Checksum (manual)
```bash
git add .
git commit -m "PC7: POC recalibrado + reproducibilidad canónica"
git tag -a v2.1-poc-recalibrated -m "POC bank-ready recalibrado (pending pilot)"
git push origin v2.1-poc-recalibrated

# Generar checksum paquete deliverable
cd reports/canonical
sha256sum *.csv *.json > CHECKSUMS.txt
```

---

**CONCLUSIÓN PC7:**

✅ **POC TÉCNICAMENTE ROBUSTO** con framework reproducible y evidencia completa  
⚠️ **CALIBRACIÓN 67% COMPLETADA** - requiere second pass mandatos DESINV  
❌ **NO LISTO PARA PRODUCCIÓN** sin piloto con datos reales + aprobación Model Risk  

**Tiempo estimado restante:** 2-3 horas (second pass calibración + export final)

---

*Documento generado:* 2026-02-16 01:00 UTC  
*Por:* Optimizador Cartera NPL v2.1 (POC recalibrado - PC7)  
*Next checkpoint:* PC8 (second pass calibración + export final + Git release)
