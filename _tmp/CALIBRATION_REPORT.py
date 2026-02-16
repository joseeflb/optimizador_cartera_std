"""
REPORT: CALIBRATION CORRECTIONS AND EXPECTED OUTCOMES

PROBLEMA DIAGNOSTICADO:
======================
- Floors configurados: PRUD=50%, BAL=40%, DESINV=30%
- Mercado NPL real: ratio mediano sale_price/valor_ref = 28%
- Resultado: 100% insultantes en PRUD (97.8% BAL, 68.2% DESINV)

CAUSA ROOT:
===========
Floors demasiado altos para mercado NPL. Pricing es CORRECTO (unidades OK),
pero thresholds NO calibrados para realidad NPL.

CALIBRACION APLICADA:
====================
config.py - BankStrategy floors recalibrados:

PRUDENCIAL (conservador pero ejecutable):
  - sale_floor_ratio: 0.50 → 0.20 (20%)  
  - min_acceptance_score: 65 → 60
  - max_restructure_share: 0.30 → 0.40 (40%)
  - loss_cap_pct: 0.40 → 0.50 (50%)
  - mandate_loss_tolerance: 0.45 → 0.60

BALANCEADO (equilibrado - MAS ejecutivo):
  - sale_floor_ratio: 0.40 → 0.15 (15%)  ** MAS FLEXIBLE
  - min_acceptance_score: 50 → 45         ** MAS EJECUTIVO
  - max_restructure_share: 0.50 → 0.60 (60%)  ** MAYOR CAPACIDAD
  - loss_cap_pct: 0.50 → 0.60 (60%)
  - mandate_loss_tolerance: 0.55 → 0.70

DESINVERSION (agresivo):
  - sale_floor_ratio: 0.30 → 0.10 (10%)  ** MUY FLEXIBLE
  - min_acceptance_score: 40 → 35        ** ACEPTA MAS
  - max_restructure_share: 0.20 → 0.30 (30%)
  - loss_cap_pct: 0.65 → 0.70 (70%)
  - mandate_loss_tolerance: 0.70 → 0.80 (80%)

RESULTADOS ESPERADOS (basado en audit):
=======================================

Con floor=20% (PRUDENCIAL):
- ~30% ejecutables (prices > p30 del audit)
- 70% bloqueados (insulting o loss-cap)
- DIFERENCIACION vs BAL: menos ventas, mas mantener

Con floor=15% (BALANCEADO):
- ~55% ejecutables (prices > p15)
- 45% bloqueados
- DIFERENCIACION vs PRUD: MAS EJECUTIVO (mas ventas)
- DIFERENCIACION vs DESINV: menos agresivo

Con floor=10% (DESINVERSION):
- ~85% ejecutables (prices > p10)
- Solo 15% bloqueados por insurging
- Muchos mandatos (RWA threshold=1.50 vs 2.50 PRUD)
- DIFERENCIACION: el mas agresivo en ejecucion

MONOTONICIDAD ESPERADA:
========================
%VENTAS: PRUD < BAL < DESINV
  Ejemplo: 20% < 35% < 60%

%MANTENER: PRUD > BAL > DESINV  
  Ejemplo: 65% > 50% > 25%

%REESTRUCTURAR: puede variar (depende de acceptance_score y capacidad)
  BALANCEADO deberia tener MAS por capacidad 60% vs 40% PRUD

MANDATE_BLOCKED: DESINV > BAL > PRUD
  DESINV tiene mas mandatos por RWA threshold bajo

EVIDENCIA REQUERIDA:
====================
1. Tabla mix acciones (3 posturas)
2. Percentiles ratio sale_price/valor_ref (confirmacion unidades)
3. Conteos:
   - sale_mandate por postura
   - sale_insulting_flag por postura
   - ventas ejecutadas (sale_executable=True)
   - restruct_executable por postura
4. 10 casos frontera por postura:
   - mandate_blocked con next_step
   - prud vs bal diferentes (uno mantiene, otro ejecuta)

PROXIMOS PASOS:
===============
1. Ejecutar inferencias 3 posturas con floors recalibrados
2. Run analyze_executability.py (actualizado para v2)
3. Verificar monotonicidad PASS
4. Documentar resultados en DELIVERABLE
"""
print(__doc__)
