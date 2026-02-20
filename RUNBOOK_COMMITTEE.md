# RUNBOOK: Generación de Committee Pack (Auditoría)

Este documento describe el procedimiento **estándar y reproducible** para generar el paquete de evidencias para el Comité de Modelos y Auditoría.

## Prerrequisitos

- Entorno Windows con PowerShell.
- Python 3.10+ instalado.
- Acceso al repositorio `optimizador_cartera_std`.
- Datos de entrada en `data/portfolio_synth.xlsx`.

## Procedimiento "1-Click"

Ejecuta los siguientes comandos en orden secuencial desde la raíz del proyecto:

### 1. Validación de Entorno (Smoke Test)
Verifica que el entorno y las dependencias básicas funcionan.
```powershell
.\smoke_test_venv.bat
```

### 2. Ejecución de Inferencia (3 Posturas)
Ejecuta el ciclo completo de inferencia para las estrategias: Prudencial, Balanceado, Desinversión.
```powershell
# Ajustar el tag según ciclo (ej: pc5_postures_validation)
.\run_3_postures_executability_venv.bat --tag pc5_postures_validation
```

### 3. Comparativa de Estrategias
Genera la tabla comparativa cruzada de decisiones y métricas financieras.
```powershell
.\.venv\Scripts\python -m reports.compare_postures --tag pc5_postures_validation
```

### 4. Evaluación vs Baselines (Regulatory Check)
Simula reglas deterministas (Prudencial/Desinv.) y compara contra RL.
```powershell
.\.venv\Scripts\python reports/evaluate_against_baselines_sim.py --tag pc5_postures_validation --select latest --out reports/evaluation_pc6.csv --report reports/evaluation_report.md
```

### 5. Generación del Committee Pack
Empaqueta todos los resultados, logs, configuración y manifiesto de ejecución en un ZIP/Folder inmutable.
```powershell
.\.venv\Scripts\python -m reports.make_committee_pack --tag pc5_postures_validation
```

### 6. Validación Final (Test Suite)
Ejecuta la batería de pruebas automatizadas para garantizar integridad.
```powershell
.\.venv\Scripts\python -m pytest -q
```

## Salida Esperada

El resultado final estará en:
`reports/committee_pack_pc5_postures_validation_<TIMESTAMP>/`

Contiene:
- `MANIFEST.json`: Trazabilidad completa (Git commit, hashes, entorno).
- `run_<posture>/`: Excel de decisiones y logs para cada estrategia.
- `evaluation_report.md`: Resumen ejecutivo de impacto financiero.
- `config.py`: Snapshot de la configuración utilizada.
