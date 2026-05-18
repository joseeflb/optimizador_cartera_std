# NPL Portfolio Optimizer — Integración n8n + Lovable Matrix

## Estructura

```
n8n/
  workflow_npl_pipeline.json    ← Importar en n8n (File → Import Workflow)

dashboard/
  api_server.py                 ← Flask API backend (lee datos reales del proyecto)
  run_dashboard.bat             ← Lanzador Windows
  static/
    index.html                  ← SPA dashboard (Chart.js + EY styling)
```

## 1. Flujo n8n – Pipeline de Automatización

### Importar
1. Abrir n8n (http://localhost:5678 o n8n cloud)
2. **Settings → Import Workflow** → seleccionar `n8n/workflow_npl_pipeline.json`
3. Ajustar `working_dir` en el nodo **"1 · Set Run Config"** si la ruta del proyecto difiere

### Nodos (9 pasos)
| Nodo | Función | Script |
|------|---------|--------|
| 1 | Set Run Config | Merge parámetros (tag, n_loans, postura, seed) |
| 2 | Generate Portfolio | `py main.py generate --n 500` |
| 2b | SHA-256 Check | Verifica integridad del Excel |
| 3a/3b | Model Check / Train | `py -m agent.train_subagents --agent both` |
| 4 | Coordinated Inference | `py main.py infer --all-postures --tag <tag>` |
| 5 | Stress Engine | `py -m engines.stress_engine --tag <tag>` |
| 6 | NPL Workout Layer | `py -m reports.npl_posture_analysis --tag <tag>` |
| 7 | Backtesting Dynamic | `py -m reports.backtesting_dynamic --tag <tag> --quarters 8` |
| 8 | Committee Pack | `py -m reports.make_committee_pack --tag <tag>` |
| 9 | Update Dashboard | HTTP POST → `http://localhost:3000/api/runs` |

### Triggers
- **Manual**: botón "Execute Workflow" en n8n
- **Cron**: semanal (configurable)
- **Webhook**: POST `http://n8n-host:5678/webhook/npl-pipeline` (para what-if desde dashboard)

---

## 2. Dashboard Lovable Matrix

### Lanzar
```bash
py dashboard/api_server.py
# → http://localhost:3000
```
O bien: doble clic en `dashboard/run_dashboard.bat`

### Requisitos
```
pip install flask flask-cors openpyxl
```

### Vistas (5)

| Vista | Contenido |
|-------|-----------|
| **KPI Executive** | Cards con ΔEVA, Capital liberado, RORWA, distribución de acciones. Semáforos (verde/ámbar/rojo). Gráficos de barras apiladas por postura. |
| **Decision Detail** | Tabla interactiva con 500 préstamos: loan_id, acción, Reason_Code, segmento, rating, EVA pre/post, capital liberado, Explanation. Filtros por texto, acción. |
| **Stress Analysis** | Backtesting dinámico 8Q × 3 posturas (EVA, curación, re-default). Matriz de estrés 4 escenarios × 3 posturas. |
| **Portfolio Map** | Scatter EAD vs PD coloreado por acción. Doughnut segmento. Barras rating. |
| **Posture Analysis** | Markdown del NPL Workout Layer: distance checks, negotiation envelopes, carve-outs. |

### API Endpoints

| Ruta | Método | Descripción |
|------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/tags` | GET | Tags de runs disponibles |
| `/api/kpis/<tag>` | GET | KPIs por postura (lee Excel real) |
| `/api/decisions/<tag>/<postura>` | GET | Detalle por préstamo |
| `/api/stress/<tag>` | GET | Backtesting dinámico + light |
| `/api/posture-analysis/<tag>` | GET | Markdown NPL Workout Layer |
| `/api/dispersion/<tag>` | GET | Análisis de dispersión |
| `/api/validation` | GET | Resultados validación agentes |
| `/api/models` | GET | Inventario modelos + SHA-256 |
| `/api/runs` | POST | Callback n8n (almacena en JSONL) |
| `/api/trigger-run` | POST | What-if: encolar nuevo run |

### What-If Analysis
El botón "⚙ What-If" del dashboard permite modificar parámetros (nº préstamos, seed, steps) y relanzar el pipeline vía el webhook n8n. Los resultados aparecen en el dashboard al completarse (~15 min).
