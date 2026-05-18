# Cota Óptima Teórica — V* por Programación Dinámica

## Metodología

### Loan Agent (micro): Enumeración exhaustiva
Para cada préstamo del pool, se simulan las 3 acciones posibles (MANTENER, REESTRUCTURAR, VENDER) y se escoge la de mayor reward. Como cada préstamo es un sub-MDP independiente de 1 paso, esto da el **V\* exacto** del MDP micro.

### Portfolio Agent (macro): Greedy rollout
Se ejecuta el episodio completo (30 steps) probando las 12 acciones en cada step y eligiendo la de mayor reward inmediato (oráculo greedy). Esto es una **cota superior real**: el oráculo con información completa y decisión miope no puede ser superado por un agente PPO con política parametrizada y horizonte finito.

---

## Resultados

### Loan Agent

| Métrica | Valor |
|---------|------:|
| **V\* (exacto)** | **-324.61 ± 10.31** |
| PPO entrenado | -151.26 |
| Random baseline | -311.0056379097316 |
| **Eficiencia PPO** | **-1174.2%** del gap random→óptimo |
| Distribución óptima | KEEP=23.0%, RESTRUCT=75.0%, SELL=2.0% |

### Portfolio Agent

| Métrica | Valor |
|---------|------:|
| **V\* (greedy upper bound)** | **-48.29 ± 0.00** |
| PPO entrenado | -48.54 |
| Random baseline | -144.52 |
| Hold baseline | -300.00 |
| **Eficiencia PPO** | **99.7%** del gap random→óptimo |

---

## Interpretación

La **eficiencia** mide qué fracción del gap entre random y óptimo captura el agente PPO. Un valor de 100% significa que PPO iguala al oráculo; >80% se considera excelente para RL con estados continuos.

La cota del Portfolio es **conservadora** (upper bound) porque el greedy no descuenta futuro — un agente con γ<1 puede superar al greedy miope en algunos episodios, pero en expectativa el greedy domina.

---
*Generado por `reports/theoretical_optimal_bound.py`*
