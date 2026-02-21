# PC10 — Semántica de EVA_post bajo venta y nuevos KPIs de pricing

## Hallazgo: EVA_post NO incorpora PnL de venta

Código de referencia: `agent/policy_inference.py` línea ~1157.

```python
decision["EVA_post"] = 0.0   # VENDER: loan sale del portfolio → EVA=0
```

Cuando la acción final es **VENDER**, `EVA_post` se fija en **0.0** porque el
préstamo abandona el balance y deja de generar EVA. Es un valor de estado
post-acción, **no** el impacto económico de la venta.

El P&L realizado queda en campos separados:
- `pnl` / `pnl_realized` — diferencia precio_bid − book_value (signo negativo típico)
- `pnl_book` — alias explícito
- `pnl_ratio_book` — pnl / book_value (ratio; <0 implica descuento)
- `audit_price_book_ratio` — precio / book_value (≈ 1 + pnl_ratio_book)

## Consecuencia en pricing_crunch

En el escenario `pricing_crunch` (haircut en bids), `EVA_post` es siempre 0 para
las ventas independientemente del haircut: **el stress económico es invisible** en
`total_eva_post` agregado, ya que el EVA_pre de esos loans simplemente desaparece.

El impacto sí es observable vía: precio medio / EAD (↓), PnL medio (↓↓),
n_sales posiblemente ↓ (guardrail fire-sale bloquea las ventas más agresivas).

## KPIs nuevos añadidos en PC10 (stress_summary)

| Columna              | Fuente                         | Notas                              |
|----------------------|--------------------------------|------------------------------------|
| `sale_pnl_total`     | sum(`pnl_realized`) para VENDER | Negativo = pérdida neta             |
| `avg_sale_pnl`       | mean(`pnl_realized`) VENDER     | € medio por venta                  |
| `avg_bid_pct_ead`    | mean(`audit_price_book_ratio`)  | Precio/book como proxy bid%        |
| `sell_blocked_count` | sum(`Sell_Blocked`)             | Ventas bloqueadas por guardrail    |
