# -*- coding: utf-8 -*-
# optimizer/guardrails.py
# Modulo de restricciones duras (Hard Constraints) para acciones de gestion de cartera.
# Implementa logica determinista "Bank-Ready" para validar REESTRUCTURACIONES y VENTAS.

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from risk.gates import check_restruct_viability, check_sell_fire_sale

logger = logging.getLogger("guardrails")

def check_restructure_constraints(loan_state: Dict[str, Any], cfg: Any = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida si una reestructuracion es viable segun politicas de riesgo (PTI, DSCR, mejora EVA).
    
    Args:
        loan_state: Diccionario con estado del prestamo (PTI, DSCR, EVA, etc.)
        cfg: Objeto de configuracion (opcional) con umbrales.
    
    Returns:
        (ok, reasons, metrics)
    """
    reasons = []
    metrics = {}
    
    # Extraer metricas clave
    pti = float(loan_state.get("pti_actual", 0.0) or 0.0)
    dscr = float(loan_state.get("dscr_actual", 0.0) or 0.0)
    
    # 1. Validacion de Viabilidad (PTI/DSCR) usando risk.gates
    # Se pasa un cfg dummy si es necesario o se usan defaults dentro de gates
    
    # Recuperar umbrales del cfg si existen
    risk_params = getattr(cfg, "risk_params", cfg) if cfg else None
    
    # Extraer limites con defaults seguros
    if risk_params:
         pti_max = float(getattr(risk_params, "pti_limit", 0.45) or 0.45)
         dscr_min = float(getattr(risk_params, "dscr_min", 1.10) or 1.10)
    else:
         pti_max = 0.45
         dscr_min = 1.10

    metrics["pti"] = pti
    metrics["dscr"] = dscr
    metrics["threshold_pti"] = pti_max
    metrics["threshold_dscr"] = dscr_min

    # Usar check_restruct_viability de risk.gates
    # Firma: check_restruct_viability(current_income, new_payment, dscr_min)
    # Pero aqui tenemos 'pti' y 'dscr' ya calculados en 'loan_state'
    # Si tenemos income/payment, podriamos recalcular, pero confiamos en lo que venga en loan_state
    
    # Check PTI (manual pues no esta en check_restruct_viability explicito como gate unico)
    if pti > pti_max:
        reasons.append(f"PTI_HIGH ({pti:.2f} > {pti_max:.2f})")
    
    # Check DSCR (usando helper si queremos o directo)
    # Si usamos check_restruct_viability necesitamos income y payment
    income = float(loan_state.get("income", 0.0) or 0.0)
    payment = float(loan_state.get("payment", 0.0) or 0.0)
    
    is_dscr_ok, dscr_val, dscr_reason = check_restruct_viability(income, payment, dscr_min)
    
    # Si no hay income/payment, usamos el dscr pre-calculado del state si existe
    if dscr_reason == "DSCR_INPUT_MISSING":
         if dscr < dscr_min and dscr > 0:
              reasons.append(f"DSCR_LOW ({dscr:.2f} < {dscr_min:.2f})")
         elif dscr <= 0:
               # Si no hay datos, es blocking bank-ready?
               # Depende policy. Asumimos que si falta data, no es viable.
               reasons.append("DSCR_MISSING_DATA")
    elif not is_dscr_ok:
         reasons.append(f"{dscr_reason} ({dscr_val:.2f} < {dscr_min:.2f})")

    # 2. Check Mejora Economica (EVA / PD / RW)
    # Esto es mas sutil, requiere comparar PRE vs POST.
    # Si no tenemos EVA_post, asumimos que el modelo lo predijo positivo, 
    # pero podemos chequear si 'eva_delta' esta en loan_state o similar.
    
    eva_pre = float(loan_state.get("eva_pre", 0.0) or 0.0)
    eva_post = float(loan_state.get("eva_post", 0.0) or 0.0) # Esperado
    
    if "eva_post" in loan_state:
        eva_delta = eva_post - eva_pre
        min_improvement = 0.0 # Debe mejorar algo
        metrics["eva_delta"] = eva_delta
        if eva_delta < min_improvement:
             reasons.append(f"EVA_NO_IMPROVEMENT ({eva_delta:,.0f} < 0)")

    ok = len(reasons) == 0
    return ok, reasons, metrics


def check_sell_constraints(loan_state: Dict[str, Any], pricing_out: Dict[str, Any], cfg: Any = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida si una venta es ejecutable (No es Fire Sale destructivo, Precio minimo).
    
    Args:
        loan_state: Estado del prestamo (Book, EAD, RW).
        pricing_out: Output del simulador de precios (Price, Costs, FireSaleMetrics).
        cfg: Configuracion con umbrales de apetito de riesgo.
    
    Returns:
        (ok, reasons, metrics)
    """
    reasons = []
    metrics = {}
    
    # Extraer datos de precio
    price = pricing_out.get("price", 0.0)
    costs = pricing_out.get("costs", 0.0) # Si simulate_npl_price devuelve costes
    net_price = pricing_out.get("net_price", price) # Asumir neto si no hay explicito
    if "price_neto" in pricing_out:
        net_price = pricing_out["price_neto"]

    book_value = float(loan_state.get("book_value", 0.0) or 0.0)
    ead = float(loan_state.get("ead", 0.0) or 0.0)
    
    pnl = net_price - book_value
    
    # Calculo de RWA y Capital
    rw = float(loan_state.get("rw", 1.5)) # Default 150%
    if rw > 10.0: rw /= 100.0
    
    # Capital Release = EAD * RW * 8% (o ratio capital)
    cap_ratio = 0.08 # Default
    if cfg and hasattr(cfg, "capital_ratio"):
        cap_ratio = float(cfg.capital_ratio)
        
    rwa_before = ead * rw
    rwa_after = 0.0 # Venta elimina RWA
    capital_release = rwa_before * cap_ratio
    
    metrics["price"] = price
    metrics["net_price"] = net_price
    metrics["costs"] = costs
    metrics["pnl"] = pnl
    metrics["capital_release"] = capital_release
    metrics["rwa_before"] = rwa_before
    
    # 1. Validación Fire Sale (Price < Threshold Book)
    # Reutilizamos la logica de gates o implementamos explicita
    
    # Recuperar umbral fire sale
    fire_sale_thr = 0.20 # Default
    allow_fs = False     # Default prudencial
    
    if cfg:
         fire_sale_options = getattr(cfg, "fire_sale", {})
         if isinstance(fire_sale_options, dict):
             fire_sale_thr = float(fire_sale_options.get("threshold_book", 0.20))
             allow_fs = bool(fire_sale_options.get("allow_fire_sale", False))
         elif hasattr(fire_sale_options, "threshold_book"):
             fire_sale_thr = float(fire_sale_options.threshold_book)
             allow_fs = getattr(fire_sale_options, "allow_fire_sale", False)
             
    # Usar check_sell_fire_sale de risk.gates
    # Firma: check_sell_fire_sale(price_neto, book_value, allow_fire_sale, thr_book)
    # Retorna: (allowed, ratio, is_fire_sale, reason)
    
    allowed, ratio, is_fs, reason = check_sell_fire_sale(net_price, book_value, allow_fs, fire_sale_thr)
    metrics["price_book_ratio"] = ratio
    metrics["fire_sale"] = is_fs
    
    if not allowed:
        reasons.append(f"FIRE_SALE_BLOCK ({reason})")
    elif is_fs:
        # Es fire sale pero esta permitido. Lo loguamos.
        pass

    # 2. Check Precio Absurdo (Guardrail de seguridad)
    if net_price <= 0:
         reasons.append("PRICE_NON_POSITIVE")
    
    # 3. Check PnL Excesivo (Doble check ademas de fire sale)
    # Ej: No perder mas del 50% del book value
    if metrics["price_book_ratio"] < 0.50:
         reasons.append(f"DEEP_DISCOUNT (Price/Book < 50%)")

    ok = len(reasons) == 0
    return ok, reasons, metrics
