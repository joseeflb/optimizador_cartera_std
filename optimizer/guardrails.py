# -*- coding: utf-8 -*-
# ============================================================
# optimizer/guardrails.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Restricciones duras (hard constraints) bank-ready para REESTRUCTURAR y VENDER (PTI, DSCR, fire-sale, mejora EVA).
# ============================================================

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from risk.gates import check_restruct_viability, check_sell_fire_sale
import config as cfg  # Importar config global

logger = logging.getLogger("guardrails")

def check_restructure_constraints(loan_state: Dict[str, Any], config_obj: Any = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida si una reestructuracion es viable segun politicas de riesgo (PTI, DSCR, mejora EVA).
    
    Args:
        loan_state: Diccionario con estado del prestamo (PTI, DSCR, EVA, etc.)
        config_obj: Objeto de configuracion (opcional) con umbrales.
    
    Returns:
        (ok, reasons, metrics)
    """
    reasons = []
    metrics = {}
    
    # 1. Recuperar umbrales desde config o fallback seguro
    # Prioridad: config_obj -> config.py global (GR_*) -> Hard Defaults
    if config_obj and hasattr(config_obj, "risk_params"):
         # Si viene un objeto custom (ej. training)
         rp = config_obj.risk_params
         pti_max = float(getattr(rp, "pti_limit", getattr(cfg, "GR_PTI_MAX", 0.45)))
         dscr_min = float(getattr(rp, "dscr_min", getattr(cfg, "GR_DSCR_MIN", 1.10)))
    else:
         # Usar globales de config.py
         pti_max = getattr(cfg, "GR_PTI_MAX", 0.45)
         dscr_min = getattr(cfg, "GR_DSCR_MIN", 1.10)

    # 2. Extraer metricas del estado
    pti = float(loan_state.get("pti_actual", 0.0) or 0.0)
    dscr = float(loan_state.get("dscr_actual", 0.0) or 0.0)

    # Calculo de headroom (margen de maniobra)
    pti_headroom = pti_max - pti
    dscr_headroom = dscr - dscr_min

    metrics.update({
        "pti_actual": pti,
        "dscr_actual": dscr,
        "pti_max": pti_max,
        "dscr_min": dscr_min,
        "pti_headroom": pti_headroom, # Positivo es bueno
        "dscr_headroom": dscr_headroom, # Positivo es bueno
        "restructure_ok": True # Provisional, se actualiza si falla
    })

    # 3. Validaciones
    
    # Check PTI
    if pti > pti_max:
        reasons.append(f"PTI_HIGH ({pti:.2f} > {pti_max:.2f})")
    
    # Check DSCR
    # Nota: check_restruct_viability de risk.gates pide income/payment.
    # Si tenemos dscr pre-calculado fiable, lo usamos directamenet para consistencia con el modelo.
    if dscr < dscr_min:
        # Validar si es 0 (missing data) o bajo real
        if dscr <= 0.01:
             reasons.append("DSCR_MISSING/ZERO")
        else:
             reasons.append(f"DSCR_LOW ({dscr:.2f} < {dscr_min:.2f})")

    # 4. Check Mejora Economica (EVA / PD / RW)
    eva_pre = float(loan_state.get("eva_pre", 0.0) or 0.0)
    eva_post = float(loan_state.get("eva_post", 0.0) or 0.0)
    
    if "eva_post" in loan_state:
        eva_delta = eva_post - eva_pre
        metrics["eva_delta"] = eva_delta
        # No bloqueamos hard si no mejora, a menos que sea politica estricta.
        # Por ahora solo PTI/DSCR son hard blocks regulatorios/bancarios.

    ok = len(reasons) == 0
    metrics["restructure_ok"] = ok
    return ok, reasons, metrics


def check_sell_constraints(loan_state: Dict[str, Any], pricing_out: Dict[str, Any], config_obj: Any = None) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida si una venta es ejecutable (pricing_out viene de simulate_npl_price).
    Reglas Bank-Ready:
      - Oferta minima (% EAD)
      - Perdida maxima (PnL vs Book o vs EAD)
      - Liberacion de capital minima
      - Checks de integridad (missing keys)
    """
    reasons = []
    metrics = {}
    
    # 1. Recuperar Umbrales
    min_bid_pct_ead = getattr(cfg, "GR_SELL_MIN_BID_PCT_EAD", 0.05)
    max_loss_pct_ead_pru = getattr(cfg, "GR_SELL_MAX_FIRE_SALE_LOSS_PCT_EAD_PRUDENCIAL", 0.40)
    min_cap_release = getattr(cfg, "GR_SELL_MIN_CAPITAL_RELEASE", 0.0)

    # 2. Validar Integridad Pricing
    required_keys = ["precio_optimo", "pnl", "capital_liberado"]
    missing = [k for k in required_keys if k not in pricing_out]
    if missing:
        return False, [f"MISSING_PRICING_OUTPUTS ({missing})"], {"sell_ok": False}

    # 3. Extraer Metricas
    ead = float(loan_state.get("ead", 0.0) or 0.0)
    if ead <= 0: ead = float(loan_state.get("EAD", 1.0)) # Fallback seguro
    
    bid_price = float(pricing_out.get("precio_optimo", 0.0))
    pnl = float(pricing_out.get("pnl", 0.0)) # Net Price - Book Value
    capital_release = float(pricing_out.get("capital_liberado", 0.0))
    coste_tx = float(pricing_out.get("coste_tx", 0.0))
    
    # Book value (intentar sacar de pricing_out primero, luego loan_state)
    book_val = float(pricing_out.get("book_value", 0.0))
    if book_val <= 0: book_val = float(loan_state.get("book_value", 0.0) or 0.0)

    # RWA (si viene de pricing_out, mejor)
    rw = float(pricing_out.get("rw", 1.5))
    rwa_before = ead * rw
    rwa_after = 0.0 # Venta = 0 RWA

    metrics.update({
        "bid_price": bid_price,
        "costs": coste_tx,
        "pnl": pnl,
        "capital_release": capital_release,
        "rwa_before": rwa_before,
        "rwa_after": rwa_after,
        "sell_ok": True # Default
    })
    
    # 4. Guardrail: Oferta Minima Absurda
    # Si ofrece menos del X% del EAD, es basura/estafa o error.
    if bid_price < (ead * min_bid_pct_ead):
        reasons.append(f"BID_TOO_LOW (<{min_bid_pct_ead*100:.0f}% EAD)")

    # 5. Guardrail: Perdida Excesiva (posture-aware)
    # La perdida maxima aceptable depende de la postura de riesgo.
    loss_absolute = -pnl if pnl < 0 else 0.0

    # Derive posture-specific threshold from config_obj if available
    actual_max_loss_pct = max_loss_pct_ead_pru  # default 40%
    if config_obj and hasattr(config_obj, "risk_params"):
        rp = config_obj.risk_params
        if hasattr(rp, "pnl_max_loss_pct_ead"):
            actual_max_loss_pct = float(rp.pnl_max_loss_pct_ead)
    # Provide per-posture defaults via config dict fallback
    if config_obj and hasattr(config_obj, "_posture"):
        posture = str(config_obj._posture).lower()
        posture_defaults = {"prudencial": 0.40, "balanceado": 0.60, "desinversion": 0.85}
        actual_max_loss_pct = posture_defaults.get(posture, actual_max_loss_pct)

    if loss_absolute > (ead * actual_max_loss_pct):
        reasons.append(f"PNL_TOO_NEGATIVE (Loss > {actual_max_loss_pct*100:.0f}% EAD)")

    # 6. Guardrail: Capital Release
    # No vender si no libera capital (ej. RWA ya era 0 o pnl muy negativo come capital)
    # Nota: capital_release ya deberia ser neto? Normalmente es RWA*8%.
    # Si el PnL negativo impacta capital CET1, deberiamos considerarlo.
    # La metrica 'capital_liberado' de price_simulator suele ser bruta RWA*8%.
    if capital_release < min_cap_release:
        reasons.append("CAPITAL_RELEASE_TOO_LOW")
        
    # 7. Integracion Check Fire Sale (risk.gates)
    # Si price_simulator ya devolvio fire_sale=True y reason, podemos usarlos.
    is_fire_sale = pricing_out.get("fire_sale", False)
    fs_reason = pricing_out.get("fire_sale_reason", "")
    
    # Si price_simulator dice que es Fire Sale prohibido (aunque aqui solo nos da el flag)
    # price_simulator ya hace check de gate internos, pero nos devuelve fire_sale=True si
    # viola el threshold book.
    # Si nuestro mandato es "NO FIRE SALES", bloqueamos.
    # Asumimos que si estamos aqui, queremos validar extra.
    # Si la config pide 'allow_fire_sale', esto seria OK.
    # Validemos contra config real si existe.
    allow_fs = False
    if config_obj and hasattr(config_obj, "fire_sale"):
         allow_fs = getattr(config_obj.fire_sale, "allow_fire_sale", False)
    
    if is_fire_sale and not allow_fs:
         reasons.append(f"FIRE_SALE_BLOCK ({fs_reason})")

    ok = len(reasons) == 0
    metrics["sell_ok"] = ok
    
    return ok, reasons, metrics
