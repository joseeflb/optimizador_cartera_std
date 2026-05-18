# -*- coding: utf-8 -*-
# ============================================================
# baselines/baseline_policies.py
# Autor: José María Fernández-Ladreda Ballvé
# Resumen: Políticas heurísticas deterministas (Prudencial / Desinversión) usadas como baseline frente al RL.
# ============================================================

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

# Imports del proyecto
from optimizer.guardrails import check_restructure_constraints, check_sell_constraints
import config as cfg

logger = logging.getLogger("baseline_policies")

# Definicion de acciones (debe coincidir con LoanEnv)
ACTION_KEEP = 0
ACTION_SELL = 1
ACTION_RESTRUCT = 2

class BaselinePolicy:
    """Clase base para políticas deterministas."""
    def __init__(self, name: str, risk_posture: str):
        self.name = name
        self.risk_posture = risk_posture
        self.stats = {
            "total": 0,
            "keep": 0,
            "sell": 0,
            "restruct": 0,
            "blocked_sell": 0,
            "blocked_restruct": 0
        }

    def predict(self, loan_row: pd.Series) -> int:
        """Devuelve la acción recomendada (0, 1, 2) para el préstamo."""
        raise NotImplementedError

    def _update_stats(self, action: int):
        self.stats["total"] += 1
        if action == ACTION_KEEP:
            self.stats["keep"] += 1
        elif action == ACTION_SELL:
            self.stats["sell"] += 1
        elif action == ACTION_RESTRUCT:
            self.stats["restruct"] += 1

    def _extract_pricing_info(self, row: pd.Series) -> Dict[str, Any]:
        """Extrae info de pricing del row para validar venta. Asume nombres de col comunes."""
        # Se asume que el row viene enriquecido (ej. 'price_to_ead', 'capital_release_sim')
        
        # Helper case insensitive get
        def get_val(keys, default=0.0):
            for k in keys:
                if k in row:
                    return float(row[k])
            return default

        ead = get_val(["EAD", "ead"], 1.0)
        price_ead = get_val(["Price_to_EAD", "price_to_ead", "price_ratio_ead"], 0.0)
        
        # Si precio_optimo existe, usarlo directo, sino calcular
        bid_price = get_val(["precio_optimo", "bid_price"], price_ead * ead)
        
        # PnL aprox (bid - net_book_value)
        nbv = get_val(["net_book_value", "book_value", "valor_referencia"], ead)
        pnl = get_val(["pnl", "PnL", "pnl_book", "pnl_realized"], bid_price - nbv)
        
        # Capital Release Simulado
        cap_release = get_val(["capital_liberado", "capital_release_sim", "capital_release_net", "capital_release"], 0.0)
        
        return {
            "precio_optimo": bid_price,
            "pnl": pnl,
            "capital_liberado": cap_release,
            "coste_tx": 0.0, # Asumir 0 o pequeno
            "rw": 1.5, # Default estandar NPL
            "book_value": nbv,
            "fire_sale": False # Asumir false por default si no hay info
        }

class BaselinePrudencialRules(BaselinePolicy):
    """
    Baseline_PRUDENCIAL_RULES:
    - Conservador. 
    - REESTRUCTURAR si viable y riesgo alto (PD/LGD).
    - VENDER solo si precio excelente (evita fire-sales).
    - MANTENER por defecto si no es viable otra cosa.
    """
    def __init__(self):
        super().__init__("Baseline_PRUDENCIAL", "prudencial")
        # Umbrales especificos de esta heuristica
        self.min_price_ead = 0.50  # Solo vende si recupera > 50% EAD (muy exigente)
        self.high_pd_threshold = 0.10 # 10% PD -> Alto riesgo
        self.high_lgd_threshold = 0.45 # 45% LGD -> Alto riesgo

    def predict(self, loan_row: pd.Series) -> int:
        # Extraer datos relevantes
        def get_val(keys, default=0.0):
            for k in keys:
                if k in loan_row:
                    return float(loan_row[k])
            return default

        price_ead = get_val(["Price_to_EAD", "price_to_ead", "price_ratio_ead"], 0.0)
        pd_val = get_val(["PD_12m", "PD", "pd"], 0.0)
        lgd_val = get_val(["LGD_12m", "LGD", "lgd"], 0.0)
        
        # Validar viabilidad Restruct (DSCR/PTI)
        is_restruct_viable = self._check_restruct_viability(loan_row)
        
        # Logica Prudencial
        
        # 1. Chequeo de Venta (muy restrictivo)
        # Solo vende si el precio es MUY bueno (evita perdidas)
        if price_ead > self.min_price_ead:
             # Verificar guardrail de venta (fire sale check)
             # Necesitamos pricing_out mockeado o real del row
             # Si el row tiene info de simulacion de precio, la usamos.
             pricing_out = self._extract_pricing_info(loan_row)
             
             can_sell, reasons, _ = check_sell_constraints(loan_row.to_dict(), pricing_out, config_obj=None)
             if can_sell:
                 self._update_stats(ACTION_SELL)
                 return ACTION_SELL
             else:
                 self.stats["blocked_sell"] += 1

        # 2. Chequeo de Reestructuracion (preferido si riesgo alto y viable)
        if (pd_val > self.high_pd_threshold or lgd_val > self.high_lgd_threshold) and is_restruct_viable:
             # Verificar guardrai de restruct
             can_restruct, reasons, _ = check_restructure_constraints(loan_row.to_dict())
             if can_restruct:
                 self._update_stats(ACTION_RESTRUCT)
                 return ACTION_RESTRUCT
             else:
                 self.stats["blocked_restruct"] += 1
        
        # 3. Default: Mantener
        self._update_stats(ACTION_KEEP)
        return ACTION_KEEP

    def _check_restruct_viability(self, row):
        # Usar flag pre-calculado del entorno si existe (restruct_viable significa que existe una solucion valida)
        if "restruct_viable" in row:
            return bool(row["restruct_viable"])
        
        # Fallback a chequeo simple (solo viable si current cumple, lo cual casi nunca pasa en default)
        pti = float(row.get("pti_actual", 0.0) or 0.0)
        dscr = float(row.get("dscr_actual", 0.0) or 0.0)
        
        # Guardrails duros
        if dscr < cfg.GR_DSCR_MIN: return False
        if pti > cfg.GR_PTI_MAX: return False
        return True


class BaselineDesinversionRules(BaselinePolicy):
    """
    Baseline_DESINVERSION_RULES:
    - Agresivo.
    - VENDER si libera capital y precio razonable (> umbral bajo).
    - REESTRUCTURAR si no se puede vender pero es viable.
    - MANTENER solo si no hay opcion.
    """
    def __init__(self):
        super().__init__("Baseline_DESINVERSION", "desinversion")
        # Umbrales especificos
        self.min_price_ead = 0.15 # Vende incluso con descuento fuerte si libera capital
        self.min_capital_release = 0.0 # Cualquier liberacion positiva

    def predict(self, loan_row: pd.Series) -> int:
        def get_val(keys, default=0.0):
            for k in keys:
                if k in loan_row:
                    return float(loan_row[k])
            return default

        price_ead = get_val(["Price_to_EAD", "price_to_ead", "price_ratio_ead"], 0.0)
        cap_release = get_val(["capital_liberado", "capital_release_sim", "capital_release_net", "capital_release"], 0.0)
        # Si no existe capital_release pre-calculado, usar RWA pre - post aprox?
        # Asumimos que el row trae info del environment o pre-calculada.
        
        # Logica Desinversion
        
        # 1. Prioridad: VENDER
        if price_ead > self.min_price_ead:
             # Verificar guardrail "desinversion" (mas laxo)
             pricing_out = self._extract_pricing_info(loan_row)
             
             can_sell, reasons, _ = check_sell_constraints(loan_row.to_dict(), pricing_out, config_obj=None)
             if can_sell:
                 self._update_stats(ACTION_SELL)
                 return ACTION_SELL
             else:
                 self.stats["blocked_sell"] += 1

        # 2. Secundaria: REESTRUCTURAR (para mejorar perfil y vender luego?)
        # Solo si viable
        can_restruct, reasons, _ = check_restructure_constraints(loan_row.to_dict())
        if can_restruct:
             self._update_stats(ACTION_RESTRUCT)
             return ACTION_RESTRUCT
        else:
             self.stats["blocked_restruct"] += 1

        # 3. Default: Mantener
        self._update_stats(ACTION_KEEP)
        return ACTION_KEEP

