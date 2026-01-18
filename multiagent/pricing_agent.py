# ============================================
# multiagent/pricing_agent.py — Banco L1.5
# Wrapper para simulate_npl_price()
# ============================================

from __future__ import annotations
from typing import Dict, Any, Optional
import logging
import numpy as np

import config as cfg
from optimizer.price_simulator import simulate_npl_price

logger = logging.getLogger("PricingAgent")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class PricingAgent:
    """
    Micro-agente de pricing (mercado NPL).
    Envuelve simulate_npl_price() y devuelve un dict homogéneo:

        {
            "precio_optimo":,
            "pnl":,
            "capital_liberado":,
            "p5":,
            "p50":,
            "p95":,
            "ok": True/False
        }

    Coherente con el bloque VENDER de policy_inference.py.
    """

    def __init__(self) -> None:
        self.cfg_price = cfg.CONFIG.precio_venta

    # ----------------------------------------------------------
    #                     🔧 Utilidades
    # ----------------------------------------------------------
    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    # ----------------------------------------------------------
    #                    🔥 Acción principal
    # ----------------------------------------------------------
    def price(self, loan_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Devuelve pricing NPL coherente con policy_inference.
        """
        try:
            # ================================
            # 1) Construcción de argumentos
            # ================================
            sim_args: Dict[str, Any] = {}

            # EAD obligatorio
            ead = self._safe_float(loan_dict.get("EAD"), default=0.0)
            if ead <= 0.0:
                # nada que valorar
                return None
            sim_args["ead"] = ead

            # LGD (fallback razonable 0.45)
            sim_args["lgd"] = self._safe_float(loan_dict.get("LGD", 0.45), default=0.45)

            # RW si existe
            if "RW" in loan_dict:
                sim_args["rw"] = self._safe_float(loan_dict["RW"], default=1.0)

            # PD / DPD compat (idéntico espíritu policy_inference.py)
            if "DPD" in loan_dict:
                sim_args["dpd"] = self._safe_float(loan_dict["DPD"], default=120.0)
            elif "dpd" in loan_dict:
                sim_args["dpd"] = self._safe_float(loan_dict["dpd"], default=120.0)
            elif "PD" in loan_dict:
                sim_args["pd"] = self._safe_float(loan_dict["PD"], default=0.15)
            elif "pd" in loan_dict:
                sim_args["pd"] = self._safe_float(loan_dict["pd"], default=0.15)
            else:
                sim_args["dpd"] = 120.0  # default NPL clásico

            # Compatibilidad final pd/dpd
            if "pd" in sim_args and "dpd" not in sim_args:
                sim_args["dpd"] = sim_args["pd"]
            if "dpd" in sim_args and "pd" not in sim_args:
                sim_args["pd"] = sim_args["dpd"]

            # Segmento
            segment = loan_dict.get("segment") or loan_dict.get("SEGMENT") or "CORPORATE"
            sim_args["segment"] = str(segment)

            # Secured
            sim_args["secured"] = bool(loan_dict.get("secured", False))

            # ================================
            # 2) Llamada al simulador
            # ================================
            price = simulate_npl_price(**sim_args)

            if price is None:
                return None

            # ================================
            # 3) Normalización de salida
            # ================================
            resumen = price.get("resumen", {}) or {}

            precio_optimo = self._safe_float(price.get("precio_optimo"), default=0.0)
            pnl = self._safe_float(price.get("pnl"), default=0.0)
            capital_liberado = self._safe_float(
                price.get("capital_liberado"),
                default=ead * cfg.CONFIG.regulacion.required_total_capital_ratio(),
            )

            out = {
                "precio_optimo": precio_optimo,
                "pnl": pnl,
                "capital_liberado": capital_liberado,
                "p5": self._safe_float(resumen.get("p5"), default=0.0) if resumen else None,
                "p50": self._safe_float(resumen.get("p50"), default=0.0) if resumen else None,
                "p95": self._safe_float(resumen.get("p95"), default=0.0) if resumen else None,
                # ok: o bien viene del simulador, o lo inferimos como "precio razonable"
                "ok": bool(price.get("ok", precio_optimo > 0)),
            }

            return out

        except Exception as e:
            logger.warning(f"⚠ Error en PricingAgent.price(): {e}")
            return None
