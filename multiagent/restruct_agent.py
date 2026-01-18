# ============================================
# multiagent/restruct_agent.py — Banco L1.5
# Wrapper para optimize_restructure()
# ============================================

from __future__ import annotations
from typing import Dict, Any, Optional
import logging

import numpy as np

import config as cfg
from optimizer.restructure_optimizer import optimize_restructure

logger = logging.getLogger("RestructAgent")
logger.setLevel(logging.INFO)


class RestructAgent:
    """
    Micro-agente de reestructuración.
    Envuelve optimize_restructure() y lo hace robusto:
        - Normaliza inputs (EAD, rate, PD, LGD, RW...)
        - Maneja excepciones
        - Asegura formato homogéneo para CoordinatorAgent

    No decide la política (eso lo hace policy_inference / CoordinatorAgent):
    sólo propone la mejor reestructuración Banco L1.5 para un préstamo.
    """

    def __init__(self):
        self.hurdle = cfg.CONFIG.regulacion.hurdle_rate

    # ----------------------------------------------------------
    #                  🧼 Utilidades internas
    # ----------------------------------------------------------
    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            v = float(x)
            if np.isnan(v):
                return float(default)
            return v
        except Exception:
            return float(default)

    # ----------------------------------------------------------
    #                  🔥 Acción principal
    # ----------------------------------------------------------
    def propose(self, loan_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Recibe un préstamo (dict tipo LoanEnv / policy_inference):

            {
                "EAD": ...,
                "rate": ...,
                "PD": ...,
                "LGD": ...,
                "RW": ...,
                "ingreso_mensual": ...,
                "cashflow_operativo_mensual": ...,
                "EVA": ...,
                ...
            }

        Devuelve dict con (salida directa de optimize_restructure + ajustes):

            {
                "EVA_post":,
                "EVA_gain":,
                "RWA_post":,
                "RORWA_post":,
                "PTI":,
                "DSCR":,
                "plazo_optimo":,
                "tasa_nueva":,
                "quita":,
                "EL_avg_ifrs9":,
                "EL_lifetime":,
                "cured":,
                "ok": True/False,
                "msg": "...",
                ...
            }
        """
        try:
            # -----------------------------
            # 1) Normalizar inputs clave
            # -----------------------------
            # Compat EAD / ead
            ead = self._safe_float(
                loan_dict.get("EAD", loan_dict.get("ead", 0.0)),
                default=0.0,
            )
            if ead <= 0.0:
                return None

            # Tasa nominal (rate / RATE / tipo)
            rate = self._safe_float(
                loan_dict.get("rate", loan_dict.get("RATE", 0.06)),
                default=0.06,
            )

            # PD / pd
            pd = self._safe_float(
                loan_dict.get("PD", loan_dict.get("pd", 0.05)),
                default=0.05,
            )

            # LGD / lgd
            lgd = self._safe_float(
                loan_dict.get("LGD", loan_dict.get("lgd", 0.45)),
                default=0.45,
            )

            # RW (si no viene, asumimos 1.0 como RW STD default)
            rw = self._safe_float(
                loan_dict.get("RW", loan_dict.get("rw", 1.0)),
                default=1.0,
            )

            ingreso = loan_dict.get("ingreso_mensual", None)
            cfo = loan_dict.get("cashflow_operativo_mensual", None)

            # EVA base para calcular ΔEVA si el optimizador no lo hace
            eva_base = self._safe_float(loan_dict.get("EVA", 0.0), default=0.0)

            # -----------------------------
            # 2) Llamar al optimizador L1.5 real
            # -----------------------------
            out = optimize_restructure(
                ead=ead,
                rate=rate,
                pd=pd,
                lgd=lgd,
                rw=rw,
                ingreso_mensual=ingreso,
                cashflow_operativo_mensual=cfo,
                hurdle=self.hurdle,
            )

            # -----------------------------
            # 3) Normalizar salida
            # -----------------------------
            eva_post = self._safe_float(out.get("EVA_post", eva_base), default=eva_base)
            eva_gain = self._safe_float(out.get("EVA_gain", eva_post - eva_base), default=eva_post - eva_base)

            out.setdefault("EVA_post", eva_post)
            out.setdefault("EVA_gain", eva_gain)

            # ok = True si mejora EVA
            if "ok" not in out:
                out["ok"] = eva_post > eva_base

            # Mensaje amigable si falta
            if "msg" not in out:
                out["msg"] = "OK (mejora significativa)" if out["ok"] else "sin mejora"

            return out

        except Exception as e:
            logger.warning(f"⚠ Error en RestructAgent.propose(): {e}")
            return None
