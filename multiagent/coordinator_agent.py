# ============================================
# multiagent/coordinator_agent.py — Banco L1.5
# Coordinador jerárquico (macro ↔ micro)
# ============================================

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import logging

import numpy as np

import config as cfg

logger = logging.getLogger("CoordinatorAgent")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class CoordinatorAgent:
    """
    Agente principal que coordina:
      • PPO macro (PortfolioEnv)
      • Reestructuración micro (restruct_agent.propose → optimize_restructure)
      • Pricing micro (pricing_agent.price → simulate_npl_price)
      • StressEngine (opcional)
      • SensitivityEngine (opcional)

    Lógica jerárquica Banco L1.5:
      1. Regla prudencial (EVA / RORWA / PTI / DSCR / riesgo extremo)
      2. Sugerencias micro (ΔEVA reestructura, precio NPL)
      3. PPO macro como guía de volumen / timing, nunca contra prudencia.
    """

    def __init__(
        self,
        ppo_model,
        restruct_agent,
        pricing_agent,
        stress_engine: Optional[Any] = None,
        sensitivity_engine: Optional[Any] = None,
        deterministic: bool = True,
    ):
        self.model = ppo_model
        self.restruct_agent = restruct_agent
        self.pricing_agent = pricing_agent
        self.stress_engine = stress_engine
        self.sensitivity_engine = sensitivity_engine
        self.deterministic = deterministic

        # Parámetros regulatorios Banco L1.5
        self.hurdle = cfg.CONFIG.regulacion.hurdle_rate
        self.cap_ratio = cfg.CONFIG.regulacion.required_total_capital_ratio()

        # Umbrales prudenciales coherentes con policy_inference.py
        self.EVA_STRONGLY_NEG_EUR = -50_000.0
        self.EVA_MIN_IMPROVEMENT_EUR = 10_000.0
        self.EVA_MIN_IMPROVEMENT_PCT = 0.05
        self.PTI_MAX = cfg.CONFIG.reestructura.esfuerzo_umbral_bajo  # ~0.30
        self.PTI_CRITICO = cfg.CONFIG.reestructura.esfuerzo_umbral_alto  # ~0.50
        self.DSCR_MIN = 1.10
        self.PD_ALTO = 0.20
        self.LGD_ALTO = 0.50

        # Mapa de acciones macro PortfolioEnv
        self.ACTION_MAP: Dict[int, str] = {
            0: "MANTENER todos",
            1: "VENDER top-1 EVA negativa",
            2: "VENDER top-K EVA negativa",
            3: "REESTRUCTURAR top-1 EVA negativa",
            4: "REESTRUCTURAR top-K EVA negativa",
            5: "VENDER top-1 RORWA más bajo",
            6: "VENDER top-K RORWA más bajo",
            7: "REESTRUCTURAR top-1 PTI más alto",
            8: "REESTRUCTURAR top-K PTI más alto",
            9: "MIX vender & reestructurar",
            10: "Regla heurística Banco L1.5",
            11: "NO-OP",
        }

    # ----------------------------------------------------------
    #                       PPO MACRO
    # ----------------------------------------------------------
    def act_macro(self, portfolio_obs: np.ndarray) -> Tuple[int, str]:
        """Acción sugerida por PPO entrenado sobre PortfolioEnv."""
        action, _ = self.model.predict(portfolio_obs, deterministic=self.deterministic)
        a = int(np.squeeze(action))
        return a, self.ACTION_MAP.get(a, f"ACCION_{a}")

    # ----------------------------------------------------------
    #                       MICRO-AGENTS
    # ----------------------------------------------------------
    def act_micro(self, loan_dict: Dict[str, Any]):
        """
        Llama a los subagentes micro:
          • restruct_agent.propose(loan_dict)  → dict Banco L1.5
          • pricing_agent.price(loan_dict)     → dict simulate_npl_price
        """
        restruct_suggestion = None
        pricing_suggestion = None

        # --- Reestructura ---
        try:
            if self.restruct_agent is not None:
                restruct_suggestion = self.restruct_agent.propose(loan_dict)
        except Exception as e:
            logger.warning(f"⚠ Error en micro_restruct: {e}")

        # --- Pricing ---
        try:
            if self.pricing_agent is not None:
                pricing_suggestion = self.pricing_agent.price(loan_dict)
        except Exception as e:
            logger.warning(f"⚠ Error en micro_pricing: {e}")

        return restruct_suggestion, pricing_suggestion

    # ----------------------------------------------------------
    #                 UTILIDADES INTERNAS
    # ----------------------------------------------------------
    def _safe_float(self, v: Any, default: float = np.nan) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _infer_pti_dscr(self, loan_dict: Dict[str, Any]) -> Tuple[float, float]:
        """
        Intenta inferir PTI y DSCR si no vienen explícitos:
          PTI ≈ cuota_mensual / ingreso_mensual
          DSCR ≈ cashflow_operativo_mensual / cuota_mensual
        """
        pti = loan_dict.get("PTI", None)
        dscr = loan_dict.get("DSCR", None)

        pti_f = self._safe_float(pti, default=np.nan)
        dscr_f = self._safe_float(dscr, default=np.nan)

        if np.isnan(pti_f):
            cuota = self._safe_float(loan_dict.get("cuota_mensual"), default=np.nan)
            ingreso = self._safe_float(loan_dict.get("ingreso_mensual"), default=np.nan)
            if not np.isnan(cuota) and cuota > 0 and not np.isnan(ingreso) and ingreso > 0:
                pti_f = cuota / ingreso

        if np.isnan(dscr_f):
            cuota = self._safe_float(loan_dict.get("cuota_mensual"), default=np.nan)
            cfo = self._safe_float(loan_dict.get("cashflow_operativo_mensual"), default=np.nan)
            if not np.isnan(cuota) and cuota > 0 and not np.isnan(cfo):
                dscr_f = cfo / cuota

        return pti_f, dscr_f

    # ----------------------------------------------------------
    #        🔥 FUSIÓN MACRO + MICRO + REGLA FINANCIERA
    # ----------------------------------------------------------
    def decide(self, portfolio_obs: np.ndarray, loan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Devuelve una decisión micro final para un préstamo concreto, combinando:

            1) PPO macro (acción de cartera)
            2) Reglas prudenciales Banco L1.5 (EVA, RORWA, PTI, DSCR, riesgo extremo)
            3) Sugerencias micro:
                - optimize_restructure (ΔEVA, PTI_post, DSCR_post, cured)
                - simulate_npl_price (precio_optimo, capital_liberado)
            4) SensitivityEngine (si está disponible) para enriquecer la explicación.

        Retorna dict:
            {
              "accion_macro": int,
              "accion_macro_desc": str,
              "micro_restruct": dict | None,
              "micro_price": dict | None,
              "accion_final": "MANTENER" / "REESTRUCTURAR" / "VENDER",
              "razon": [str, ...],
            }
        """

        razon: list[str] = []

        # ==========================================================
        # 1) PPO Macro
        # ==========================================================
        accion_macro_id, accion_macro_desc = self.act_macro(portfolio_obs)
        razon.append(f"PPO_macro sugiere: {accion_macro_desc}")

        # ==========================================================
        # 2) Micro (restruct + pricing)
        # ==========================================================
        restruct_sug, price_sug = self.act_micro(loan_dict)

        # Métricas de base del préstamo
        eva_pre = self._safe_float(loan_dict.get("EVA", 0.0), default=0.0)
        rorwa_pre = self._safe_float(loan_dict.get("RORWA", 0.0), default=0.0)
        pd_pre = self._safe_float(loan_dict.get("PD", 0.05), default=0.05)
        lgd_pre = self._safe_float(loan_dict.get("LGD", 0.40), default=0.40)

        pti_pre, dscr_pre = self._infer_pti_dscr(loan_dict)

        # Sensitividades (opcional)
        sens_score = None
        if self.sensitivity_engine is not None:
            try:
                sens_score = self.sensitivity_engine.global_sensitivity_score(loan_dict)
                razon.append(f"Sensibilidad global (palanca) ≈ {sens_score:,.0f}.")
            except Exception as e:
                logger.warning(f"⚠ Error en SensitivityEngine: {e}")

        # ==========================================================
        # 3) Reglas prudenciales Banco L1.5 (estado base)
        # ==========================================================
        if eva_pre > 0 and rorwa_pre >= (self.hurdle + 0.002):
            razon.append(
                f"Préstamo BUENO: EVA_pre={eva_pre:,.0f}€ > 0 y RORWA_pre={rorwa_pre:.2%} ≥ hurdle."
            )
            base_state = "BUENO"
        elif eva_pre <= self.EVA_STRONGLY_NEG_EUR or rorwa_pre < (self.hurdle - 0.002):
            razon.append(
                f"Préstamo MALO: EVA_pre={eva_pre:,.0f}€ muy bajo o RORWA_pre={rorwa_pre:.2%} < hurdle-δ."
            )
            base_state = "MALO"
        else:
            razon.append(
                f"Préstamo AMBIGUO: EVA_pre={eva_pre:,.0f}€, RORWA_pre={rorwa_pre:.2%} ~ hurdle."
            )
            base_state = "AMBIGUO"

        # Riesgo extremo PD/LGD sin reestructura viable
        extreme_risk = (pd_pre >= self.PD_ALTO and lgd_pre >= self.LGD_ALTO)
        if extreme_risk:
            razon.append(
                f"Riesgo extremo (PD={pd_pre:.1%}, LGD={lgd_pre:.1%}) → prioridad venta si reestructura no es viable."
            )

        # ==========================================================
        # 4) Evaluar sugerencia de reestructuración (si existe)
        # ==========================================================
        restruct_ok = bool(restruct_sug and restruct_sug.get("ok", False))
        eva_gain = float(restruct_sug.get("EVA_gain", 0.0)) if restruct_sug else 0.0

        # Mejora “material” de EVA según Banco L1.5
        restruct_material = False
        if restruct_ok and restruct_sug is not None:
            eva_gain_pct = abs(eva_gain) / (abs(eva_pre) + 1e-6)
            restruct_material = (
                (eva_gain > self.EVA_MIN_IMPROVEMENT_EUR)
                or (eva_gain_pct > self.EVA_MIN_IMPROVEMENT_PCT)
            )

            pti_post = self._safe_float(restruct_sug.get("PTI"), default=np.nan)
            dscr_post = self._safe_float(restruct_sug.get("DSCR"), default=np.nan)
            cured = bool(restruct_sug.get("cured", False))

            razon.append(
                f"Reestructura micro: EVA_post={restruct_sug.get('EVA_post', np.nan):,.0f}€ "
                f"(ΔEVA={eva_gain:,.0f}), PTI_post={pti_post if not np.isnan(pti_post) else 'NA'}, "
                f"DSCR_post={dscr_post if not np.isnan(dscr_post) else 'NA'}, cured={cured}."
            )

        # ==========================================================
        # 5) Base preference (regla prudencial)
        # ==========================================================
        if base_state == "BUENO":
            # Sólo reestructurar si mejora MUCHO
            if restruct_material:
                base_pref = "REESTRUCTURAR"
                razon.append("Base: préstamo BUENO pero reestructura crea valor material → REESTRUCTURAR.")
            else:
                base_pref = "MANTENER"
                razon.append("Base: préstamo BUENO, la mejora no compensa intervenir → MANTENER.")

        elif base_state == "MALO":
            # Si reestructuración viable, se intenta salvar; si no, venta
            if restruct_ok and restruct_material:
                base_pref = "REESTRUCTURAR"
                razon.append("Base: préstamo MALO pero reestructura viable y mejora EVA → REESTRUCTURAR.")
            else:
                base_pref = "VENDER"
                razon.append("Base: préstamo MALO sin alternativa clara → VENDER.")

        else:  # AMBIGUO
            if restruct_ok and restruct_material:
                base_pref = "REESTRUCTURAR"
                razon.append("Base: préstamo AMBIGUO y reestructura aporta mejora material → REESTRUCTURAR.")
            else:
                base_pref = "MANTENER"
                razon.append("Base: préstamo AMBIGUO sin reestructura claramente favorable → MANTENER.")

        # Asequibilidad extrema siempre inclina a VENDER
        if (not np.isnan(pti_pre) and pti_pre > self.PTI_CRITICO) or (
            not np.isnan(dscr_pre) and dscr_pre < 1.0
        ):
            base_pref = "VENDER"
            razon.append(
                f"Asequibilidad crítica (PTI≈{pti_pre:.2f}, DSCR≈{dscr_pre:.2f}) → prioridad VENDER."
            )

        if extreme_risk and not restruct_ok:
            base_pref = "VENDER"
            razon.append("Riesgo extremo sin reestructura viable → prioridad VENDER.")

        # ==========================================================
        # 6) Heurísticas micro adicionales
        # ==========================================================
        micro_pref: Optional[str] = None

        # Reestructuración con ΔEVA muy fuerte
        if restruct_ok and restruct_material and eva_gain > 2 * self.EVA_MIN_IMPROVEMENT_EUR:
            micro_pref = "REESTRUCTURAR"
            razon.append("Micro: reestructura con ΔEVA muy fuerte → sugerencia REESTRUCTURAR.")

        # Venta micro si precio NPL favorable y EVA_pre < 0
        if price_sug is not None:
            precio = self._safe_float(price_sug.get("precio_optimo"), default=0.0)
            if precio > 0 and eva_pre < 0:
                micro_pref = "VENDER"
                razon.append(
                    f"Micro: precio NPL favorable (≈{precio:,.0f}€) con EVA_pre<0 → sugerencia VENDER."
                )

        # ==========================================================
        # 7) Fusión jerárquica PPO + prudencial + micro
        # ==========================================================
        accion_final = base_pref

        if accion_macro_desc.startswith(base_pref):
            razon.append("PPO_macro está alineado con la regla prudencial.")
        elif accion_macro_desc.startswith("VENDER") and base_pref == "MANTENER":
            razon.append("Override prudencial: evitar venta de un préstamo que ya crea valor.")
            accion_final = "MANTENER"
        elif accion_macro_desc.startswith("MANTENER") and micro_pref == "REESTRUCTURAR":
            razon.append("Override micro: reestructuración muy beneficiosa, se antepone a PPO.")
            accion_final = "REESTRUCTURAR"
        elif micro_pref is not None:
            razon.append(f"Override micro → {micro_pref}.")
            accion_final = micro_pref
        else:
            razon.append(f"Fallback prudencial → {base_pref}.")

        # ==========================================================
        # 8) Salida final
        # ==========================================================
        return {
            "accion_macro": accion_macro_id,
            "accion_macro_desc": accion_macro_desc,
            "micro_restruct": restruct_sug,
            "micro_price": price_sug,
            "accion_final": accion_final,
            "razon": razon,
        }
