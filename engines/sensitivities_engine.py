# ============================================
# engines/sensitivities_engine.py — Banco L1.5
# Sensitividades financieras STD (Basilea III)
# ============================================

import numpy as np
import config as cfg


class SensitivityEngine:
    """
    Motor unificado de sensitividades financieras usado en:
        - PortfolioEnv (ranking top-K)
        - policy_inference (Explain_Steps)
        - train_agent (reward shaping)
        - StressEngine (coherencia métrica)

    Mide cómo pequeños cambios en PD, LGD, tasa, EAD y RW afectan:
        • NI          (net income)
        • RORWA       (rentabilidad por riesgo)
        • EVA         (economic value added)
        • Capital     (RWA × ratio)

    Se asegura consistencia 100% con LoanEnv:
        NI  = EAD*rate − PD*LGD*EAD − fundingCost
        RWA = EAD * RW
        RORWA = NI / RWA
        EVA = RWA * (RORWA − hurdle)
    """

    def __init__(self):
        self.hurdle     = cfg.CONFIG.regulacion.hurdle_rate
        self.cap_ratio  = cfg.CONFIG.regulacion.required_total_capital_ratio()
        self.cost_fund  = 0.006  # mismo funding cost que LoanEnv

        self.rw_cure = cfg.CONFIG.sensibilidad_reestructura.rw_perf_guess
        self.pd_cure_th = cfg.CONFIG.sensibilidad_reestructura.pd_cure_threshold

    # ============================================================
    # 🔹 DERIVADAS PRIMERAS DE NI
    # ============================================================
    def dNI_dPD(self, loan):
        """∂NI/∂PD = - LGD * EAD"""
        return float(-loan["LGD"] * loan["EAD"])

    def dNI_dLGD(self, loan):
        """∂NI/∂LGD = - PD * EAD"""
        return float(-loan["PD"] * loan["EAD"])

    def dNI_drate(self, loan):
        """∂NI/∂rate = EAD"""
        return float(loan["EAD"])

    def dNI_dEAD(self, loan):
        """
        ∂NI/∂EAD = rate − PD*LGD − fundingCost
        (coherente con LoanEnv)
        """
        return float(loan["rate"] - loan["PD"]*loan["LGD"] - self.cost_fund)

    # ============================================================
    # 🔹 SENSIBILIDAD RORWA
    # ============================================================
    def dRORWA(self, dNI, RWA):
        """∂RORWA = dNI / RWA"""
        return float(dNI / max(RWA, 1e-9))

    # ============================================================
    # 🔹 SENSIBILIDAD EVA = RWA*(RORWA−hurdle)
    # ============================================================
    def partial_eva_pd(self, loan):
        dni  = self.dNI_dPD(loan)
        drrw = self.dRORWA(dni, loan["RWA"])
        return float(loan["RWA"] * drrw)

    def partial_eva_lgd(self, loan):
        dni  = self.dNI_dLGD(loan)
        drrw = self.dRORWA(dni, loan["RWA"])
        return float(loan["RWA"] * drrw)

    def partial_eva_rate(self, loan):
        dni  = self.dNI_drate(loan)
        drrw = self.dRORWA(dni, loan["RWA"])
        return float(loan["RWA"] * drrw)

    def partial_eva_ead(self, loan):
        """
        ∂EVA/∂EAD =
            RWA * (∂RORWA/∂EAD)
            + (RORWA - hurdle) * ∂RWA/∂EAD

        Con:
            ∂RWA/∂EAD = RW
            ∂NI/∂EAD = rate - PD*LGD - funding
        """
        RW  = loan["RW"]
        RWA = loan["RWA"]

        dNI = self.dNI_dEAD(loan)
        dRORWA = dNI / max(RWA, 1e-9)

        return float(RWA * dRORWA + (loan["RORWA"] - self.hurdle) * RW)

    # ============================================================
    # 🔹 SENSIBILIDADES DE RWA Y CAPITAL
    # ============================================================
    def partial_rwa_rw(self, loan):
        """∂RWA/∂RW = EAD"""
        return float(loan["EAD"])

    def partial_capital_rw(self, loan):
        """∂Capital/∂RW = EAD × cap_ratio"""
        return float(loan["EAD"] * self.cap_ratio)

    # ============================================================
    # 🔹 SENSIBILIDAD A LA CURACIÓN (PD < threshold)
    # ============================================================
    def sensitivity_cure(self, loan):
        """
        ∂EVA/∂(cure flag) ≈ EVA_cured − EVA_actual

        Lógica coherente con StressEngine y LoanEnv.
        """

        # Ya está curado -> sensibilidad 0
        if loan.get("cured", False):
            return 0.0

        EAD = loan["EAD"]

        RW_c = self.rw_cure
        RWA_c = EAD * RW_c

        # NI aproximado sin cambiar PD ni LGD (proxy suave)
        NI_c = (
            EAD * loan["rate"]
            - loan["PD"] * loan["LGD"] * EAD
            - EAD * self.cost_fund
        )

        RORWA_c = NI_c / max(RWA_c, 1e-9)
        EVA_c = RWA_c * (RORWA_c - self.hurdle)

        return float(EVA_c - loan["EVA"])

    # ============================================================
    # 🔹 SCORE GLOBAL (ranking top-K)
    # ============================================================
    def global_sensitivity_score(self, loan):
        """
        Combina todas las sensibilidades principales.
        Se usa para:
            • top-K en PortfolioEnv
            • heurísticas macro
            • Explain_Steps (policy_inference_portfolio)
        """

        # Pesos calibrados Banco L1.5 (robustos y prudenciales)
        w_pd   = 0.35
        w_lgd  = 0.25
        w_rate = 0.10
        w_ead  = 0.10
        w_rw   = 0.10
        w_cure = 0.10

        s_pd   = abs(self.partial_eva_pd(loan))
        s_lgd  = abs(self.partial_eva_lgd(loan))
        s_rate = abs(self.partial_eva_rate(loan))
        s_ead  = abs(self.partial_eva_ead(loan))
        s_rw   = abs(self.partial_capital_rw(loan))
        s_cure = abs(self.sensitivity_cure(loan))

        score = (
            w_pd*s_pd +
            w_lgd*s_lgd +
            w_rate*s_rate +
            w_ead*s_ead +
            w_rw*s_rw +
            w_cure*s_cure
        )

        return float(score)
