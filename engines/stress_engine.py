# ============================================
# engines/stress_engine.py — Banco L1.5
# Motor macro de estrés multi-periodo (BCE/EBA)
# ============================================

import numpy as np
import config as cfg


class StressEngine:
    """
    Motor de estrés macro coherente con Banco L1.5.
    Corrige:
    - __init__ mal definido
    - coherencia EVA/RORWA con LoanEnv
    - dinámica PD–LGD–RW con límites regul.
    - efecto cured realista
    - DPD/meses_en_default compatible con PortfolioEnv
    """

    def __init__(self, n_periods: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_periods = int(n_periods)

        # Parámetros Banco L1.5
        self.hurdle = cfg.CONFIG.regulacion.hurdle_rate
        self.cap_ratio = cfg.CONFIG.regulacion.required_total_capital_ratio()
        self.cost_funding = 0.006

        sens = cfg.CONFIG.sensibilidad_reestructura
        self.pd_cure_th = sens.pd_cure_threshold
        self.rw_cured = sens.rw_perf_guess

    # ----------------------------------------------------------------------
    #   Shock macro estilo EBA/BCE
    # ----------------------------------------------------------------------
    def _scenario_shock(self, scenario: str):
        """
        Multiplicadores para PD, LGD, RW, rate y shock DPD.
        Los valores son escalados prudenciales: permitemos
        shocks >1 pero con clipping posterior estricto.
        """
        if scenario == "baseline":
            return dict(
                pd_mu=1.00, pd_sd=0.02,
                lgd_mu=1.00, lgd_sd=0.01,
                rw_mu=1.00, rw_sd=0.01,
                rate_mu=1.00, rate_sd=0.01,
                dpd_add=15
            )

        elif scenario == "adverse":
            return dict(
                pd_mu=1.20, pd_sd=0.05,
                lgd_mu=1.05, lgd_sd=0.03,
                rw_mu=1.05, rw_sd=0.02,
                rate_mu=0.98, rate_sd=0.02,
                dpd_add=45
            )

        elif scenario == "severe":
            return dict(
                pd_mu=1.35, pd_sd=0.08,
                lgd_mu=1.10, lgd_sd=0.04,
                rw_mu=1.10, rw_sd=0.03,
                rate_mu=0.95, rate_sd=0.03,
                dpd_add=75
            )

        elif scenario == "extreme":
            return dict(
                pd_mu=1.55, pd_sd=0.12,
                lgd_mu=1.20, lgd_sd=0.06,
                rw_mu=1.15, rw_sd=0.05,
                rate_mu=0.92, rate_sd=0.04,
                dpd_add=120
            )

        else:
            raise ValueError(f"Escenario macro no reconocido: {scenario}")

    # ----------------------------------------------------------------------
    #   Shock a un solo periodo
    # ----------------------------------------------------------------------
    def apply_macro_shock_once(self, loan: dict, scenario="baseline") -> dict:
        """
        Aplica un shock macro de 1 periodo. Mantiene coherencia con LoanEnv.
        """
        sh = self._scenario_shock(scenario)

        # ----------------------------
        # 1. Shock correlacionado
        # ----------------------------
        PD_mul = self.rng.normal(sh["pd_mu"], sh["pd_sd"])
        LGD_mul = self.rng.normal(sh["lgd_mu"], sh["lgd_sd"])
        RW_mul = self.rng.normal(sh["rw_mu"], sh["rw_sd"])
        rate_mul = self.rng.normal(sh["rate_mu"], sh["rate_sd"])

        PD = float(loan["PD"] * PD_mul)
        LGD = float(loan["LGD"] * LGD_mul)
        RW = float(loan["RW"] * RW_mul)
        rate = float(loan["rate"] * rate_mul)

        # ----------------------------
        # 2. Clipping prudencial
        # ----------------------------
        PD = np.clip(PD, 0.0001, 1.0)
        LGD = np.clip(LGD, 0.05, 1.0)
        RW = np.clip(RW, 0.05, 3.0)
        rate = np.clip(rate, 0.0, 0.25)

        # ----------------------------
        # 3. Actualización DPD
        # ----------------------------
        DPD_prev = float(loan.get("DPD", 120.0))
        DPD = DPD_prev + sh["dpd_add"]

        # Estimación fija coherente con portfolio_env:
        meses_en_default = int(DPD // 30)

        # ----------------------------
        # 4. Cálculo STD coherente LoanEnv
        # ----------------------------
        EAD = float(loan["EAD"])
        EL = PD * LGD * EAD

        NI = EAD * rate - EL - EAD * self.cost_funding

        RWA = EAD * RW
        RORWA = float(NI / max(RWA, 1e-9))
        EVA = float(RWA * (RORWA - self.hurdle))

        # ----------------------------
        # 5. Efecto cure (coherente)
        # ----------------------------
        cured = PD < self.pd_cure_th

        if cured:
            RW = float(self.rw_cured)
            RWA = EAD * RW
            RORWA = float(NI / max(RWA, 1e-9))
            EVA = float(RWA * (RORWA - self.hurdle))

        # ----------------------------
        # 6. Guardar estado actualizado
        # ----------------------------
        loan.update(
            PD=float(PD),
            LGD=float(LGD),
            RW=float(RW),
            rate=float(rate),
            DPD=float(DPD),
            meses_en_default=meses_en_default,
            RWA=float(RWA),
            RORWA=float(RORWA),
            EVA=float(EVA),
            cured=bool(cured),
        )

        return loan

    # ----------------------------------------------------------------------
    #   Trayectoria multiperiodo (6–8 trimestres)
    # ----------------------------------------------------------------------
    def apply_stress_path(self, loan: dict, scenario="baseline") -> dict:
        """
        Aplica n_periods shocks consecutivos.
        """
        loan = dict(loan)
        for _ in range(self.n_periods):
            loan = self.apply_macro_shock_once(loan, scenario)
        return loan

    # ----------------------------------------------------------------------
    #   Estrés a cartera completa
    # ----------------------------------------------------------------------
    def stress_portfolio(self, df, scenario="baseline"):
        """
        Aplica estrés multi-periodo a toda la cartera.
        Devuelve lista de dicts (compatible con portfolio_env y RL).
        """
        out = []
        for _, row in df.iterrows():
            loan = row.to_dict()
            stressed = self.apply_stress_path(loan, scenario)
            out.append(stressed)
        return out
