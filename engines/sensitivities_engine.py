# ============================================
# engines/sensitivities_engine.py — Banco L1.5
# Sensitividades financieras STD (Basilea III)
# ============================================

import os, sys, argparse, logging, time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import config as cfg
try:
    from data.ingest_portfolio import load_and_validate_portfolio
except ImportError:
    def load_and_validate_portfolio(p):
        return pd.read_excel(p) if p.endswith('.xlsx') else pd.read_csv(p)

# Import stress engine dependencies to apply shocks easily
from engines.stress_engine import apply_shocks, run_coordinator_inference

logger = logging.getLogger("sensitivities_engine")
logging.basicConfig(level=logging.INFO)

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
    """

    def __init__(self):
        self.hurdle     = float(cfg.CONFIG.regulacion.hurdle_rate)
        # Handle callable or float
        val = cfg.CONFIG.regulacion.required_total_capital_ratio
        self.cap_ratio  = float(val()) if callable(val) else float(val)
        self.cost_fund  = 0.006  # mismo funding cost que LoanEnv

        sens = cfg.CONFIG.sensibilidad_reestructura
        self.rw_cure = float(sens.rw_perf_guess)
        self.pd_cure_th = float(sens.pd_cure_threshold)

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
    
    # ... (rest of old logic conceptually remains, but we extend for Tarea 3C)


# ----------------------------------------------------------------------
# NEW: Sensitivity Grid (Tarea 3C)
# ----------------------------------------------------------------------
def run_sensitivity_grid(tag: str, portfolio_path: str, parameter: str, start: float, end: float, step: float, postures: List[str]):
    """
    Runs a grid of stress tests on a single parameter (e.g., PD_mult from 1.0 to 1.5).
    """
    logger.info(f"Starting Sensitivity Grid: {parameter} [{start} -> {end}] step {step}")
    
    # 1. Load Base Portfolio
    df_base = load_and_validate_portfolio(portfolio_path)
    
    # 2. Generate Grid
    # Use numpy for float range
    grid_values = np.arange(start, end + 0.0001, step)
    
    summary_records = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(ROOT_DIR, "reports", f"sensitivity_{tag}_{parameter}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Find Policy Model (Single Check)
    model_micro = os.path.join(ROOT_DIR, "models", "checkpoints", "policy_latest")
    if not os.path.exists(model_micro) and not os.path.exists(model_micro + ".zip"):
         zips = [f for f in os.listdir(os.path.join(ROOT_DIR, "models", "checkpoints")) if f.endswith(".zip")]
         if zips:
               model_micro = os.path.join(ROOT_DIR, "models", "checkpoints", zips[0].replace(".zip", ""))

    for val in grid_values:
        val = float(round(val, 4))
        sc_name = f"{parameter}_{val}"
        
        # Build shock dict
        shocks = {parameter: val}
        
        try:
            df_sens = apply_shocks(df_base, sc_name, shocks)
            
            # Save temp portfolio
            temp_port_path = os.path.join(base_output_dir, f"port_{sc_name}.xlsx")
            df_sens.to_excel(temp_port_path, index=False)
            
            for posture in postures:
                 run_out_dir = os.path.join(base_output_dir, sc_name, posture)
                 os.makedirs(run_out_dir, exist_ok=True)
                 
                 # Run Inference
                 _, excel_path = run_coordinator_inference(
                    model_micro=model_micro,
                    portfolio_path=temp_port_path, 
                    risk_posture=posture,
                    tag=f"{tag}_{sc_name}",
                    base_output_dir=run_out_dir,
                    device="cpu", # Force CPU
                    model_macro=None, 
                    export_audit_csv=False # Keep it light
                )
                 
                 if excel_path and os.path.exists(excel_path):
                    df_res = pd.read_excel(excel_path)
                    
                    summary_records.append({
                        "parameter": parameter,
                        "value": val,
                        "posture": posture,
                        "total_eva": df_res["EVA"].sum() if "EVA" in df_res.columns else 0,
                        "total_rwa": df_res["RWA"].sum() if "RWA" in df_res.columns else 0,
                        "capital_release_realized": df_res["capital_release_realized"].sum() if "capital_release_realized" in df_res.columns else 0,
                        "sell_count": (df_res["Accion_final"] == "VENDER").sum(),
                        "restruct_count": (df_res["Accion_final"] == "REESTRUCTURAR").sum(),
                        "override_count": df_res["override_applied"].sum() if "override_applied" in df_res.columns else 0
                    })
                 else:
                     logger.warning(f"No result for {sc_name}/{posture}")

        except ValueError as e:
            logger.warning(f"Skipping val {val} due to validaton error: {e}")
            continue
        except Exception as e:
            logger.error(f"Error at {val}: {e}")
    
    # Save Results
    if summary_records:
        df_out = pd.DataFrame(summary_records)
        out_path = os.path.join(ROOT_DIR, "reports", f"sensitivity_{tag}_{parameter}.csv")
        df_out.to_csv(out_path, index=False)
        logger.info(f"Sensitivity Analysis Complete. Saved to {out_path}")
        print(df_out.to_markdown())
    else:
        logger.error("No sensitivity results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--parameter", default="PD_mult")
    parser.add_argument("--start", type=float, default=1.0)
    parser.add_argument("--end", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--postures", nargs="+", default=["balanceado"])
    
    args = parser.parse_args()
    
    run_sensitivity_grid(args.tag, args.portfolio, args.parameter, args.start, args.end, args.step, args.postures)

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
