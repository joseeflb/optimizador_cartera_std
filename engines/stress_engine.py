# ============================================
# engines/stress_engine.py — Banco L1.5
# Motor macro de estrés multi-periodo (BCE/EBA)
# ============================================

import os, sys, yaml, argparse, logging, time
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

# We need access to the Coordinator Logic or Policy Inference
# For strictly "stress", we might want to run the full coordinator pipeline 
# OR just the policy inference if macro is static. 
# However, Tarea 3B says "ejecutar inferencia coordinada".
from agent.coordinator_inference import run_coordinator_inference, InferenceConfig

logger = logging.getLogger("stress_engine")
logging.basicConfig(level=logging.INFO)

class StressEngine:
    """
    Motor de estrés macro coherente con Banco L1.5.
    (Legacy methods kept for backward compat if needed, but extended for Tarea 3B)
    """

    def __init__(self, n_periods: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_periods = int(n_periods)
        
        # Parámetros Banco L1.5
        self.hurdle = float(cfg.CONFIG.regulacion.hurdle_rate)
        val = cfg.CONFIG.regulacion.required_total_capital_ratio
        self.cap_ratio = float(val()) if callable(val) else float(val)
        self.cost_funding = 0.006

        sens = cfg.CONFIG.sensibilidad_reestructura
        self.pd_cure_th = float(sens.pd_cure_threshold)
        self.rw_cured = float(sens.rw_perf_guess)

    # ----------------------------------------------------------------------
    #   Shock macro estilo EBA/BCE (Legacy / Internal simulation)
    # ----------------------------------------------------------------------
    def _scenario_shock(self, scenario: str):
        """
        Multiplicadores para PD, LGD, RW, rate y shock DPD.
        Los valores son escalados prudenciales.
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
                lgd_mu=1.10, lgd_sd=0.03,
                rw_mu=1.15, rw_sd=0.02,
                rate_mu=1.05, rate_sd=0.02,
                dpd_add=45
            )
        # Fallback
        return self._scenario_shock("baseline")


# ----------------------------------------------------------------------
# NEW: Stress Execution Wrapper (Tarea 3B)
# ----------------------------------------------------------------------
def apply_shocks(df: pd.DataFrame, scenario_name: str, shocks: Dict[str, float]) -> pd.DataFrame:
    """
    Applies defined shocks to a portfolio dataframe.
    Respects ALLOW_CLIP_OUT_OF_RANGE config for bounds checking.
    """
    out = df.copy()
    logger.info(f"Applying scenario '{scenario_name}' shocks: {shocks}")
    
    allow_clip = getattr(cfg, "ALLOW_CLIP_OUT_OF_RANGE", False)

    # 1. PD Multiplier
    if "PD_mult" in shocks and "PD" in out.columns:
        out["PD"] = out["PD"] * shocks["PD_mult"]
        
    # 2. LGD Additive
    if "LGD_add" in shocks and "LGD" in out.columns:
        out["LGD"] = out["LGD"] + shocks["LGD_add"]

    # 3. RW Multiplier
    if "RW_mult" in shocks and "RW" in out.columns:
        out["RW"] = out["RW"] * shocks["RW_mult"]

    # 4. Collateral Multiplier
    if "collateral_mult" in shocks and "collateral_value" in out.columns:
        out["collateral_value"] = out["collateral_value"] * shocks["collateral_mult"]

    # 5. Interest Rate
    if "interest_rate_add" in shocks and "interest_rate" in out.columns:
         out["interest_rate"] = out["interest_rate"] + shocks["interest_rate_add"]

    # --- Validation / Clipping ---
    # PD [0, 1]
    if "PD" in out.columns:
        mask_pd = (out["PD"] > 1.0) | (out["PD"] < 0.0)
        if mask_pd.any():
            msg = f"Scenario '{scenario_name}': PD out of range [0,1] for {mask_pd.sum()} loans."
            if allow_clip:
                logger.warning(f"{msg} Clipping.")
                out["PD"] = out["PD"].clip(0.0, 1.0)
            else:
                raise ValueError(f"STRICT STRESS ERROR: {msg} (Set ALLOW_CLIP_OUT_OF_RANGE=True)")

    # LGD [0, 1]
    if "LGD" in out.columns:
        mask_lgd = (out["LGD"] > 1.0) | (out["LGD"] < 0.0)
        if mask_lgd.any():
            msg = f"Scenario '{scenario_name}': LGD out of range [0,1] for {mask_lgd.sum()} loans."
            if allow_clip:
                logger.warning(f"{msg} Clipping.")
                out["LGD"] = out["LGD"].clip(0.0, 1.0)
            else:
                 raise ValueError(f"STRICT STRESS ERROR: {msg} (Set ALLOW_CLIP_OUT_OF_RANGE=True)")

    # RW often allows >1 (e.g. 150%) but if logic expects [0,1], check model. 
    # Usually RW is valid > 100%. No clipping for high RW usually, just limit to reasonable max (e.g. 1250% for securitization, or 150% for defaulted)
    
    return out


def run_stress_pipeline(tag: str, portfolio_path: str, scenarios_yaml: str, postures: List[str]):
    """
    Main orchestration for stress testing.
    """
    # 1. Load Base Portfolio
    logger.info(f"Loading base portfolio: {portfolio_path}")
    df_base = load_and_validate_portfolio(portfolio_path)
    
    # 2. Load Scenarios
    if not os.path.exists(scenarios_yaml):
         raise FileNotFoundError(f"Scenarios file not found: {scenarios_yaml}")
    
    with open(scenarios_yaml, 'r') as f:
         scenarios_cfg = yaml.safe_load(f)
         
    # Fix: scenarios might be at root or under 'scenarios' key
    scenarios = scenarios_cfg.get("scenarios", scenarios_cfg) 
    
    # 3. Iterate Scenarios & Postures
    summary_records = []
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(ROOT_DIR, "reports", f"stress_{tag}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)

    for sc_name, sc_data in scenarios.items():
        if not isinstance(sc_data, dict): continue # Skip metadata if any
        
        logger.info(f"--- Processing Scenario: {sc_name} ---")
        shocks = sc_data.get("shocks", {})
        
        # Apply Shocks
        try:
            df_stressed = apply_shocks(df_base, sc_name, shocks)
        except ValueError as e:
            logger.error(f"Skipping scenario {sc_name} due to strict validation error: {e}")
            continue
        
        # Save stressed portfolio (temp)
        stressed_port_path = os.path.join(base_output_dir, f"portfolio_{sc_name}.xlsx")
        df_stressed.to_excel(stressed_port_path, index=False)
        
        for posture in postures:
            logger.info(f"   > Running Posture: {posture}")
            
            run_output_dir = os.path.join(base_output_dir, sc_name, posture)
            os.makedirs(run_output_dir, exist_ok=True)
            
            # Direct call to run_coordinator_inference with arguments
            # Note: We adapt based on the signature seen in coordinator_inference.py
            
            try:
                 # run_coordinator_inference signature:
                 # (model_micro, portfolio_path, risk_posture, tag, base_output_dir, ...)
                 
                 # We assume model paths are standard unless provided
                 model_micro = os.path.join(ROOT_DIR, "models", "checkpoints", "policy_latest")
                 # Check if policy_latest exists, else try to find one
                 if not os.path.exists(model_micro) and not os.path.exists(model_micro + ".zip"):
                      # Find any .zip in checkpoints
                      zips = [f for f in os.listdir(os.path.join(ROOT_DIR, "models", "checkpoints")) if f.endswith(".zip")]
                      if zips:
                           model_micro = os.path.join(ROOT_DIR, "models", "checkpoints", zips[0])
                           model_micro = model_micro.replace(".zip", "") # stable-baselines3 adds .zip automatically usually

                 # Execute
                 out_dir, excel_path = run_coordinator_inference(
                    model_micro=model_micro,
                    portfolio_path=stressed_port_path, # Shocked portfolio
                    risk_posture=posture,
                    tag=f"{tag}_{sc_name}",
                    base_output_dir=run_output_dir,
                    device="cpu", # Force CPU for stability in batch
                    # Defaults for others
                    model_macro=None, 
                    export_audit_csv=True
                )
                
                 # Check results
                 if excel_path and os.path.exists(excel_path):
                    df_res = pd.read_excel(excel_path)

                    summary_records.append({
                        "scenario": sc_name,
                        "posture": posture,
                        "total_ead": df_res["EAD"].sum(),
                        "total_eva": df_res["EVA"].sum() if "EVA" in df_res.columns else 0,
                        "total_rwa": df_res["RWA"].sum() if "RWA" in df_res.columns else 0,
                        "n_sales": (df_res["Accion_final"] == "VENDER").sum(),
                        "n_restruct": (df_res["Accion_final"] == "REESTRUCTURAR").sum(),
                    })
                 else:
                    logger.warning(f"No results found for {sc_name} / {posture}")

            except Exception as e:
                logger.error(f"Inference failed for {sc_name}/{posture}: {e}")
                # import traceback
                # traceback.print_exc()

    # 4. Save Summary
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        summary_path = os.path.join(ROOT_DIR, "reports", f"stress_summary_{tag}.csv")
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Stress Test Complete. Summary saved to: {summary_path}")
        print(df_summary.to_markdown())
    else:
        logger.error("No stress test results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--scenarios", default="configs/stress_scenarios.yaml")
    parser.add_argument("--postures", nargs="+", default=["prudencial", "balanceado", "desinversion"])
    args = parser.parse_args()
    
    run_stress_pipeline(args.tag, args.portfolio, args.scenarios, args.postures)


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
