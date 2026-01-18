# -*- coding: utf-8 -*-
# ============================================
# main.py — Orquestador del pipeline completo
# Banco L1.5 · Método Estándar · PPO RL
# ============================================

from __future__ import annotations

import os
import sys
import argparse
import logging
import traceback
from typing import Optional
from datetime import datetime

# ---------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
RUNS_DIR = os.path.join(REPORTS_DIR, "runs")  # corridas de inferencia aquí
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, RUNS_DIR, LOGS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "main.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _auto_pick(path: Optional[str]) -> Optional[str]:
    """Devuelve path si existe; si no, None. Acepta ''/None sin romper."""
    if path and str(path).strip() and os.path.exists(path):
        return path
    return None


def _stamp() -> str:
    """Timestamp estable para evitar colisiones de carpetas."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------
# 1) GENERACIÓN CARTERA SINTÉTICA
# ---------------------------------------------------------------------
def cmd_generate(ns: argparse.Namespace) -> None:
    try:
        from data.generate_portfolio import generate_portfolio, export_excel

        n = int(getattr(ns, "n", 1000))
        out_path = getattr(ns, "out", os.path.join(DATA_DIR, "portfolio_synth.xlsx"))

        logger.info(f"Generando cartera sintética con {n} préstamos…")
        df = generate_portfolio(n=n)
        export_excel(df, out_path)

        logger.info(f"Cartera sintética exportada en: {os.path.abspath(out_path)}")

    except Exception as e:
        logger.error(f"Error durante la generación: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------
# 2) ENTRENAMIENTO (MICRO / MACRO / BOTH)
# ---------------------------------------------------------------------
def cmd_train(ns: argparse.Namespace) -> None:
    try:
        from agent.train_subagents import train_loan_agent, train_portfolio_agent

        portfolio_path = ns.portfolio
        total_steps = int(ns.total_steps)
        agent = ns.agent

        if not os.path.exists(portfolio_path):
            raise FileNotFoundError(f"No se encuentra la cartera: {portfolio_path}")

        logger.info(f"Iniciando entrenamiento RL (modo={agent})")

        if agent in ("loan", "both"):
            logger.info("Entrenando subagente MICRO (LoanEnv)…")
            train_loan_agent(
                portfolio_path=portfolio_path,
                total_timesteps=total_steps,
                device=ns.device,
            )

        if agent in ("portfolio", "both"):
            logger.info("Entrenando subagente MACRO (PortfolioEnv)…")
            train_portfolio_agent(
                portfolio_path=portfolio_path,
                total_timesteps=total_steps,
                device=ns.device,
                top_k=int(ns.top_k),
                scenario=ns.scenario,
            )

        logger.info("Entrenamiento completado correctamente.")

    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------
# 3) INFERENCIA COORDINADA (MACRO+MICRO+GUARDRAILS CIB)
# ---------------------------------------------------------------------
def cmd_infer(ns: argparse.Namespace) -> None:
    """
    Inferencia coordinada:
      - Multi-postura por defecto (prudencial/balanceado/desinversion)
      - Postura única si pasas --single-posture (y eliges --risk-posture)
    """
    try:
        from agent.coordinator_inference import (
            CoordinatorInferenceConfig,
            run_coordinator_inference,
            run_coordinator_inference_multi_posture,
        )

        model_path_micro = ns.model_micro
        portfolio_path = ns.portfolio
        model_path_macro = ns.model_macro

        # VecNormalize (opcionales)
        vn_micro = _auto_pick(ns.vn_micro)
        vn_macro = _auto_pick(ns.vn_macro)
        vn_loan = _auto_pick(ns.vn_loan)

        # knobs macro (diagnóstico/steering) — ya vienen en ns.macro_n_steps / ns.macro_top_k
        macro_n_steps = int(ns.macro_n_steps)
        macro_top_k = int(ns.macro_top_k)

        # Validaciones básicas
        if not os.path.exists(model_path_micro):
            raise FileNotFoundError(f"No existe modelo MICRO (LoanEnv): {model_path_micro}")
        if not os.path.exists(portfolio_path):
            raise FileNotFoundError(f"No existe cartera: {portfolio_path}")
        if model_path_macro and not os.path.exists(model_path_macro):
            raise FileNotFoundError(f"No existe modelo MACRO (PortfolioEnv): {model_path_macro}")

        all_postures = bool(ns.all_postures)

        logger.info("Ejecutando inferencia COORDINADA…")
        logger.info(f"  • All-postures:           {all_postures}")
        logger.info(f"  • Modelo micro:           {model_path_micro}")
        logger.info(f"  • VN micro:               {vn_micro or 'None'}")
        logger.info(f"  • Modelo macro:           {model_path_macro}")
        logger.info(f"  • VN macro:               {vn_macro or 'None'}")
        logger.info(f"  • VN loan(re-rank):       {vn_loan or 'None'}")
        logger.info(f"  • Cartera:                {portfolio_path}")
        logger.info(f"  • Tag:                    {ns.tag}")
        logger.info(f"  • Device/Seed/Det:        {ns.device} / {ns.seed} / {ns.deterministic}")
        logger.info(f"  • Macro n_steps / topK:   {macro_n_steps} / {macro_top_k}")
        logger.info(f"  • Deliverable-only:       {not ns.keep_all}")
        logger.info(f"  • Export audit CSV:       {ns.export_audit_csv}")
        logger.info(f"  • Runs root dir:          {RUNS_DIR}")

        if all_postures:
            # Multi-postura: base_output_dir es la raíz (RUNS_DIR). El coordinator ya crea subcarpetas timestamped.
            cfg_inf = CoordinatorInferenceConfig(
                model_path_micro=model_path_micro,
                portfolio_path=portfolio_path,
                vecnormalize_path_micro=vn_micro,
                model_path_macro=model_path_macro,
                vecnormalize_path_macro=vn_macro,
                vecnormalize_path_loan=vn_loan,
                device=ns.device,
                seed=int(ns.seed),
                deterministic=bool(ns.deterministic),
                tag=ns.tag,
                n_steps=macro_n_steps,
                top_k=macro_top_k,
                deliverable_only=(not ns.keep_all),
                export_audit_csv=bool(ns.export_audit_csv),
                base_output_dir=RUNS_DIR,
            )

            outs = run_coordinator_inference_multi_posture(cfg_inf)

            deliverable_dir = outs[0] if (cfg_inf.deliverable_only and outs) else (outs[-1] if outs else None)

            logger.info("Inferencia coordinada MULTI-POSTURA completada.")
            if deliverable_dir and os.path.isdir(deliverable_dir):
                logger.info(f"DELIVERABLE: {deliverable_dir}")
            else:
                logger.info(f"Outputs: {outs}")

        else:
            # Postura única: OJO con colisiones. Creamos subcarpeta timestamped dentro de RUNS_DIR.
            run_dir = os.path.join(RUNS_DIR, f"{_stamp()}_{ns.tag}_{ns.risk_posture}")
            os.makedirs(run_dir, exist_ok=True)

            out_dir, excel_path = run_coordinator_inference(
                model_micro=model_path_micro,
                portfolio_path=portfolio_path,
                vecnorm_micro_path=vn_micro,
                model_macro=model_path_macro,
                risk_posture=ns.risk_posture,
                n_steps=macro_n_steps,
                top_k=macro_top_k,
                tag=ns.tag,
                base_output_dir=run_dir,  # <-- subcarpeta (evita overwrite)
                device=ns.device,
                seed=int(ns.seed),
                deterministic=bool(ns.deterministic),
                export_audit_csv=bool(ns.export_audit_csv),
                vecnorm_macro_path=vn_macro,
                vecnorm_loan_path=vn_loan,
            )
            logger.info("Inferencia coordinada (postura única) completada.")
            logger.info(f"Carpeta: {out_dir}")
            logger.info(f"Excel final: {excel_path}")

    except Exception as e:
        logger.error(f"Error en inferencia coordinada: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------
# 4) CONSOLIDACIÓN Y RESUMEN EJECUTIVO (OPCIONAL)
# ---------------------------------------------------------------------
def cmd_summary(ns: argparse.Namespace) -> None:
    try:
        # En tu repo real está en reports/summary/results_summary.py
        from reports.results_summary import (
            consolidate_summaries,
            compute_ratios,
            export_outputs,
            generate_executive_summary,
            plot_metric_evolution,
            plot_comparison_bars,
        )

        # Por defecto consolida SOLO desde reports/runs/
        source = ns.source or RUNS_DIR

        logger.info("Consolidando summary.csv…")
        df = consolidate_summaries(source)
        df = compute_ratios(df)
        export_outputs(df, save_excel=bool(ns.excel), save_json=bool(ns.json))

        charts_dir = os.path.join(REPORTS_DIR, "summary", "charts")
        os.makedirs(charts_dir, exist_ok=True)

        if bool(ns.charts):
            plot_metric_evolution(df, os.path.join(charts_dir, "metric_evolution.png"))
            plot_comparison_bars(df, os.path.join(charts_dir, "comparison_bars.png"))

        if bool(ns.executive):
            exec_txt = os.path.join(REPORTS_DIR, "summary", "executive_summary.txt")
            generate_executive_summary(df, exec_txt)

        logger.info("Resumen ejecutivo completado.")

    except Exception as e:
        logger.error(f"Error en consolidación: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Optimizador de Carteras en Default — Banco L1.5")
    sub = parser.add_subparsers(dest="command", required=True)

    # GENERATE
    p_gen = sub.add_parser("generate", help="Genera cartera sintética (Excel)")
    p_gen.add_argument("--n", type=int, default=1000)
    p_gen.add_argument("--out", type=str, default=os.path.join(DATA_DIR, "portfolio_synth.xlsx"))
    p_gen.set_defaults(func=cmd_generate)

    # TRAIN
    p_train = sub.add_parser("train", help="Entrena subagentes (loan, portfolio, both)")
    p_train.add_argument("--agent", type=str, choices=["loan", "portfolio", "both"], default="both")
    p_train.add_argument("--portfolio", type=str, required=True)
    p_train.add_argument("--total-steps", type=int, default=500000, dest="total_steps")
    p_train.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_train.add_argument("--scenario", type=str, choices=["baseline", "adverse", "severe"], default="baseline")
    p_train.add_argument("--device", type=str, default="auto")
    p_train.set_defaults(func=cmd_train)

    # INFER
    p_inf = sub.add_parser("infer", help="Inferencia COORDINADA (multi-postura por defecto)")
    p_inf.add_argument(
        "--model-micro",
        type=str,
        required=True,
        dest="model_micro",
        help="Modelo micro (LoanEnv, best_model.zip)",
    )
    p_inf.add_argument("--portfolio", type=str, required=True, help="Cartera de partida (Excel/CSV)")

    p_inf.add_argument(
        "--model-macro",
        type=str,
        required=False,
        dest="model_macro",
        default=os.path.join(MODELS_DIR, "best_model_portfolio.zip"),
        help="Modelo macro (PortfolioEnv). Default: models/best_model_portfolio.zip",
    )

    # VecNormalize (opcionales)
    p_inf.add_argument("--vn-micro", type=str, required=False, default=None, dest="vn_micro",
                       help="VecNormalize micro (LoanEnv).")
    p_inf.add_argument("--vn-macro", type=str, required=False, default=None, dest="vn_macro",
                       help="VecNormalize macro (PortfolioEnv).")
    p_inf.add_argument("--vn-loan", type=str, required=False, default=None, dest="vn_loan",
                       help="VecNormalize loan (re-ranking micro dentro del macro).")

    # Multi-postura por defecto: True
    p_inf.set_defaults(all_postures=True)
    p_inf.add_argument("--all-postures", action="store_true", dest="all_postures",
                       help="Fuerza multi-postura (prudencial/balanceado/desinversion).")
    p_inf.add_argument("--single-posture", action="store_false", dest="all_postures",
                       help="Fuerza postura única (usar --risk-posture).")

    p_inf.add_argument(
        "--risk-posture",
        type=str,
        choices=["prudencial", "balanceado", "desinversion"],
        default="balanceado",
        help="Solo aplica si usas --single-posture.",
    )

    p_inf.add_argument("--tag", type=str, default="coordinated_policy",
                       help="Etiqueta de corrida (carpeta deliverable)")
    p_inf.add_argument("--device", type=str, default="auto")
    p_inf.add_argument("--seed", type=int, default=42)

    p_inf.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    p_inf.add_argument("--non-deterministic", dest="deterministic", action="store_false")

    # === Macro steering diagnostic knobs (con alias legacy) ===
    # Soporta:
    #   --macro-n-steps / --macro-top-k  (nombres nuevos)
    #   --n-steps / --top-k              (legacy)
    p_inf.add_argument(
        "--macro-n-steps", "--n-steps",
        type=int, default=1, dest="macro_n_steps",
        help="Nº de steps macro (PortfolioEnv). Alias legacy: --n-steps",
    )
    p_inf.add_argument(
        "--macro-top-k", "--top-k",
        type=int, default=5, dest="macro_top_k",
        help="Top-K loans tocados por macro por step. Alias legacy: --top-k",
    )

    p_inf.add_argument("--keep-all", action="store_true", default=False,
                       help="Si se activa, conserva outputs intermedios (no solo DELIVERABLE).")
    p_inf.add_argument("--export-audit-csv", action="store_true", default=False,
                       help="Exporta CSV de auditoría además del Excel final.")
    p_inf.set_defaults(func=cmd_infer)

    # SUMMARY
    p_sum = sub.add_parser("summary", help="Consolida summary.csv y genera informes (opcional)")
    p_sum.add_argument("--source", type=str, default=RUNS_DIR)  # default a runs/

    p_sum.add_argument("--charts", dest="charts", action="store_true", default=True)
    p_sum.add_argument("--no-charts", dest="charts", action="store_false")
    p_sum.add_argument("--excel", dest="excel", action="store_true", default=True)
    p_sum.add_argument("--no-excel", dest="excel", action="store_false")
    p_sum.add_argument("--json", dest="json", action="store_true", default=True)
    p_sum.add_argument("--no-json", dest="json", action="store_false")
    p_sum.add_argument("--executive", dest="executive", action="store_true", default=True)
    p_sum.add_argument("--no-executive", dest="executive", action="store_false")

    p_sum.set_defaults(func=cmd_summary)

    ns = parser.parse_args()

    logger.info(f"Ejecutando comando: {ns.command}")
    ns.func(ns)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Ejecución interrumpida manualmente.")
    except Exception as e:
        logger.error(f"Error crítico en main: {e}")
        traceback.print_exc()
        raise
