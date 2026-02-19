# ========================================
# Instalador inteligente de dependencias POC 2 (Santander – NTT Data)
# Compatible con Python 3.14 (Windows-friendly)
# SIN psycopg (dependencia opcional eliminada)
# ========================================

import importlib
import subprocess
import sys
import os

python_exec = sys.executable


def _run(cmd, check=True, quiet=False):
    kwargs = {}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.run(cmd, check=check, **kwargs)


# ------------------------------------------------
# Actualización de pip
# ------------------------------------------------
try:
    _run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], quiet=True)
    print("[OK] pip actualizado correctamente.\n")
except Exception:
    print("[WARN]  No se pudo actualizar pip, se usará la versión actual.\n")


# ------------------------------------------------
# Paquetes requeridos (STACK COMPLETO PROYECTO)
# ------------------------------------------------
requirements = {
    # Core numérico
    "numpy": "2.3.4",
    "pandas": "2.3.3",
    "scipy": "1.14.1",
    "joblib": "1.4.2",
    "tqdm": "4.66.5",
    "rich": "13.9.4",   # [REQ] NECESARIO para progress_bar de SB3

    # Excel / reporting
    "openpyxl": "3.1.5",

    # Visualización / dashboards
    "matplotlib": "3.10.7",
    "seaborn": "0.13.2",
    "plotly": "5.24.1",
    "dash": "2.18.1",
    "dash_bootstrap_components": "1.6.0",

    # RL
    "gymnasium": "1.0.0",
    "stable_baselines3": "2.6.0",
    "torch": "2.9.1",

    # Otros ML (puede fallar en 3.14 según wheels disponibles)
    "xgboost": "2.1.2",

    # DB / logging (sin psycopg)
    "sqlalchemy": "2.0.36",
    "tensorboard": "2.18.0",

    # Parquet
    "pyarrow": "22.0.0",
    "fastparquet": "2024.5.0",
}


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def is_installed(pkg_name: str, expected_version: str) -> bool:
    try:
        module = importlib.import_module(pkg_name)
        version = getattr(module, "__version__", None)
        if version == expected_version:
            print(f"[OK] {pkg_name} ya está instalado ({version})")
            return True
        elif version:
            print(f"[WARN]  {pkg_name} instalado ({version}), esperado {expected_version}")
            return False
        else:
            print(f"[WARN]  {pkg_name} instalado sin versión visible")
            return False
    except ModuleNotFoundError:
        print(f"[ERR] {pkg_name} no está instalado.")
        return False
    except ImportError as e:
        print(f"[WARN]  {pkg_name} instalado pero no importable ({e})")
        return False


def install(pkg_name: str, version: str):
    pip_name = pkg_name.replace("_", "-")
    print(f"[INSTALL] Instalando {pip_name}=={version} ...")
    try:
        subprocess.check_call([
            python_exec, "-m", "pip", "install",
            f"{pip_name}=={version}",
            "--prefer-binary",
            "--no-build-isolation",
            "--upgrade-strategy", "only-if-needed"
        ])
        print(f"[OK] {pkg_name} instalado correctamente.\n")
    except subprocess.CalledProcessError:
        print(
            f"[ERR] Error instalando {pkg_name}. "
            f"Puede no existir wheel para Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        )


def install_torch_tolerant(expected_version: str):
    if is_installed("torch", expected_version):
        return

    print(f"[INSTALL] Instalando torch=={expected_version} ...")
    try:
        subprocess.check_call([
            python_exec, "-m", "pip", "install",
            f"torch=={expected_version}",
            "--prefer-binary"
        ])
        import torch  # noqa
        print("[OK] torch instalado correctamente.\n")
    except Exception:
        print(
            "[ERR] No se pudo instalar torch correctamente.\n"
            "[INFO] Recomendación profesional: Python 3.11 / 3.12 para RL estable.\n"
        )


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    print("========================================")
    print("[START] Instalación dependencias POC 2 (Santander – NTT Data)")
    print(f"[PYTHON] Python: {sys.version}")
    print("========================================\n")

    for pkg, version in requirements.items():
        if pkg == "torch":
            install_torch_tolerant(version)
            continue

        if not is_installed(pkg, version):
            install(pkg, version)

    print("\n[DONE] Dependencias instaladas / verificadas.\n")

    # Validación Excel
    try:
        import openpyxl  # noqa
        print("[OK] Validación Excel: openpyxl OK.")
    except Exception as e:
        print(f"[ERR] openpyxl fallo tras instalación ({e})")

    # Validación RL (incluye progress bar)
    try:
        import gymnasium, stable_baselines3, tqdm, rich  # noqa
        print("[OK] Validación RL: gymnasium + SB3 + progress bar OK.")
    except Exception as e:
        print(f"[WARN]  Validación RL incompleta ({e})")

    # Validación Parquet
    try:
        import pandas as pd
        pd.DataFrame({"test": [1]}).to_parquet("_test.parquet")
        print("[OK] Validación Parquet OK.")
    except Exception as e:
        print(f"[WARN]  Validación Parquet falló ({e})")
    finally:
        if os.path.exists("_test.parquet"):
            os.remove("_test.parquet")


if __name__ == "__main__":
    main()
