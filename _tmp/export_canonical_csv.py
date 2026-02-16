#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_canonical_csv.py

Exporta decisiones a CSV canónico (formato estable, reproducible, auditable)
con orden fijo, float_format consistente y MANIFEST.json con hashes SHA256.

USO:
    python _tmp/export_canonical_csv.py --xlsx reports/decisiones_NPL_BANKGRADE_pru_XXXXXX.xlsx --output-dir reports/canonical/
    
OUTPUT:
    - decisiones_pru_canonical.csv (orden fijo, columnas estables)
    - MANIFEST.json (SHA256 hash + metadata)
    - verification_report.txt (checksums para auditoría)
"""
import os
import sys
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Orden fijo de columnas (bank-grade schema)
CANONICAL_COLUMNS_ORDER = [
    # Identificación
    "loan_id", "segmento", "rating", "secured_flag",
    # Exposición y riesgo
    "EAD", "PD", "LGD", "RW", "RWA", "book_value",
    # Estado NPL
    "DPD", "meses_en_default", "age_npl_m",
    # Decisión final
    "Accion_final", "Reason_Code", "Convergencia_Caso",
    # Recomendaciones micro/macro
    "Accion_micro", "Accion_macro", "Macro_Assignment",
    # EVA y capital
    "EVA_pre", "EVA_post", "EVA_delta",
    "capital_release_cf", "capital_release_realized",
    # Reestructura (params + viabilidad)
    "plazo_optimo", "tasa_nueva", "quita",
    "PTI", "DSCR", "acceptance_score",
    "recovery_restruct", "EVA_restruct",
    # Venta NPL (pricing + ejecutabilidad)
    "precio_optimo", "recovery_sale", "pnl_realized",
    "sale_insulting_flag", "sale_within_loss_cap", "sale_meets_recovery_min",
    "sale_executable", "sale_loss_pct",
    # Mandatos y gates
    "sale_mandate", "sale_mandate_reason",
    "valor_referencia", "recovery_rate",
    # Escalación (casos críticos)
    "case_status", "next_step", "next_step_reason",
    "review_due_days", "required_data_flags", "override_reason",
]


def compute_sha256(filepath: str) -> str:
    """Calcula SHA256 de archivo (auditable, deterministic)."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def export_canonical_csv(
    xlsx_path: str,
    output_csv: str,
    sheet_name: str = "Decisiones",
    float_format: str = "%.6f"
) -> pd.DataFrame:
    """
    Exporta decisiones a CSV canónico con orden fijo y formato estable.
    
    Returns:
        df: DataFrame exportado (para auditoría)
    """
    print(f"[1/4] Leyendo XLSX: {xlsx_path}")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    print(f"   → {len(df)} préstamos, {len(df.columns)} columnas")
    
    # Orden canónico (missing columns → skip)
    available_cols = [c for c in CANONICAL_COLUMNS_ORDER if c in df.columns]
    df_canonical = df[available_cols].copy()
    
    print(f"[2/4] Ordenando por loan_id (deterministic sort)")
    if "loan_id" in df_canonical.columns:
        df_canonical = df_canonical.sort_values("loan_id").reset_index(drop=True)
    
    print(f"[3/4] Exportando CSV canónico: {output_csv}")
    # Format estable: números con 6 decimales, sin index
    df_canonical.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        float_format=float_format,
        lineterminator="\n"  # Unix-style (consistente cross-platform)
    )
    
    print(f"   ✅ CSV guardado: {os.path.basename(output_csv)}")
    return df_canonical


def generate_manifest(
    csv_path: str,
    xlsx_path: str,
    output_manifest: str,
    tag: str = "unknown",
    posture: str = "unknown"
):
    """Genera MANIFEST.json con hashes SHA256 y metadata (auditable)."""
    print(f"[4/4] Generando MANIFEST.json: {output_manifest}")
    
    csv_hash = compute_sha256(csv_path)
    xlsx_hash = compute_sha256(xlsx_path)
    
    manifest = {
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tag": tag,
        "risk_posture": posture,
        "files": {
            "canonical_csv": {
                "path": os.path.basename(csv_path),
                "sha256": csv_hash,
                "size_bytes": os.path.getsize(csv_path),
                "format": "CSV UTF-8 (Unix line endings)",
                "float_format": "%.6f",
                "order": "loan_id ASC"
            },
            "source_xlsx": {
                "path": os.path.basename(xlsx_path),
                "sha256": xlsx_hash,
                "size_bytes": os.path.getsize(xlsx_path),
                "warning": "XLSX hash puede variar por metadata Excel; usar CSV como referencia canónica"
            }
        },
        "verification": {
            "command": f"python _tmp/verify_manifest.py --csv {os.path.basename(csv_path)} --manifest {os.path.basename(output_manifest)}",
            "expected_csv_hash": csv_hash
        }
    }
    
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ MANIFEST guardado")
    print(f"\n🔐 HASHES:")
    print(f"   CSV:  {csv_hash}")
    print(f"   XLSX: {xlsx_hash}")
    print(f"\n📋 VERIFICACIÓN:")
    print(f"   {manifest['verification']['command']}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Exportar decisiones a CSV canónico con MANIFEST (reproducible, auditable)"
    )
    parser.add_argument("--xlsx", required=True, help="Path al XLSX de decisiones")
    parser.add_argument("--output-dir", default="reports/canonical", help="Directorio de salida")
    parser.add_argument("--tag", default="unknown", help="Tag de la corrida (ej: NPL_BANKGRADE)")
    parser.add_argument("--posture", default="unknown", help="Postura de riesgo (prudencial/balanceado/desinversion)")
    parser.add_argument("--sheet", default="Decisiones", help="Nombre de la hoja Excel")
    
    args = parser.parse_args()
    
    # Crear output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths de salida
    posture_short = args.posture[:3].lower() if args.posture != "unknown" else "unk"
    csv_filename = f"decisiones_{posture_short}_canonical.csv"
    manifest_filename = f"MANIFEST_{posture_short}.json"
    
    csv_path = os.path.join(args.output_dir, csv_filename)
    manifest_path = os.path.join(args.output_dir, manifest_filename)
    
    print("=" * 70)
    print("🔐 EXPORT CANÓNICO BANK-READY (CSV + MANIFEST)")
    print("=" * 70)
    
    # Export CSV canónico
    df_canonical = export_canonical_csv(
        xlsx_path=args.xlsx,
        output_csv=csv_path,
        sheet_name=args.sheet
    )
    
    # Generar MANIFEST
    manifest = generate_manifest(
        csv_path=csv_path,
        xlsx_path=args.xlsx,
        output_manifest=manifest_path,
        tag=args.tag,
        posture=args.posture
    )
    
    print("\n" + "=" * 70)
    print("✅ EXPORT CANÓNICO COMPLETADO")
    print("=" * 70)
    print(f"📁 Output dir: {args.output_dir}")
    print(f"   - {csv_filename} ({len(df_canonical)} loans)")
    print(f"   - {manifest_filename}")
    print(f"\n🔐 Para verificar integridad:")
    print(f"   python _tmp/verify_manifest.py --manifest {manifest_path}")
    

if __name__ == "__main__":
    main()
