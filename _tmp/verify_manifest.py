#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_manifest.py

Verifica integridad de CSV canónico contra MANIFEST.json (auditoría bank-ready).

USO:
    python _tmp/verify_manifest.py --manifest reports/canonical/MANIFEST_pru.json
    
OUTPUT:
    - ✅ CSV integrity OK (hash matches)
    - ❌ CSV tampered / modified (hash mismatch)
"""
import os
import sys
import argparse
import hashlib
import json
from pathlib import Path


def compute_sha256(filepath: str) -> str:
    """Calcula SHA256 de archivo."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_manifest(manifest_path: str) -> bool:
    """
    Verifica integridad del CSV canónico contra MANIFEST.json.
    
    Returns:
        True si hash coincide (integridad OK), False si hay tampering.
    """
    print("=" * 70)
    print("🔐 VERIFICACIÓN DE INTEGRIDAD (MANIFEST + CSV)")
    print("=" * 70)
    
    # Leer MANIFEST
    print(f"\n[1/3] Leyendo MANIFEST: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    csv_info = manifest["files"]["canonical_csv"]
    expected_hash = csv_info["sha256"]
    csv_filename = csv_info["path"]
    
    print(f"   → CSV esperado: {csv_filename}")
    print(f"   → Hash esperado: {expected_hash}")
    
    # Localizar CSV (mismo directorio que MANIFEST)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    csv_path = os.path.join(manifest_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: CSV no encontrado: {csv_path}")
        return False
    
    # Calcular hash actual
    print(f"\n[2/3] Calculando hash actual de CSV: {csv_filename}")
    actual_hash = compute_sha256(csv_path)
    print(f"   → Hash actual: {actual_hash}")
    
    # Comparar
    print(f"\n[3/3] Verificando integridad...")
    if actual_hash == expected_hash:
        print("\n" + "=" * 70)
        print("✅ INTEGRIDAD OK - CSV canónico NO ha sido modificado")
        print("=" * 70)
        print(f"   - CSV: {csv_filename}")
        print(f"   - Hash: {actual_hash}")
        print(f"   - Generado: {manifest.get('generated_at', 'unknown')}")
        print(f"   - Tag: {manifest.get('tag', 'unknown')}")
        print(f"   - Postura: {manifest.get('risk_posture', 'unknown')}")
        print("\n📋 RESULTADO: APTO PARA AUDITORÍA")
        return True
    else:
        print("\n" + "=" * 70)
        print("❌ INTEGRIDAD COMPROMETIDA - CSV ha sido modificado o corrupto")
        print("=" * 70)
        print(f"   - CSV: {csv_filename}")
        print(f"   - Hash esperado: {expected_hash}")
        print(f"   - Hash actual:   {actual_hash}")
        print("\n⚠️ ACCIÓN REQUERIDA:")
        print("   1. Re-exportar CSV canónico desde XLSX original")
        print("   2. Verificar que XLSX no ha sido editado manualmente")
        print("   3. Usar comando de export canónico para regenerar")
        print("\n📋 RESULTADO: RECHAZADO PARA AUDITORÍA")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verificar integridad de CSV canónico contra MANIFEST (bank-ready auditing)"
    )
    parser.add_argument("--manifest", required=True, help="Path al MANIFEST.json")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.manifest):
        print(f"❌ ERROR: MANIFEST no encontrado: {args.manifest}")
        sys.exit(1)
    
    # Verificar
    integrity_ok = verify_manifest(args.manifest)
    
    # Exit code (0=OK, 1=FAIL para CI/CD)
    sys.exit(0 if integrity_ok else 1)


if __name__ == "__main__":
    main()
