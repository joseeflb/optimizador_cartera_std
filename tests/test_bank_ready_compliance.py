#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_bank_ready_compliance.py

Test automático que valida criterios "bank-ready":
- CRÍTICO: Restricciones duras PTI/DSCR en reestructuraciones
- CRÍTICO: Fire-sale protections en ventas
- IMPORTANTE: Auditoría completa (reason codes, convergencia)
- IMPORTANTE: Posturas diferenciadas medibles
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def test_restructure_pti_dscr(df: pd.DataFrame, postura: str) -> Dict[str, Any]:
    """
    Valida que TODAS las reestructuraciones tengan PTI_post o DSCR_post calculados.
    """
    restruct = df[df['Accion_final'] == 'REESTRUCTURAR'].copy()
    
    results = {
        "test": "PTI/DSCR en reestructuras",
        "postura": postura,
        "total_restruct": len(restruct),
        "passed": True,
        "errors": [],
    }
    
    if len(restruct) == 0:
        return results
    
    # Verificar que al menos uno de PTI_post o DSCR_post esté presente
    pti_missing = restruct['PTI_post'].isna().sum()
    dscr_missing = restruct['DSCR_post'].isna().sum()
    
    # ❌ CRÍTICO: PTI_post debería estar calculado para retail/mortgage
    retail_segs = ['MORTGAGE', 'RETAIL', 'CONSUMER']
    retail_restruct = restruct[restruct['segment'].isin(retail_segs)]
    if len(retail_restruct) > 0:
        retail_pti_missing = retail_restruct['PTI_post'].isna().sum()
        if retail_pti_missing > 0:
            results["passed"] = False
            results["errors"].append(
                f"⚠️ CRÍTICO: {retail_pti_missing}/{len(retail_restruct)} reestructuras retail/mortgage sin PTI_post"
            )
    
    # ❌ CRÍTICO: DSCR_post debería estar calculado para corporate
    corp_segs = ['CORPORATE', 'SME', 'BANK', 'SOVEREIGN']
    corp_restruct = restruct[restruct['segment'].isin(corp_segs)]
    if len(corp_restruct) > 0:
        corp_dscr_missing = corp_restruct['DSCR_post'].isna().sum()
        if corp_dscr_missing > 0:
            results["passed"] = False
            results["errors"].append(
                f"⚠️ CRÍTICO: {corp_dscr_missing}/{len(corp_restruct)} reestructuras corporate sin DSCR_post"
            )
    
    # Validar rangos razonables para los que SÍ tienen valores
    pti_values = pd.to_numeric(restruct['PTI_post'], errors='coerce').dropna()
    if len(pti_values) > 0:
        pti_too_low = (pti_values < 0.10).sum()
        pti_too_high = (pti_values > 0.70).sum()
        
        if pti_too_low > 0:
            results["errors"].append(f"⚠ {pti_too_low} reestructuras con PTI_post < 0.10 (inusualmente bajo)")
        if pti_too_high > 0:
            results["passed"] = False
            results["errors"].append(f"⚠️ {pti_too_high} reestructuras con PTI_post > 0.70 (insostenible)")
    
    dscr_values = pd.to_numeric(restruct['DSCR_post'], errors='coerce').dropna()
    if len(dscr_values) > 0:
        dscr_too_low = (dscr_values < 1.00).sum()
        
        if dscr_too_low > 0:
            results["passed"] = False
            results["errors"].append(f"⚠️ CRÍTICO: {dscr_too_low} reestructuras con DSCR_post < 1.00 (inviable)")
    
    return results


def test_fire_sale_protections(df: pd.DataFrame, postura: str) -> Dict[str, Any]:
    """
    Valida que fire-sale protections funcionen correctamente.
    """
    sells = df[df['Accion_final'] == 'VENDER'].copy()
    
    results = {
        "test": "Fire-sale protections",
        "postura": postura,
        "total_sells": len(sells),
        "passed": True,
        "errors": [],
    }
    
    if len(sells) == 0:
        return results
    
    # Para posturas prudencial/balanceado, NO debería haber fire-sales ejecutados
    if postura in ['prudencial', 'balanceado']:
        fire_sales = sells[sells.get('Fire_Sale', False) == True]
        if len(fire_sales) > 0:
            results["passed"] = False
            results["errors"].append(
                f"⚠️ CRÍTICO: {len(fire_sales)} fire-sales ejecutados en postura {postura} (debería bloquearse)"
            )
        
        # Verificar que hay bloqueos si hubo triggers
        if 'Sell_Blocked' in sells.columns:
            blocked = sells[sells['Sell_Blocked'] == True]
            results[f"fire_sales_blocked"] = len(blocked)
    
    # Validar precios razonables
    price_ratio = pd.to_numeric(sells.get('Price_to_EAD', np.nan), errors='coerce')
    if price_ratio.notna().any():
        very_low_price = (price_ratio < 0.05).sum()
        if very_low_price > 0 and postura != 'desinversion':
            results["errors"].append(
                f"⚠ {very_low_price} ventas con Price/EAD < 5% (pérdida severa)"
            )
    
    return results


def test_auditability(df: pd.DataFrame, postura: str) -> Dict[str, Any]:
    """
    Valida que todos los campos de auditoría estén poblados.
    """
    results = {
        "test": "Auditabilidad (Reason_Code, Convergencia_Caso)",
        "postura": postura,
        "total_loans": len(df),
        "passed": True,
        "errors": [],
    }
    
    # Verificar Reason_Code
    if 'Reason_Code' in df.columns:
        rc_missing = df['Reason_Code'].isna().sum()
        rc_empty = (df['Reason_Code'].astype(str).str.strip() == '').sum()
        
        if rc_missing + rc_empty > 0:
            results["passed"] = False
            results["errors"].append(
                f"⚠️ IMPORTANTE: {rc_missing + rc_empty}/{len(df)} préstamos sin Reason_Code"
            )
    else:
        results["passed"] = False
        results["errors"].append("⚠️ CRÍTICO: Columna 'Reason_Code' no existe")
    
    # Verificar Convergencia_Caso
    if 'Convergencia_Caso' in df.columns:
        conv_missing = df['Convergencia_Caso'].isna().sum()
        conv_empty = (df['Convergencia_Caso'].astype(str).str.strip() == '').sum()
        
        if conv_missing + conv_empty > 0:
            results["passed"] = False
            results["errors"].append(
                f"⚠️ IMPORTANTE: {conv_missing + conv_empty}/{len(df)} préstamos sin Convergencia_Caso"
            )
    else:
        results["passed"] = False
        results["errors"].append("⚠️ CRÍTICO: Columna 'Convergencia_Caso' no existe")
    
    return results


def test_posture_divergence(summaries: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Valida que las 3 posturas tengan mix de decisiones claramente diferenciados.
    """
    results = {
        "test": "Divergencia de posturas",
        "passed": True,
        "errors": [],
        "decision_mix": {},
    }
    
    for postura, df in summaries.items():
        if df is None or len(df) == 0:
            continue
        
        mix = df['Accion_final'].value_counts(normalize=True) * 100
        results["decision_mix"][postura] = mix.to_dict()
    
    # Validar diferencias esperadas
    if len(results["decision_mix"]) >= 3:
        # DESINVERSION debe vender > 70%
        if 'desinversion' in results["decision_mix"]:
            sell_pct = results["decision_mix"]['desinversion'].get('VENDER', 0)
            if sell_pct < 70:
                results["passed"] = False
                results["errors"].append(
                    f"⚠️ DESINVERSION solo vende {sell_pct:.1f}% (esperado >70%)"
                )
        
        # PRUDENCIAL debe mantener/reestructurar > 90%
        if 'prudencial' in results["decision_mix"]:
            sell_pct = results["decision_mix"]['prudencial'].get('VENDER', 0)
            if sell_pct > 10:
                results["passed"] = False
                results["errors"].append(
                    f"⚠️ PRUDENCIAL vende {sell_pct:.1f}% (esperado <10%)"
                )
    
    return results


def run_bank_ready_tests(decisiones_paths: Dict[str, str]) -> bool:
    """
    Ejecuta todos los tests de compliance "bank-ready".
    
    Args:
        decisiones_paths: Dict con {postura: path_to_decisiones_explicadas.xlsx}
    
    Returns:
        True si TODOS los tests pasan
    """
    print("="*80)
    print("VALIDACIÓN BANK-READY - CRITERIOS DE ACEPTACIÓN")
    print("="*80)
    
    all_passed = True
    summaries = {}
    
    for postura, path in decisiones_paths.items():
        print(f"\n📂 Cargando decisiones: {postura}")
        try:
            df = pd.read_excel(path)
            summaries[postura] = df
            print(f"   ✓ Cargado: {len(df)} préstamos")
        except Exception as e:
            print(f"   ✗ Error cargando {path}: {e}")
            all_passed = False
            continue
        
        # Test 1: PTI/DSCR
        result = test_restructure_pti_dscr(df, postura)
        print(f"\n🔍 {result['test']} [{postura}]")
        if result['passed']:
            print(f"   ✅ PASS - {result['total_restruct']} reestructuras validadas")
        else:
            print(f"   ❌ FAIL")
            for error in result['errors']:
                print(f"      {error}")
            all_passed = False
        
        # Test 2: Fire-sale
        result = test_fire_sale_protections(df, postura)
        print(f"\n🔍 {result['test']} [{postura}]")
        if result['passed']:
            print(f"   ✅ PASS - {result['total_sells']} ventas validadas")
            if 'fire_sales_blocked' in result:
                print(f"      ℹ️  Fire-sales bloqueados: {result['fire_sales_blocked']}")
        else:
            print(f"   ❌ FAIL")
            for error in result['errors']:
                print(f"      {error}")
            all_passed = False
        
        # Test 3: Auditabilidad
        result = test_auditability(df, postura)
        print(f"\n🔍 {result['test']} [{postura}]")
        if result['passed']:
            print(f"   ✅ PASS - {result['total_loans']} préstamos con auditoría completa")
        else:
            print(f"   ❌ FAIL")
            for error in result['errors']:
                print(f"      {error}")
            all_passed = False
    
    # Test 4: Divergencia de posturas
    if len(summaries) >= 2:
        result = test_posture_divergence(summaries)
        print(f"\n🔍 {result['test']}")
        
        for postura, mix in result["decision_mix"].items():
            print(f"   {postura.upper()}:")
            for action, pct in mix.items():
                print(f"      {action}: {pct:.1f}%")
        
        if result['passed']:
            print(f"   ✅ PASS - Posturas claramente diferenciadas")
        else:
            print(f"   ❌ FAIL")
            for error in result['errors']:
                print(f"      {error}")
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON - SISTEMA BANK-READY")
    else:
        print("❌ ALGUNOS TESTS FALLARON - REVISAR CRITERIOS")
    print(f"{'='*80}\n")
    
    return all_passed


if __name__ == "__main__":
    # Ejemplo: validar últimas inferencias
    from pathlib import Path
    
    REPORTS_DIR = ROOT / "reports"
    
    # Buscar carpetas más recientes por postura
    posturas_paths = {}
    
    for postura in ['prudencial', 'balanceado', 'desinversion']:
        pattern = f"*{postura}*/decisiones_explicadas.xlsx"
        matching = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if matching:
            posturas_paths[postura] = str(matching[0])
            print(f"Found {postura}: {matching[0].parent.name}")
        else:
            print(f"⚠️  No se encontraron decisiones para postura '{postura}'")
    
    if len(posturas_paths) < 2:
        print("\n❌ No hay suficientes inferencias para validar. Ejecuta inferencia primero.")
        sys.exit(1)
    
    success = run_bank_ready_tests(posturas_paths)
    sys.exit(0 if success else 1)
