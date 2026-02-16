"""
Análisis PC7 SECOND PASS: mandatos tiering + reestructuras + cap voluntarias.

Validación requerida:
1. Tabla mixes (%MANTENER/%REESTRUCTURAR/%VENDER) + KPIs
2. %sale_mandate por postura (DESINV target 20-30%) + desglose Tier1/Tier2
3. Confirmación cap voluntarias <= max_sell_share, mandatos dominan
4. Conteos bloqueo gates: loss_cap, recovery_min, insulting_price
5. 10 casos frontera: PRUD vs BAL, BAL vs DESINV
"""

import pandas as pd
import glob
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

def analyze():
    print("\n" + "="*80)
    print("PC7 SECOND PASS - VALIDACIÓN CALIBRACIÓN FINAL (mandate tiering + restructures)")
    print("="*80 + "\n")

    # Buscar archivos decisiones
    pru_files = glob.glob("reports/*PC7_SECOND_PASS_pru*/decisiones_finales*.xlsx")
    bal_files = glob.glob("reports/*PC7_SECOND_PASS_bal*/decisiones_finales*.xlsx")
    des_files = glob.glob("reports/*PC7_SECOND_PASS_des*/decisiones_finales*.xlsx")
    
    if not (pru_files and bal_files and des_files):
        print("❌ ERROR: No se encontraron archivos decisiones PC7_SECOND_PASS")
        print(f"   PRU: {len(pru_files)} | BAL: {len(bal_files)} | DES: {len(des_files)}")
        return False
    
    pru_path = sorted(pru_files, key=lambda x: Path(x).stat().st_mtime)[-1]
    bal_path = sorted(bal_files, key=lambda x: Path(x).stat().st_mtime)[-1]
    des_path = sorted(des_files, key=lambda x: Path(x).stat().st_mtime)[-1]
    
    print(f"📂 Archivos analizados:")
    print(f"   PRU: {Path(pru_path).parent.name}")
    print(f"   BAL: {Path(bal_path).parent.name}")
    print(f"   DES: {Path(des_path).parent.name}\n")
    
    # Leer datos
    pru_df = pd.read_excel(pru_path, sheet_name="decisiones")
    bal_df = pd.read_excel(bal_path, sheet_name="decisiones")
    des_df = pd.read_excel(des_path, sheet_name="decisiones")
    
    # ========================================
    # 1) MIXES + KPIs
    # ========================================
    print("\n📋 TABLA 1: MIX DE DECISIONES")
    print("-" * 80)
    
    def compute_mix(df, postura):
        total = len(df)
        mantener = (df["Accion_final"] == "MANTENER").sum()
        restruct = (df["Accion_final"] == "REESTRUCTURAR").sum()
        vender = (df["Accion_final"] == "VENDER").sum()
        return {
            "postura": postura,
            "total": total,
            "MANTENER": mantener,
            "REESTRUCTURAR": restruct,
            "VENDER": vender,
            "pct_MANTENER": mantener/total*100,
            "pct_REESTRUCTURAR": restruct/total*100,
            "pct_VENDER": vender/total*100
        }
    
    mix_pru = compute_mix(pru_df, "PRUDENCIAL")
    mix_bal = compute_mix(bal_df, "BALANCEADO")
    mix_des = compute_mix(des_df, "DESINVERSION")
    
    mix_table = pd.DataFrame([mix_pru, mix_bal, mix_des])
    print(mix_table.to_string(index=False))
    
    print("\n📋 TABLA 2: KPIs AGREGADOS (EUR)")
    print("-" * 80)
    
    def compute_kpis(df, postura):
        eva_post = df["EVA_post"].sum()
        eva_delta = df.get("EVA_delta", pd.Series(0, index=df.index)).sum()
        rwa_post = df["RWA_post"].sum()
        cap_rel = df["capital_release_realized"].sum()
        return {
            "postura": postura,
            "EVA_post_total": eva_post,
            "EVA_delta_total": eva_delta,
            "RWA_post_total": rwa_post,
            "capital_release_total": cap_rel
        }
    
    kpi_pru = compute_kpis(pru_df, "PRUDENCIAL")
    kpi_bal = compute_kpis(bal_df, "BALANCEADO")
    kpi_des = compute_kpis(des_df, "DESINVERSION")
    
    kpi_table = pd.DataFrame([kpi_pru, kpi_bal, kpi_des])
    kpi_table["EVA_post_total"] = kpi_table["EVA_post_total"].apply(lambda x: f"{x:,.0f}")
    kpi_table["EVA_delta_total"] = kpi_table["EVA_delta_total"].apply(lambda x: f"{x:,.0f}")
    kpi_table["RWA_post_total"] = kpi_table["RWA_post_total"].apply(lambda x: f"{x:,.0f}")
    kpi_table["capital_release_total"] = kpi_table["capital_release_total"].apply(lambda x: f"{x:,.0f}")
    print(kpi_table.to_string(index=False))
    
    # ========================================
    # 2) MANDATOS (tiering + percentiles)
    # ========================================
    print("\n📋 TABLA 3: MANDATOS DE VENTA (tiering + percentiles)")
    print("-" * 80)
    
    def compute_mandates(df, postura):
        total = len(df)
        vendidas = (df["Accion_final"] == "VENDER").sum()
        mandatos = df.get("sale_mandate", pd.Series(False, index=df.index))
        
        n_mandatos_total = mandatos.sum()
        tier1 = (df.get("sale_mandate_tier", "") == "TIER1_SEVERE").sum()
        tier2 = (df.get("sale_mandate_tier", "") == "TIER2_CAPITAL_PRESSURE").sum()
        
        ventas_mandate = ((df["Accion_final"] == "VENDER") & mandatos).sum()
        ventas_voluntary = vendidas - ventas_mandate
        
        # Block stats
        mandatos_blocked = (mandatos & (df["Accion_final"] != "VENDER")).sum()
        
        return {
            "postura": postura,
            "mandatos_totales": n_mandatos_total,
            "pct_mandatos": n_mandatos_total/total*100,
            "tier1_severe": tier1,
            "tier2_capital_pressure": tier2,
            "ventas_por_mandato": ventas_mandate,
            "ventas_voluntarias": ventas_voluntary,
            "mandatos_bloqueados": mandatos_blocked
        }
    
    mand_pru = compute_mandates(pru_df, "PRUDENCIAL")
    mand_bal = compute_mandates(bal_df, "BALANCEADO")
    mand_des = compute_mandates(des_df, "DESINVERSION")
    
    mand_table = pd.DataFrame([mand_pru, mand_bal, mand_des])
    print(mand_table.to_string(index=False))
    
    # ========================================
    # 3) CAP VOLUNTARIAS (mandatos exentos)
    # ========================================
    print("\n📋 TABLA 4: CAP VENTAS VOLUNTARIAS (mandatos exentos)")
    print("-" * 80)
    
    # Leer max_sell_share de config
    from config import BANK_STRATEGIES, BankProfile
    pru_cap = BANK_STRATEGIES[BankProfile.PRUDENTE].max_sell_share
    bal_cap = BANK_STRATEGIES[BankProfile.BALANCEADO].max_sell_share
    des_cap = BANK_STRATEGIES[BankProfile.DESINVERSION].max_sell_share
    
    def compute_cap_compliance(df, postura, cap):
        total = len(df)
        vendidas = (df["Accion_final"] == "VENDER").sum()
        mandatos = df.get("sale_mandate", pd.Series(False, index=df.index))
        ventas_mandate = ((df["Accion_final"] == "VENDER") & mandatos).sum()
        ventas_voluntary = vendidas - ventas_mandate
        
        cap_max_voluntary = int(total * cap)
        within_cap = ventas_voluntary <= cap_max_voluntary
        
        return {
            "postura": postura,
            "max_sell_share_config": f"{cap:.0%}",
            "ventas_voluntarias": ventas_voluntary,
            "cap_max_voluntary": cap_max_voluntary,
            "ventas_mandato_exentas": ventas_mandate,
            "ventas_totales": vendidas,
            "cap_compliant": "✅ OK" if within_cap else "❌ VIOLATED"
        }
    
    cap_pru = compute_cap_compliance(pru_df, "PRUDENCIAL", pru_cap)
    cap_bal = compute_cap_compliance(bal_df, "BALANCEADO", bal_cap)
    cap_des = compute_cap_compliance(des_df, "DESINVERSION", des_cap)
    
    cap_table = pd.DataFrame([cap_pru, cap_bal, cap_des])
    print(cap_table.to_string(index=False))
    
    # ========================================
    # 4) BLOQUEOS POR GATES
    # ========================================
    print("\n📋 TABLA 5: BLOQUEOS POR GATES DE EJECUTABILIDAD")
    print("-" * 80)
    
    def compute_blocks(df, postura):
        insulting = df.get("sale_insulting_flag", pd.Series(False, index=df.index)).sum()
        loss_cap_fail = (~df.get("sale_within_loss_cap", pd.Series(True, index=df.index))).sum()
        recovery_low = (~df.get("sale_meets_recovery_min", pd.Series(True, index=df.index))).sum()
        restruct_low_accept = ((df.get("restruct_viable", False)) & 
                               (df.get("acceptance_score", 100) < df.get("acceptance_score", 100).max())).sum()
        
        return {
            "postura": postura,
            "insulting_price": insulting,
            "loss_cap_exceeded": loss_cap_fail,
            "recovery_too_low": recovery_low,
            "acceptance_score_low": restruct_low_accept
        }
    
    block_pru = compute_blocks(pru_df, "PRUDENCIAL")
    block_bal = compute_blocks(bal_df, "BALANCEADO")
    block_des = compute_blocks(des_df, "DESINVERSION")
    
    block_table = pd.DataFrame([block_pru, block_bal, block_des])
    print(block_table.to_string(index=False))
    
    # ========================================
    # 5) CASOS FRONTERA
    # ========================================
    print("\n📋 TABLA 6: CASOS FRONTERA (Top 10 diferencias PRUD ≠ BAL)")
    print("-" * 80)
    
    # Merge PRU + BAL
    merged = pru_df[["loan_id", "EAD", "Accion_final", "EVA_post", "capital_release_realized"]].merge(
        bal_df[["loan_id", "Accion_final", "EVA_post", "capital_release_realized"]],
        on="loan_id", suffixes=("_pru", "_bal")
    )
    
    # Filter donde acciones difieren
    merged_diff = merged[merged["Accion_final_pru"] != merged["Accion_final_bal"]].copy()
    
    # Importance score simple: EAD * |EVA_delta|
    merged_diff["importance_score"] = merged_diff["EAD"] * abs(
        merged_diff["EVA_post_pru"] - merged_diff["EVA_post_bal"]
    )
    
    # Top 10
    top10_pru_bal = merged_diff.nlargest(10, "importance_score")[[
        "loan_id", "EAD", "Accion_final_pru", "EVA_post_pru", "capital_release_realized_pru",
        "Accion_final_bal", "EVA_post_bal", "capital_release_realized_bal", "importance_score"
    ]]
    
    top10_pru_bal["EAD"] = top10_pru_bal["EAD"].apply(lambda x: f"{x:,.0f}")
    top10_pru_bal["EVA_post_pru"] = top10_pru_bal["EVA_post_pru"].apply(lambda x: f"{x:,.0f}")
    top10_pru_bal["capital_release_realized_pru"] = top10_pru_bal["capital_release_realized_pru"].apply(lambda x: f"{x:,.0f}")
    top10_pru_bal["EVA_post_bal"] = top10_pru_bal["EVA_post_bal"].apply(lambda x: f"{x:,.0f}")
    top10_pru_bal["capital_release_realized_bal"] = top10_pru_bal["capital_release_realized_bal"].apply(lambda x: f"{x:,.0f}")
    top10_pru_bal["importance_score"] = top10_pru_bal["importance_score"].apply(lambda x: f"{x:,.0f}")
    
    print(top10_pru_bal.to_string(index=False))
    
    print("\n📋 TABLA 7: CASOS FRONTERA (Top 10 diferencias BAL ≠ DESINV)")
    print("-" * 80)
    
    # Merge BAL + DESINV
    merged_bal_des = bal_df[["loan_id", "EAD", "Accion_final", "EVA_post", "sale_mandate"]].merge(
        des_df[["loan_id", "Accion_final", "EVA_post", "sale_mandate", "sale_mandate_tier"]],
        on="loan_id", suffixes=("_bal", "_des")
    )
    
    # Filter donde DESINV vende por mandato y BAL no
    merged_bal_des_diff = merged_bal_des[
        (merged_bal_des["Accion_final_des"] == "VENDER") & 
        (merged_bal_des["sale_mandate_des"] == True) &
        (merged_bal_des["Accion_final_bal"] != "VENDER")
    ].copy()
    
    # Importance score: EAD (mandatos más grandes primero)
    merged_bal_des_diff["importance_score"] = merged_bal_des_diff["EAD"]
    
    # Top 10
    if len(merged_bal_des_diff) > 0:
        top10_bal_des = merged_bal_des_diff.nlargest(10, "importance_score")[[
            "loan_id", "EAD", "Accion_final_bal", "EVA_post_bal",
            "Accion_final_des", "EVA_post_des", "sale_mandate_tier", "importance_score"
        ]]
        
        top10_bal_des.rename(columns={"sale_mandate_tier": "sale_mandate_tier_des"}, inplace=True)
    else:
        top10_bal_des = pd.DataFrame(columns=[
            "loan_id", "EAD", "Accion_final_bal", "EVA_post_bal",
            "Accion_final_des", "EVA_post_des", "sale_mandate_tier_des", "importance_score"
        ])
    
    top10_bal_des["EAD"] = top10_bal_des["EAD"].apply(lambda x: f"{x:,.0f}")
    top10_bal_des["EVA_post_bal"] = top10_bal_des["EVA_post_bal"].apply(lambda x: f"{x:,.0f}")
    top10_bal_des["EVA_post_des"] = top10_bal_des["EVA_post_des"].apply(lambda x: f"{x:,.0f}")
    top10_bal_des["importance_score"] = top10_bal_des["importance_score"].apply(lambda x: f"{x:,.0f}")
    
    print(top10_bal_des.to_string(index=False))
    
    # ========================================
    # VALIDACIÓN CRÍTICA
    # ========================================
    print("\n" + "="*80)
    print("✅ VALIDACIÓN DE CALIBRACIÓN PC7 SECOND PASS")
    print("="*80 + "\n")
    
    # Check 1: DESINV mandatos 20-30%
    des_mandate_pct = mand_des["pct_mandatos"]
    if 20 <= des_mandate_pct <= 30:
        print(f"✅ DESINV mandatos: {des_mandate_pct:.1f}% (dentro rango 20-30%)")
    else:
        print(f"⚠️  DESINV mandatos: {des_mandate_pct:.1f}% (fuera de rango 20-30%)")
    
    # Check 2: Reestructuras no triviales en PRUD/BAL
    pru_restruct_pct = mix_pru["pct_REESTRUCTURAR"]
    bal_restruct_pct = mix_bal["pct_REESTRUCTURAR"]
    if pru_restruct_pct >= 5 and bal_restruct_pct >= 5:
        print(f"✅ Reestructuras viables: PRUD={pru_restruct_pct:.1f}%, BAL={bal_restruct_pct:.1f}% (>5%)")
    else:
        print(f"⚠️  Reestructuras: PRUD={pru_restruct_pct:.1f}%, BAL={bal_restruct_pct:.1f}% (objetivo >5%)")
    
    # Check 3: PRUD más conservador que BAL
    if mix_pru["pct_MANTENER"] > mix_bal["pct_MANTENER"]:
        print(f"✅ PRUD más conservador: {mix_pru['pct_MANTENER']:.1f}% mantener vs {mix_bal['pct_MANTENER']:.1f}% BAL")
    else:
        print(f"⚠️  PRUD NO más conservador que BAL")
    
    # Check 4: Cap voluntarias OK
    if cap_pru["cap_compliant"] == "✅ OK" and cap_bal["cap_compliant"] == "✅ OK" and cap_des["cap_compliant"] == "✅ OK":
        print(f"✅ Cap voluntarias respetado en las 3 posturas (mandatos exentos)")
    else:
        print(f"⚠️  Cap voluntarias violado en alguna postura")
    
    # Check 5: Diferenciación PRUD vs BAL
    n_diff_pru_bal = len(merged_diff)
    if n_diff_pru_bal >= 50:
        print(f"✅ Diferenciación PRUD vs BAL: {n_diff_pru_bal} préstamos con decisiones distintas")
    else:
        print(f"⚠️  Poca diferenciación PRUD vs BAL: solo {n_diff_pru_bal} préstamos difieren")
    
    # Check 6: Mandatos en BAL/DESINV tienen tier
    des_tier1 = mand_des["tier1_severe"]
    des_tier2 = mand_des["tier2_capital_pressure"]
    if des_tier1 > 0 and des_tier2 > 0:
        print(f"✅ DESINV mandatos con tiering: TIER1={des_tier1}, TIER2={des_tier2}")
    else:
        print(f"⚠️  DESINV mandatos sin tiering adecuado")
    
    print("\n" + "="*80)
    print("RESUMEN COMPLETADO")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = analyze()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
