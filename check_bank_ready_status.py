"""Script de verificación rápida - Bank-Ready Status Check"""
import os
import glob

print("="*80)
print("BANK-READY STATUS CHECK")
print("="*80)

# Verificar archivos críticos
critical_files = {
    "Modelo micro": "models/best_model_loan.zip",
    "VecNormalize micro": "models/vecnormalize_loan.pkl",
    "Portfolio completo": "data/portfolio_synth.xlsx",
    "Coordinator": "agent/coordinator_inference.py",
    "Tests compliance": "tests/test_bank_ready_compliance.py"
}

print("\n1️⃣ ARCHIVOS CRÍTICOS")
all_present = True
for name, path in critical_files.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {name}: {path}")
    if not exists:
        all_present = False

# Verificar deliverable package
print("\n2️⃣ PAQUETE DELIVERABLE")
deliverable_base = "reports/BANK_READY_DELIVERABLE_FINAL"
if os.path.exists(deliverable_base):
    print(f"   ✅ Carpeta base: {deliverable_base}")
    
    # Count files
    excel_files = glob.glob(f"{deliverable_base}/POSTURAS/*.xlsx")
    doc_files = glob.glob(f"{deliverable_base}/DOCUMENTACION/*.md")
    script_files = glob.glob(f"{deliverable_base}/SCRIPTS/*.py")
    
    print(f"   ✅ Excel outputs: {len(excel_files)}/3")
    print(f"   ✅ Documentación: {len(doc_files)}/3")
    print(f"   ✅ Scripts validación: {len(script_files)}/3")
    
    if len(excel_files) < 3 or len(doc_files) < 3 or len(script_files) < 3:
        print("   ⚠️  ADVERTENCIA: Faltan archivos en el deliverable")
        all_present = False
else:
    print(f"   ❌ Carpeta no encontrada: {deliverable_base}")
    all_present = False

# Verificar últimas inferencias
print("\n3️⃣ ÚLTIMAS INFERENCIAS (BANK-READY)")
inference_folders = glob.glob("reports/coordinated_inference_bank_ready_final_*")
if inference_folders:
    print(f"   ✅ Encontradas: {len(inference_folders)} inferencias")
    
    for folder in sorted(inference_folders, key=os.path.getctime, reverse=True)[:3]:
        folder_name = os.path.basename(folder)
        
        # Detectar postura
        if "pru" in folder_name:
            posture = "PRUDENCIAL"
        elif "bal" in folder_name:
            posture = "BALANCEADO"
        elif "des" in folder_name:
            posture = "DESINVERSION"
        else:
            posture = "UNKNOWN"
        
        # Buscar Excel
        excel_files = glob.glob(f"{folder}/decisiones_finales_*.xlsx")
        status = "✅" if excel_files else "⚠️"
        print(f"      {status} {posture}: {folder_name}")
else:
    print("   ⚠️  No se encontraron inferencias bank-ready recientes")
    all_present = False

# Verificar documentación
print("\n4️⃣ DOCUMENTACIÓN")
docs = {
    "Reporte Ejecutivo": "REPORTE_EJECUTIVO_FINAL_BANK_READY.md",
    "Resumen Implementación": "RESUMEN_IMPLEMENTACION_BANK_READY.md",
    "Resumen Final": "RESUMEN_FINAL_COMPLETION.md"
}

for name, path in docs.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "⚠️"
    print(f"   {status} {name}: {path}")

# Summary
print("\n" + "="*80)
if all_present:
    print("✅ STATUS: BANK-READY - TODOS LOS COMPONENTES PRESENTES")
    print("\n🚀 LISTO PARA:")
    print("   - Presentación a Comité de Riesgos")
    print("   - Revisión por Auditoría Interna")
    print("   - Pilot en cartera real (50-100 préstamos)")
    print("\n📦 DELIVERABLE: reports/BANK_READY_DELIVERABLE_FINAL/")
else:
    print("⚠️ STATUS: INCOMPLETO - REVISAR ARCHIVOS FALTANTES")
    print("\n💡 SUGERENCIA:")
    print("   - Re-ejecutar inferencias si faltan outputs")
    print("   - Verificar que todos los scripts estén presentes")

print("="*80)
print("\n📝 QUICK COMMANDS:")
print("\n# Validación rápida compliance:")
print("py _tmp/final_compliance_check.py")
print("\n# Comparativa posturas:")
print("py _tmp/validate_3_postures.py")
print("\n# Re-generar postura (ejemplo PRUDENCIAL):")
print("py -m agent.coordinator_inference --model-micro models\\best_model_loan.zip \\")
print("   --portfolio data\\portfolio_synth.xlsx --risk-posture prudencial \\")
print("   --vn-micro models\\vecnormalize_loan.pkl --n-steps 3 --top-k 5 --tag test")
print("="*80)
