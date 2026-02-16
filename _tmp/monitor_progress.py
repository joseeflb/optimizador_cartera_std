"""Monitor de progreso de inferencias en background"""
import os
import glob
import time
from datetime import datetime

print("="*80)
print("MONITOR DE INFERENCIAS BANK-READY")
print("="*80)

# Buscar carpetas de inferencia recientes
pattern = "reports/coordinated_inference_bank_ready_final_*"
folders = glob.glob(pattern)

if not folders:
    print("\n⏳ Esperando que inicien las inferencias...")
else:
    folders_sorted = sorted(folders, key=os.path.getctime, reverse=True)
    
    print(f"\n📂 Inferencias encontradas: {len(folders_sorted)}")
    
    for folder in folders_sorted[:6]:  # últimas 6
        folder_name = os.path.basename(folder)
        ctime = os.path.getctime(folder)
        time_str = datetime.fromtimestamp(ctime).strftime("%H:%M:%S")
        
        # Determinar postura
        if "pru" in folder_name.lower():
            posture = "PRUDENCIAL"
        elif "bal" in folder_name.lower():
            posture = "BALANCEADO"
        elif "des" in folder_name.lower():
            posture = "DESINVERSION"
        else:
            posture = "UNKNOWN"
        
        # Buscar archivo de decisiones
        excel_path = os.path.join(folder, f"decisiones_finales_{posture.lower()}.xlsx")
        if not os.path.exists(excel_path):
            # Intentar otros nombres
            excel_candidates = glob.glob(os.path.join(folder, "decisiones_finales_*.xlsx"))
            if excel_candidates:
                excel_path = excel_candidates[0]
                status = "✅ COMPLETO"
            else:
                status = "⏳ EN PROGRESO"
        else:
            status = "✅ COMPLETO"
        
        print(f"\n   {status} | {posture:15} | {time_str} | {folder_name}")
        
        if status == "✅ COMPLETO":
            try:
                import pandas as pd
                df = pd.read_excel(excel_path)
                print(f"             → {len(df)} préstamos procesados")
                
                # Mix rápido
                if 'Accion_final' in df.columns:
                    mix = df['Accion_final'].value_counts()
                    for action, count in mix.items():
                        pct = count/len(df)*100
                        print(f"             → {action}: {count} ({pct:.1f}%)")
            except Exception as e:
                print(f"             → Error leyendo output: {e}")

print("\n" + "="*80)
