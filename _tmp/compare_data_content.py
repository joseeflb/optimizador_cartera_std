"""Compara contenido de datos (no binario) entre run1 y run2"""
import pandas as pd
import numpy as np

print("="*80)
print("COMPARACIÓN DE DATOS (RUN1 vs RUN2)")
print("="*80)

posturas = ['PRUDENCIAL', 'BALANCEADO', 'DESINVERSION']

all_identical = True

for posture in posturas:
    print(f"\n--- {posture} ---")
    
    try:
        # Cargar ambos
        file1 = f"reports/BANK_READY_DELIVERABLE_FINAL_run1/POSTURAS/DECISIONES_{posture}_500loans.xlsx"
        file2 = f"reports/BANK_READY_DELIVERABLE_FINAL/POSTURAS/DECISIONES_{posture}_500loans.xlsx"
        
        df1 = pd.read_excel(file1)
        df2 = pd.read_excel(file2)
        
        print(f"   RUN1: {len(df1)} filas, {len(df1.columns)} columnas")
        print(f"   RUN2: {len(df2)} filas, {len(df2.columns)} columnas")
        
        # Comparar dimensiones
        if df1.shape != df2.shape:
            print(f"   ❌ DIMENSIONES DIFERENTES: {df1.shape} vs {df2.shape}")
            all_identical = False
            continue
        
        # Comparar loan_ids (debe ser igual)
        if not df1['loan_id'].equals(df2['loan_id']):
            print(f"   ❌ loan_ids DIFERENTES")
            all_identical = False
            continue
        
        # Comparar columnas
        if not set(df1.columns) == set(df2.columns):
            print(f"   ❌ COLUMNAS DIFERENTES")
            dif = set(df1.columns).symmetric_difference(set(df2.columns))
            print(f"      Diferencias: {dif}")
            all_identical = False
            continue
        
        # Comparar valores celda por celda
        differences = []
        for col in df1.columns:
            # Para columnas numéricas, usar tolerancia
            if df1[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if not np.allclose(df1[col].fillna(0), df2[col].fillna(0), rtol=1e-9, atol=1e-9, equal_nan=True):
                    # Count exact differences
                    mask = ~np.isclose(df1[col].fillna(0), df2[col].fillna(0), rtol=1e-9, atol=1e-9, equal_nan=True)
                    n_diff = mask.sum()
                    if n_diff > 0:
                        differences.append(f"{col} ({n_diff} difs)")
            else:
                # String comparison
                if not df1[col].equals(df2[col]):
                    mask = df1[col] != df2[col]
                    n_diff = mask.sum()
                    differences.append(f"{col} ({n_diff} difs)")
        
        if differences:
            print(f"   ⚠️  DIFERENCIAS EN {len(differences)} COLUMNAS:")
            for diff in differences[:10]:  # Show first 10
                print(f"      - {diff}")
            if len(differences) > 10:
                print(f"      ... y {len(differences)-10} más")
            all_identical = False
        else:
            print(f"   ✅ DATOS IDÉNTICOS (tolerancia 1e-9)")
    
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        all_identical = False

print("\n" + "="*80)
if all_identical:
    print("✅ TODOS LOS DATOS SON IDÉNTICOS")
    print("\n💡 Los hashes binarios difieren por metadata del Excel (timestamps, etc.)")
    print("   pero el CONTENIDO DE DATOS es idéntico → REPRODUCIBILIDAD OK")
else:
    print("❌ DATOS DIFERENTES - REVISAR DETERMINISMO")
    print("\n⚠️  Posibles causas:")
    print("   - Semillas no fijadas correctamente")
    print("   - Operaciones no-deterministas en RL/optimización")
    print("   - Timestamps incluidos en los datos")
print("="*80)
