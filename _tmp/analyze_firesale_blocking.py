"""Verificar si hay fire-sales bloqueados en el output"""
import pandas as pd

output_file = r'reports\coordinated_inference_bank_ready_v1_20260215_192825_prudencial\decisiones_finales_prudencial.xlsx'
df = pd.read_excel(output_file)

print("="*80)
print("ANÁLISIS DE FIRE-SALES")
print("="*80)

print(f"\n1. Total préstamos: {len(df)}")

# Fire-sale detection
if 'Fire_Sale' in df.columns:
    print(f"\n2. Fire_Sale:")
    print(df['Fire_Sale'].value_counts())

if 'FireSale_Triggered' in df.columns:
    print(f"\n3. FireSale_Triggered:")
    print(df['FireSale_Triggered'].value_counts())

if 'Sell_Blocked' in df.columns:
    print(f"\n4. Sell_Blocked:")
    print(df['Sell_Blocked'].value_counts())

# Decisiones
print(f"\n5. Accion_final:")
print(df['Accion_final'].value_counts())

# Acción micro vs final
if 'Accion_micro' in df.columns:
    print(f"\n6. Accion_micro vs Accion_final (divergencia):")
    divergent = df[df['Accion_micro'] != df['Accion_final']]
    print(f"   Total divergencias: {len(divergent)}")
    
    if len(divergent) > 0:
        print(f"\n   Mix de divergencias:")
        print(divergent[['Accion_micro', 'Accion_final']].value_counts())
        
        # Ver si algunos eran VENDER → MANTENER por fire-sale
        vender_to_mantener = divergent[(divergent['Accion_micro'] == 'VENDER') & 
                                        (divergent['Accion_final'] == 'MANTENER')]
        print(f"\n   VENDER→MANTENER: {len(vender_to_mantener)}")
        
        if len(vender_to_mantener) > 0:
            print(f"\n   Muestra VENDER→MANTENER (primeras 5):")
            cols = ['loan_id', 'segment', 'Accion_micro', 'Accion_final', 'Fire_Sale', 
                    'FireSale_Triggered', 'Sell_Blocked', 'restruct_viable', 
                    'case_status', 'override_reason']
            cols = [c for c in cols if c in vender_to_mantener.columns]
            print(vender_to_mantener[cols].head().to_string())

# Ver si restruct_viable=False está presente
if 'restruct_viable' in df.columns:
    print(f"\n7. restruct_viable:")
    print(df['restruct_viable'].value_counts())
    
    # Casos críticos: Fire-sale + restruct_viable=False
    if 'Fire_Sale' in df.columns:
        critical = df[(df['Fire_Sale'] == True) & (df['restruct_viable'] == False)]
        print(f"\n8. CASOS CRÍTICOS (Fire_Sale=True + restruct_viable=False): {len(critical)}")
        
        if len(critical) > 0:
            print(f"   Accion_final:")
            print(critical['Accion_final'].value_counts())
            
            print(f"\n   case_status:")
            print(critical['case_status'].value_counts())

print("\n\n✅ Análisis completado")
