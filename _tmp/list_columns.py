"""Check available columns in inference output"""
import pandas as pd
import glob

pattern = "reports/coordinated_inference_executability_v1_des*"
folders = glob.glob(pattern)
latest = max(folders, key=lambda x: x)
excel_path = glob.glob(f"{latest}/decisiones_finales_*.xlsx")[0]

df = pd.read_excel(excel_path)

print("="*80)
print("COLUMNAS DISPONIBLES EN DECISIONES_FINALES")
print("="*80)
print(f"\nArchivo: {excel_path}")
print(f"Total columnas: {len(df.columns)}\n")

# Buscar columnas relacionadas con precio/venta
sale_cols = [col for col in df.columns if any(x in col.lower() for x in ['precio', 'price', 'sale', 'venta', 'npl'])]
ref_cols = [col for col in df.columns if any(x in col.lower() for x in ['valor', 'value', 'ref', 'recovery'])]
exec_cols = [col for col in df.columns if any(x in col.lower() for x in ['executable', 'ejecut', 'mandate', 'insulting', 'acceptance'])]

print("COLUMNAS DE PRECIO/VENTA:")
for col in sale_cols:
    print(f"  - {col}")

print("\nCOLUMNAS DE VALOR/REFERENCIA:")
for col in ref_cols:
    print(f"  - {col}")

print("\nCOLUMNAS DE EJECUTABILIDAD  (nuevas):")
for col in exec_cols:
    print(f"  - {col}")

print("\nTODAS LAS COLUMNAS (alfabético):")
for i, col in enumerate(sorted(df.columns), 1):
    print(f"  {i:3}. {col}")
