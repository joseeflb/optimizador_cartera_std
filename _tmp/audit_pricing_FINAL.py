"""Audit de pricing - Validar unidades y escala de sale_price vs value_ref"""
import pandas as pd
import numpy as np
import glob

print("="*80)
print("AUDIT DE PRICING: UNIDADES Y ESCALA")
print("="*80)

# Cargar inferencia DESINVERSION
pattern = "reports/coordinated_inference_executability_v1_des*"
folders = glob.glob(pattern)
if not folders:
    print("ERROR: No se encontro inferencia DESINVERSION")
    exit(1)

latest = max(folders, key=lambda x: x)
excel_path = glob.glob(f"{latest}/decisiones_finales_*.xlsx")[0]
df = pd.read_excel(excel_path)

print(f"\n[INFO] Archivo: {excel_path}")
print(f"[INFO] Total prestamos: {len(df)}\n")

# Extraer columnas relevantes
ead = pd.to_numeric(df["EAD"], errors="coerce")
sale_price = pd.to_numeric(df["precio_optimo"], errors="coerce")
valor_ref = pd.to_numeric(df["valor_referencia"], errors="coerce")
book_value = pd.to_numeric(df["book_value"], errors="coerce")
recovery_sale = pd.to_numeric(df["recommended_sale_recovery"], errors="coerce")
sale_insulting = df["sale_insulting_flag"] if "sale_insulting_flag" in df.columns else None

print("="*80)
print("1. DISTRIBUCION DE VALORES ABSOLUTOS")
print("="*80)

def print_stats(series, name):
    clean = series.dropna()
    if len(clean) == 0:
        print(f"\n{name}: SIN DATOS")
        return
    print(f"\n{name}:")
    print(f"  Count: {len(clean)} | NaN: {series.isna().sum()}")
    print(f"  Min:     {clean.min():>15,.2f} EUR")
    print(f"  P5:      {clean.quantile(0.05):>15,.2f} EUR")
    print(f"  P25:     {clean.quantile(0.25):>15,.2f} EUR")
    print(f"  Median:  {clean.quantile(0.50):>15,.2f} EUR")
    print(f"  P75:     {clean.quantile(0.75):>15,.2f} EUR")
    print(f"  P95:     {clean.quantile(0.95):>15,.2f} EUR")
    print(f"  Max:     {clean.max():>15,.2f} EUR")

print_stats(ead, "EAD")
print_stats(book_value, "BOOK_VALUE")
print_stats(sale_price, "PRECIO_OPTIMO (precio simulado NPL)")
print_stats(valor_ref, "VALOR_REFERENCIA")

print("\n" + "="*80)
print("2. RATIOS CRITICOS")
print("="*80)

# Ratio sale_price / valor_ref
ratio = (sale_price / valor_ref).replace([np.inf, -np.inf], np.nan)
print_stats(ratio, "RATIO sale_price/valor_ref")

# Ratio sale_price / EAD
ratio_ead = (sale_price / ead).replace([np.inf, -np.inf], np.nan)
print_stats(ratio_ead, "RATIO sale_price/EAD")

# Ratio sale_price / book
ratio_book = (sale_price / book_value).replace([np.inf, -np.inf], np.nan)
print_stats(ratio_book, "RATIO sale_price/book_value")

print("\n" + "="*80)
print("3. DIAGNOSIS: PROBLEMA DE ESCALA")
print("="*80)

median_ratio = ratio.median()
print (f"\nRATIO MEDIANO sale_price/valor_ref = {median_ratio:.4f} ({median_ratio*100:.2f}%)")

if median_ratio < 0.15:
    print("\n[ALERTA CRITICA] Ratio < 15%")
    print("  Causas posibles:")
    print("  1) FLOOR_RATIO muy alto (0.50/0.40/0.30) para mercado NPL realista")
    print("  2) valor_ref inflado (usa nominal vs sale_price usa recovery)")
    print("  3) price_simulator con descuentos excesivos")
elif median_ratio < 0.30:
    print("\n[WARNING]  Ratio 15-30% - Mercado NPL bajo pero posible")
else:
    print("\n[OK] Ratio >30% - Coherente con mercado NPL normal")

print("\n" + "="*80)
print("4. FLOOR_RATIOS CONFIGURADOS vs REALIDAD")
print("="*80)

floor_ratios = {"PRUDENCIAL": 0.50, "BALANCEADO": 0.40, "DESINVERSION": 0.30}

print("\nFLOORS ACTUALES:")
for posture, floor in floor_ratios.items():
    print(f"  {posture:15}: {floor:.0%}")

print(f"\nEJEMPLO DESINVERSION (floor=30%):")
print(f"  Valor_ref mediano:     {valor_ref.median():>12,.0f} EUR")
print(f"  Precio minimo (30%):   {valor_ref.median()*0.30:>12,.0f} EUR")
print(f"  Precio simulado real:  {sale_price.median():>12,.0f} EUR")
gap = valor_ref.median()*0.30 - sale_price.median()
print(f"  GAP:                   {gap:>12,.0f} EUR ({gap/(valor_ref.median()*0.30)*100:.1f}% deficit)")

print("\n" + "="*80)
print("5. SIMULACION: % INSULTING CON FLOORS ACTUALES")
print("="*80)

for posture, floor in floor_ratios.items():
    n_insulting = (sale_price < floor * valor_ref).sum()
    pct = n_insulting / len(df) * 100
    print(f"  {posture:15} (floor={floor:.0%}): {n_insulting:3}/{len(df)} ({pct:5.1f}%) insultantes")

if sale_insulting is not None:
    actual_insulting = sale_insulting.sum()
    print(f"\n[REAL] sale_insulting_flag=True: {actual_insulting}/{len(df)} ({actual_insulting/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("6. PROPUESTA DE RECALIBRACION")
print("="*80)

# Calcular floors que darían 20-30% insulting
print("\nFLOORS QUE DARIAN ~30% INSULTING:")
for target_pct in [0.30]:
    target_floor = sale_price.quantile(target_pct) / valor_ref.median()
    print(f"  Floor para ~{target_pct*100:.0f}% insulting: {target_floor:.2f} ({target_floor*100:.0f}%)")

# Propuesta realista
print("\n[PROPUESTA] FLOORS REALISTAS PARA MERCADO NPL:")
print("  PRUDENCIAL:   0.20 (20%)  - conservador pero ejecutable")
print("  BALANCEADO:   0.15 (15%)  - equilibrado")
print("  DESINVERSION: 0.10 (10%)  - agresivo")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("1. Floors actuales (50%/40%/30%) son IRREALISTICAMENTE ALTOS para NPL")
print("2. Mercado NPL tipico: precios = 10-30% valor referencia")
print("3. ACCION REQUERIDA: Recalibrar floor_ratio en config.py")
print("4. Mantener logica de gates pero con thresholds realistas")
print("="*80)

