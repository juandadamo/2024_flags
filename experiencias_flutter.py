import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import numpy as np

# Lee el archivo CSV
df = pd.read_csv('datos_flutter.csv', names=['L', 'Uonset', 'Ustop'])

# Muestra cómo se ve originalmente
print("Datos originales:")
print(df.head(15))
print(f"\nDimensiones originales: {df.shape}")

# Elimina filas donde todas las columnas son NaN
df_clean = df.dropna(how='all')

# Convierte las columnas a numérico (maneja comas como decimales si es necesario)
# Si tus decimales usan punto (como en el ejemplo), esto está bien
df_clean['L'] = pd.to_numeric(df_clean['L'], errors='coerce')
df_clean['Uonset'] = pd.to_numeric(df_clean['Uonset'], errors='coerce')
df_clean['Ustop'] = pd.to_numeric(df_clean['Ustop'], errors='coerce')

# También puedes eliminar filas donde la columna L sea NaN (si L es tu identificador)
df_clean = df_clean.dropna(subset=['L'])

# Resetea el índice si lo deseas
df_clean = df_clean.reset_index(drop=True)

print("\n" + "="*50)
print("Datos limpios:")
print(df_clean.head(20))
print(f"\nDimensiones limpias: {df_clean.shape}")

# Información sobre los datos
print("\n" + "="*50)
print("Información del DataFrame:")
print(df_clean.info())

print("\n" + "="*50)
print("Valores únicos en L:")
print(sorted(df_clean['L'].unique()))

# Si quieres también puedes optar por mantener todas las filas y solo convertir a numérico
# y luego usar los datos según necesites

# Opción 2: Si prefieres llenar NaN con algún valor específico
# df_filled = df_clean.fillna(0)  # o cualquier otro valor
