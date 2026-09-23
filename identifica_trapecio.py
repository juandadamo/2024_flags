"""
identificar_trapecio.py
Paso 1: marcar a mano los 4 vértices del trapecio sobre la imagen de referencia completa.
Guarda los puntos en trapecio_puntos.npz para el paso siguiente.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# --- rutas ---
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'
fn_ref = dird0 + "frame_000001.tiff"

# --- cargar imagen completa (sin recorte) ---
ref_full = imread(fn_ref).astype(np.double)
print(f"Dimensiones de ref_full: {ref_full.shape}")

# --- mostrar y pedir 4 clics ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(ref_full, cmap='gray')
ax.set_title('Clic en los 4 vértices: sup-izq, sup-der, inf-der, inf-izq')
ax.set_xlabel('col [px]')
ax.set_ylabel('fila [px]')

puntos = plt.ginput(4, timeout=0)
plt.close(fig)

puntos = np.array(puntos, dtype=np.float64)
print("Puntos marcados (col, fila):")
print(puntos)

# --- guardar ---
np.savez('trapecio_puntos.npz', puntos=puntos, fn_ref=fn_ref)
print("Guardado: trapecio_puntos.npz")
