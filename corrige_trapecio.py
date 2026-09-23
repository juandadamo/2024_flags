"""
corregir_trapecio.py
Paso 2: aplicar transformación de perspectiva a la imagen de referencia.
Los 4 puntos del trapecio se mapean al bounding box rectangular.
Genera una función reutilizable para aplicar a cualquier frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import ProjectiveTransform, warp

# --- rutas ---
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'
fn_ref = dird0 + "frame_000001.tiff"

# --- cargar imagen completa ---
ref_full = imread(fn_ref).astype(np.double)

# --- puntos del trapecio (los que marcaste) ---
# orden: sup-izq, sup-der, inf-der, inf-izq
src = np.array([
    [  68.04545455,  110.62987013],
    [1279.13636364,   86.12987013],
    [1277.62987013,  915.99350649],
    [  69.92857143,  893.14935065],
])

# --- rectángulo destino (bounding box) ---
x0 = src[:, 0].min()
x1 = src[:, 0].max()
y0 = src[:, 1].min()
y1 = src[:, 1].max()
ancho = x1 - x0
alto  = y1 - y0

dst = np.array([
    [0,     0],
    [ancho, 0],
    [ancho, alto],
    [0,     alto],
])

# --- función reutilizable ---
def corregir_trapecio(img, src, dst):
    """
    Aplica transformación de perspectiva a una imagen.
    img: array 2D
    src: 4x2 puntos origen (col, fila)
    dst: 4x2 puntos destino (col, fila)
    """
    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    return warp(img, tform.inverse, output_shape=(int(alto), int(ancho)),
                preserve_range=True)

# --- aplicar ---
ref_corr = corregir_trapecio(ref_full, src, dst)

# --- mostrar comparación ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].imshow(ref_full, cmap='gray')
ax[0].plot(src[:, 0], src[:, 1], 'r.-', lw=1.5, ms=10)
ax[0].set_title('Original (con trapecio)')
ax[1].imshow(ref_corr, cmap='gray')
ax[1].set_title('Corregida (rectángulo)')

plt.tight_layout()
plt.savefig('comparacion_trapecio.png', dpi=150)
plt.show()

# --- guardar ---
np.savez('trapecio_correccion.npz',
         src=src, dst=dst,
         ancho=ancho, alto=alto)
print(f"Tamaño destino: {int(ancho)} x {int(alto)} px")
print("Guardado: comparacion_trapecio.png y trapecio_correccion.npz")
