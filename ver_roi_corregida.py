"""
ver_roi_corregida.py
Muestra la imagen de referencia corregida con ejes, para redefinir la ROI.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import ProjectiveTransform, warp

dird0 = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'
fn_ref = dird0 + "frame_000001.tiff"
ref_full = imread(fn_ref).astype(np.double)

src = np.array([
    [  68.04545455,  110.62987013],
    [1279.13636364,   86.12987013],
    [1277.62987013,  915.99350649],
    [  69.92857143,  893.14935065],
])
ancho = src[:,0].max() - src[:,0].min()
alto  = src[:,1].max() - src[:,1].min()
dst = np.array([[0,0],[ancho,0],[ancho,alto],[0,alto]])

tform = ProjectiveTransform()
tform.estimate(src, dst)
ref_corr = warp(ref_full, tform.inverse, output_shape=(int(alto), int(ancho)),
                preserve_range=True)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(ref_corr, cmap='gray')
ax.set_title('Imagen corregida. Leer límites de la bandera.')
ax.set_xlabel('col [px]')
ax.set_ylabel('fila [px]')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Dimensiones: {ref_corr.shape}")
