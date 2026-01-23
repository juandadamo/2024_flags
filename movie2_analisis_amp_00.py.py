import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
#%matplotlib widget
import serial, socket, os, glob, sys
#import atexit
import numpy as np
import pandas as pd
import time, threading
colores = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
import tifffile as tif
import skimage as sk
from IPython.display import Latex
from funciones_flag import *
from scipy.signal import find_peaks
mks = ['s','o','>','p','v','^','*']
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

from skimage.morphology import skeletonize, thin, remove_small_objects, closing, square, disk, medial_axis, binary_opening, binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
plt.rcParams.update({
    "text.usetex": True,          # Usar LaTeX para renderizar texto
    "font.family": "serif",       # Familia de fuente (p. ej., Times New Roman)
    "font.size": 18,              # Tamaño base de la fuente
    "axes.titlesize": 16,         # Tamaño del título
    "axes.labelsize": 18,         # Tamaño de etiquetas de ejes
    "xtick.labelsize": 16,        # Tamaño de etiquetas del eje x
    "ytick.labelsize": 16,        # Tamaño de etiquetas del eje y
    "legend.fontsize": 17,        # Tamaño de la leyenda
})

plt.close('all')

import gc
gc.collect()
escalax = 1/0.138 # px/mm
Lbandera = 138.5 # mm

if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'

dirw = 'figures/'
caso = 'rect'
caso = 'triang'
caso = 'full'

if caso == 'full':
    npoints = 5
    frec_c = 12.7
elif caso == 'triang':
    npoints = 5
    frec_c = 13

lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*'))
lista_caso_2d = np.delete(lista_caso_2d, [2,7,8])

Velocidad, Amplitud = np.zeros((2, len(lista_caso_2d)))
for j, filej in enumerate(lista_caso_2d[:]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']
    YT = A1['A_curva_i']
    frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
    Velocidad[j] = veloc_tunel_ib(frec_j)
    if np.logical_and(caso == 'triang', j > -1):
        factor_thresh = 1
        Asum = Asum**(1/3)
    else:
        factor_thresh = 1.5

    print(f"Velocidad del túnel de viento: {Velocidad[j]:.2f} m/s")
    umbral_intensidad = sk.filters.threshold_otsu(Asum)/factor_thresh
    A2 = Asum >= umbral_intensidad
    A_clean = binary_closing(A2, square(3))  # Elimina píxeles aislados
    A_clean = binary_opening(A_clean, square(3))  # Suaviza bordes
    label_image = label(A_clean)
    image = Asum.copy()

    aux = []
    for region in regionprops(label_image):
        if region.area > 1020:
            coord_amp = region.coords
            aux.append((coord_amp[:,0].max(), coord_amp[:,0].min()))
    aux = np.asarray(aux)

    delta_coord = np.abs(coord_amp[:,0].max() - coord_amp[:,0].min())
    delta_coord = np.abs(aux[:,0].max() - aux[:,1].min())
    Amplitud[j] = delta_coord * 1.0 / escalax  # mm

    # --- Animación eficiente con FuncAnimation ---
    import matplotlib.animation as animation

    fig0, ax0 = plt.subplots()
    ax0.imshow(Asum, cmap='inferno', origin='lower')
    line, = ax0.plot([], [], marker='o', color='white', markersize=0.25, linestyle='none')

    def update(k):
        YT_k = YT[k]
        line.set_data(np.arange(len(YT_k)), YT_k)
        # Guardar cada frame como imagen
        fig0.savefig(dirw+f'video_amp/snapshots_veloc_{Velocidad[j]:.1f}_'+caso+f'{k:04d}.png', dpi=300, bbox_inches='tight')
        return line,

    ani = animation.FuncAnimation(fig0, update, frames=range(min(550, YT.shape[0])), blit=True)
    plt.close(fig0)