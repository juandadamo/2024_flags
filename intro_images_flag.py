import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob,sys
# from tikzplotlib import save as tikz_save
from matplotlib import rcParams
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
import tifffile as tif
# Cerrar todas las figuras existentes
plt.close('all')


import gc
gc.collect()

fsampling = 1000  # camera sampling rate Hz
Lflag = 138.5  # mm
scalax = 9.2 # px/mm
dt = 1 / fsampling  # s
tmin = 0
tmax = 1
# flag coordinates
nyorigin = 491
nxorigin = 41

# read tiffs

files_list = np.sort(glob.glob('Full - 13.4/*.tiff'))




# load
A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']



