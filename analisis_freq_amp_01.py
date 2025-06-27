import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
#%matplotlib widget
import serial,socket,os,glob,sys
#import atexit
import numpy as np
import pandas as pd
import time, threading,sys,glob
colores = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
import tifffile as tif
import skimage as sk
from IPython.display import Latex
from funciones_flag import *
from scipy.signal import find_peaks
mks = ['s','o','>','p','v','^','*']
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import feature, morphology
if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'

plt.close('all')

# Opcional: Forzar la recolección de basura (gc) para liberar memoria
import gc
gc.collect()

rhoa = 1.2
rhoa_b = 1.0888  #densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa
Uinf = 12
delta_cl = 18e-3 # espesor de capa limite para velocidad 12m/s

#longitud caracteristica de la placa plana (tunel) en base a la medicion en Balseiro
x_carac = longitud_equivalente_capa_limite_turbulenta(delta_cl,Uinf,nu)
U = 12
delta_U12 = delta_turb(x_carac,U,nu)



fsampling = 1000 # Hz
escalax = 1/0.138 # px/mm
Lbandera = 138.5 # mm



Papel_80.L = Lbandera*1e-3  # Convertir a metros
Papel_80.freq_nat()
fn = np.zeros((3,1))
for i in range(3):
    fn[i] = Papel_80.fn[i]


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
 


caso = 'rect'
caso = 'triang'
# caso = 'full'

if caso == 'full':
    npoints = 5
    frec_c = 12.7
elif caso == 'triang':
    npoints = 4
    frec_c = 11.4


lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*'))

 
Velocidad, Amplitud, Frecuencia = np.zeros((3,len(lista_caso_2d)))
for j, filej in enumerate(lista_caso_2d[:]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']



    # Cargar imagen en escala de grises
    image = Asum**.12
    YT = A1['A_curva_i']
    frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
    Velocidad[j] =  veloc_tunel_ib(frec_j)
    # Detección de bordes con Canny

    edges = feature.canny(image, sigma=4)
    closed_edges = morphology.closing(edges, morphology.disk(radius=5))
    image[closed_edges] = 1
    image[np.logical_not(closed_edges)] = 0

    lim_superior =np.nonzero(image==1)[0].max()
    lim_inferior = np.nonzero(image==1)[0].min()
    delta_coord = lim_superior - lim_inferior
    # raise ValueError()
    Amplitud[j]  = delta_coord*1.0/ escalax  # mm

    Fourier_YT = np.fft.fft(YT.T,axis=1)
    FYT = np.abs(Fourier_YT).sum(axis=0)
    freq_YT = np.fft.fftfreq(len(YT), d=1/fsampling)  
    peak_freqs, _ = find_peaks(FYT, height=0.1*np.max(FYT))
    peak_freqs = peak_freqs[freq_YT[peak_freqs]>0]
    Frecuencia[j] = freq_YT[peak_freqs][FYT[peak_freqs].argmax()]
    print(f"Frecuencia de la señal: {Frecuencia[j]:.2f} Hz")
    plt.subplots()

    plt.semilogy(freq_YT[freq_YT>0], FYT[freq_YT>0], 'k-')
    plt.grid()
    plt.xlabel('Frecuency (Hz)')
    plt.ylabel('PSD')
    plt.plot(freq_YT[peak_freqs], FYT[peak_freqs], 'ro')
    plt.xlim([0, 100])
    plt.tight_layout()
    plt.savefig(dirw+'Fourier_YT_'+caso+'_'+str(j)+'.png')

plt.close('All')
 
gc.collect()
 

Amplitud = Amplitud/Lbandera
Uc = veloc_tunel_ib(frec_c)
U = Velocidad - Uc
Velocidad_m = Velocidad/2



fig4,ax4 = plt.subplots()
deltaw = delta_turb(x_carac,Velocidad,nu)
ax4.plot(np.sqrt(U), Frecuencia*deltaw/Velocidad_m, 'ks', fillstyle='none')
ax4.grid()
ax4.set_ylabel(r'$f_{foil}\delta_w/U$')
ax4.set_xlabel(r'$\sqrt{U-U_c}$')
fig4.tight_layout()
fig4.savefig(dirw+'Freq_sqrt_V'+caso+'.png')



fig6,ax6 = plt.subplots()
delta_w = delta_turb(x_carac,Velocidad,nu)

ax6.plot(Frecuencia*delta_w/(Velocidad_m),Amplitud, 'ks', fillstyle='none')
ax6.grid()
ax6.set_ylabel(r'$A/L$')
ax6.set_xlabel(r'$f_{foil}\delta_w/U$')
ax6.set_ylim([0,0.8])
ax6.set_yticks(np.arange(0, 0.9, 0.1))    
fig6.tight_layout()
fig6.savefig(dirw+'Freq_Amp'+caso+'.png')

((Papel_80.E*Papel_80.thickness**3) / (rhoa*Papel_80.L**3))**0.5


fig7,ax7 = plt.subplots()
ax7.plot(Velocidad,Frecuencia, 'ks', fillstyle='none')
ax7.grid()
ax7.set_xlabel(r'$U$')
ax7.set_ylabel(r'$f_{foil}$ [hz]')
ax7.set_ylim([9,26])
ax7.set_xlim([6,15])
# ax7.set_yticks(np.arange(0, 0.9, 0.1))    
fig7.tight_layout()
fig7.savefig(dirw+'Freq_Veloc_'+caso+'.png')