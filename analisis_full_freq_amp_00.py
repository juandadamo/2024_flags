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
from sklearn.cluster import KMeans




mks = ['s','o','>','p','v','^','*']
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/'
# dirw = 'figures/'
plt.close('all')
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
caso = 'full'

if caso == 'full':
    npoints = 5
    frec_c = 12.7
elif caso == 'triang':
    npoints = 5
    frec_c = 13


lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*.npz'))

#lista_caso_2d = np.delete(lista_caso_2d,[2,7,8])
 

 
Velocidad, Amplitud, Frecuencia = np.zeros((3,len(lista_caso_2d)))
factor_thresh = 1.5
for j, filej in enumerate(lista_caso_2d[2:3]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']
    
    YT = A1['A_curva_i']
    YT2 = YT.copy()
    YT = YT[:,45:1000]
    frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
    Velocidad[j] =  veloc_tunel_ib(frec_j)


    print(f"Velocidad del túnel de viento: {Velocidad[j]:.2f} m/s")



    Fourier_YT = np.fft.fft(YT.T,axis=1)
    FYT = np.abs(Fourier_YT).sum(axis=0)
    freq_YT = np.fft.fftfreq(len(YT), d=1/fsampling)  
    # peak_freqs, _ = find_peaks(FYT, height=0.1*np.max(FYT))
    # Frecuencia[j] = freq_YT[peak_freqs][FYT[peak_freqs].argmax()]

    Amatrix = YT - YT.mean(0)
    U,s,Vh = np.linalg.svd(Amatrix,full_matrices=True)
    n_clusters = 36
    kmeans = KMeans(n_clusters, random_state=0, n_init="auto").fit(YT)
    # raise ValueError()
    fig,ax = plt.subplots()
    for i in range (n_clusters):
        ax.plot(YT[kmeans.labels_==i].mean(axis=0))



raise ValueError()
plt.close('All')
 
gc.collect()
 

Amplitud = Amplitud/Lbandera
Uc = veloc_tunel_ib(frec_c)
U = Velocidad - Uc
Velocidad_m = Velocidad/2
p1 = np.polyfit(U[:npoints]**.5, Amplitud [:npoints],1)
fun_Amplitud = np.poly1d(p1)




UB = 1/L * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5

sigma = rhoa_b*L/(rho_papel*Papel_80.thickness)
# Contenido en frecuencia de la señal!!!!!!!

fig4,ax4 = plt.subplots()
deltaw = delta_turb(x_carac,Velocidad,nu)
ax4.plot(Velocidad/UB/2, Frecuencia*Lbandera*1e-3/Velocidad_m, 'ks', fillstyle='none')
ax4.grid()
#ax4.set_ylim([0.045, 0.065])
ax4.set_ylabel(r'$f_{foil}L/U$')
# ax4.set_xlabel(r'$\sqrt{U-U_c}$ [m/s$^{1/2}$]')
ax4.set_xlabel(r'$u^*$')
# ax4.set_xlim([0,2.5])
ax4.set_ylim([0,0.5])
fig4.tight_layout()
fig4.savefig(dirw+'Freq_adim_V'+caso+'.png',dpi=300, bbox_inches='tight')



fig6,ax6 = plt.subplots()
delta_w = delta_turb(x_carac,Velocidad,nu)

ax6.plot(Frecuencia*delta_w/(Velocidad_m),Amplitud, 'ks', fillstyle='none')
ax6.grid()
ax6.set_xlabel(r'$A/L$')
ax6.set_ylabel(r'$f_{foil}\delta_w/U$')
ax6.set_ylim([0,0.8])
ax6.set_yticks(np.arange(0, 0.9, 0.1))    
fig6.tight_layout()
fig6.savefig(dirw+'Freq_Amp'+caso+'.png',dpi=300, bbox_inches='tight')

((Papel_80.E*Papel_80.thickness**3) / (rhoa*Papel_80.L**3))**0.5


fig7,ax7 = plt.subplots()
ax7.plot(Velocidad,Frecuencia, 'ks', fillstyle='none')
ax7.grid()
ax7.set_xlabel(r'$U$ [m/s]')
ax7.set_ylabel(r'$f_{foil}$ [hz]')
ax7.set_ylim([9,26])
ax7.set_xlim([6,15])
# ax7.set_yticks(np.arange(0, 0.9, 0.1))    
fig7.tight_layout()
fig7.savefig(dirw+'Freq_Veloc_'+caso+'.png',dpi=300, bbox_inches='tight')
