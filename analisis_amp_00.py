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
# delta_x_px = 105
# delta_x = 15
# escalax = delta_x_px/delta_x  # px/mm


# delta_y_px = 471
# delta_y = 50
# escalay = delta_y_px/delta_y  # px/mm

# delta_x2_px = 201
# delta_x2 = 28
# escalax2 = delta_x2_px/delta_x2  # px/mm

escalax = 1/0.138 # px/mm


if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'


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

lista_caso_2d = np.delete(lista_caso_2d,[2,7,8])
 

 
Velocidad, Amplitud = np.zeros((2,len(lista_caso_2d)))
for j, filej in enumerate(lista_caso_2d[:]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']
    
    YT = A1['A_curva_i']
    frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
    Velocidad[j] =  veloc_tunel_ib(frec_j)
    if np.logical_and(caso == 'triang', j>-1):
        factor_thresh = 1
        Asum = Asum**(1/3)
    else:
        factor_thresh = 1.5

    print(f"Velocidad del túnel de viento: {Velocidad[j]:.2f} m/s")
    umbral_intensidad = sk.filters.threshold_otsu(Asum)/factor_thresh
    A2 = Asum>= umbral_intensidad
    A_clean = binary_closing(A2, square(3))  # Elimina píxeles aislados
    A_clean = binary_opening(A_clean, square(3))  # Suaviza bordes
    label_image = label(A_clean)
    image = Asum.copy()
 
    # raise ValueError()
    aux = []
    for region in regionprops(label_image):
        if region.area>1020:
            coord_amp = region.coords
            aux.append((coord_amp[:,0].max(),coord_amp[:,0].min()))

    aux = np.asarray(aux)

    delta_coord = np.abs(coord_amp[:,0].max()-coord_amp[:,0].min())
    delta_coord = np.abs(aux[:,0].max()-aux[:,1].min())
    # raise ValueError()
    Amplitud[j]  = delta_coord*1.0/ escalax  # mm
    if j==1:
        # raise ValueError()
        fig0,ax0 = plt.subplots()
        ax0.imshow(Asum)
        for YT_k in YT[100:150:10]:
            ax0.plot(YT_k,marker='o',color='tab:orange',markersize=0.5,linestyle='none')
        fig0.savefig(dirw+'image_sum_'+caso+'.png')
        #fig1,ax1 = plt.subplots()
        #ax1.imshow(label_image)
        ax0.plot(coord_amp[:,1],coord_amp[:,0],marker='o',fillstyle='none',linestyle='none',markersize=0.1,color='tab:orange')
        fig0.savefig(dirw+'image_label_'+caso+'.png')
        ax0.plot([1,len(YT.T)],[aux[:,0].max(),aux[:,0].max()],color='w',linewidth=3)
        ax0.plot([1,len(YT.T)],[aux[:,1].min(),aux[:,1].min()],color='w',linewidth=3)
        fig0.savefig(dirw+'image_label_amp_'+caso+'.png')
fig,ax = plt.subplots()
#Uc = Velocidad[0]


Uc = veloc_tunel_ib(frec_c)
U = Velocidad - Uc
ax.plot(Velocidad,Amplitud,'ks',fillstyle='none',linestyle='none')
ax.set_xlabel('$U$[m/s]')
ax.set_ylabel('$A$')
ax.grid()
fig.savefig(dirw+'Amplitudes_'+caso+'.png')




p1 = np.polyfit(U[:npoints]**.5, Amplitud [:npoints],1)
fun_Amplitud = np.poly1d(p1)


# Graficar ajuste
fig3,ax3 = plt.subplots()

ax3.plot(np.sqrt(U), Amplitud, 'ks', fillstyle='none', label='Data')
Us = np.linspace(0,U[npoints],100)
# ax3.plot(Us, intercept + slope * Us[:], 'r--', label=f'Ajuste lineal ($R^2 = {r_value**2:.3f}$)')
ax3.plot(Us**.5,fun_Amplitud(Us**.5), 'r--', label=f'Linear Fit')
ax3.set_xlabel(r'$\sqrt{U - U_c}$[m/s]$^{1/2}$')
ax3.set_ylabel('$A$')
# ax3.plot([0,0],[1.78e5,5e5],'k--')
ax3.legend()
ax3.grid()
fig3.tight_layout()
fig3.savefig(dirw+'Amplitudes_'+caso+'_ajuste.png')



# Contenido en frecuencia de la señal!!!!!!!

