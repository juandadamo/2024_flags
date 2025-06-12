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
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'

lista_full_2d = np.sort(glob.glob('data_out/full_freq*'))

Velocidad, Amplitud = np.zeros((2,len(lista_full_2d)))
for j, filej in enumerate(lista_full_2d[:]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']
    YT = A1['A_curva_i']
    frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
    Velocidad[j] =  veloc_tunel_ib(frec_j)

    umbral_intensidad = sk.filters.threshold_otsu(Asum)/1.5
    A2 = Asum>= umbral_intensidad
    A_clean = binary_closing(A2, square(3))  # Elimina píxeles aislados
    A_clean = binary_opening(A_clean, square(3))  # Suaviza bordes
    label_image = label(A_clean)



    aux = []
    for region in regionprops(label_image):
        if region.area>1020:
            coord_amp = region.coords
            aux.append((coord_amp[:,0].max(),coord_amp[:,0].min()))

    aux = np.asarray(aux)

    delta_coord = np.abs(coord_amp[:,0].max()-coord_amp[:,0].min())
    delta_coord = np.abs(aux[:,0].max()-aux[:,1].min())
    # raise ValueError()
    Amplitud[j]  = delta_coord*1.0
    if j==3:
        fig0,ax0 = plt.subplots()
        ax0.imshow(Asum)
        for YT_k in YT[100:150:10]:
            ax0.plot(YT_k,marker='o',color='tab:orange',markersize=0.5,linestyle='none')
        fig0.savefig(dirw+'image_sum.png')
        #fig1,ax1 = plt.subplots()
        #ax1.imshow(label_image)
        ax0.plot(coord_amp[:,1],coord_amp[:,0],marker='o',fillstyle='none',linestyle='none',markersize=0.1,color='tab:orange')
        fig0.savefig(dirw+'image_label.png')
        ax0.plot([1,len(YT.T)],[aux[:,0].max(),aux[:,0].max()],color='w',linewidth=3)
        ax0.plot([1,len(YT.T)],[aux[:,1].min(),aux[:,1].min()],color='w',linewidth=3)
        fig0.savefig(dirw+'image_label_amp.png')
fig,ax = plt.subplots()
#Uc = Velocidad[0]
frec_c = 12
frec_c = 12
Uc = veloc_tunel_ib(frec_c)
U = Velocidad - Uc
ax.plot(Velocidad,Amplitud,'ks',fillstyle='none',linestyle='none')
ax.set_xlabel('$U$[m/s]')
ax.set_ylabel('$A$')
ax.grid()
fig.savefig(dirw+'Amplitudes_full.png')



from scipy.stats import linregress

# Excluimos el primer punto (U = 0, A^2 = 401^2 ≈ 160801) para ver el comportamiento
slope, intercept, r_value, p_value, std_err = linregress(U[:-4], Amplitud[:-4]**2)

p1 = np.polyfit(U[:5], Amplitud [:5]**2,1)
fun_Amplitud = np.poly1d(p1)

print(f"Pendiente: {slope:.1f}, Intercepto: {intercept:.1f}, R²: {r_value**2:.3f}")

# Graficar ajuste
fig3,ax3 = plt.subplots()

ax3.plot(U, Amplitud**2, 'ks', fillstyle='none', label='Data')
Us = np.linspace(0,U[5],100)
# ax3.plot(Us, intercept + slope * Us[:], 'r--', label=f'Ajuste lineal ($R^2 = {r_value**2:.3f}$)')
ax3.plot(Us,fun_Amplitud(Us), 'r--', label=f'Linear Fit')
ax3.set_xlabel('$U - U_c$[m/s]')
ax3.set_ylabel('$A^2$')
ax3.plot([0,0],[1.78e5,5e5],'k--')
ax3.legend()
ax3.grid()
fig3.tight_layout()
fig3.savefig(dirw+'Amplitudes_full_ajuste.png')

