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
mks = ['o','^','s','p','v','^','*']
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from tikzplotlib import save as tikz_save
if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/'
    dir_tik = '/home/juan/Documents/Publicaciones/2026_shear_flutter/tikzs/'

plt.close('all')
nfont = 20
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": nfont,
    "axes.titlesize": nfont - 2,
    "axes.labelsize": nfont,
    "xtick.labelsize": nfont - 2,
    "ytick.labelsize": nfont - 2,
    "legend.fontsize": nfont - 1,
})
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
escalax = 1/0.130
Lbandera = 138.5 # mm
Lbandera = 130
dx = 1/escalax


Papel_80.L = Lbandera*1e-3  # Convertir a metros
Papel_80.freq_nat()
fn = np.zeros((3,1))
for i in range(3):
    fn[i] = Papel_80.fn[i]


f_offset  = [11.9,  11.4,16.6]
f_onset  = [16.3,20.0, 25.9 ]
u_offset, u_onset = np.tile(np.zeros_like(f_offset),[2,1])
for i,f_offi, f_oni in zip (range(len(f_offset)),f_offset,f_onset):
    u_offset[i] = veloc_tunel_ib(f_offi)
    u_onset[i] = veloc_tunel_ib(f_oni)


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

lista_caso_2d = np.delete(lista_caso_2d,[2,7,8])
 

#
# Velocidad_full, Amplitud_full, Frecuencia_full = np.zeros((3,len(lista_caso_2d)))
#
# Velocidad_full, Amplitud_full, Frecuencia_full = lee_datos_foil(lista_caso_2d,Velocidad_full,Amplitud_full,Frecuencia_full)
plt.close('All')
 
gc.collect()
 
#
# Amplitud_full = Amplitud_full/Lbandera
# Uc = veloc_tunel_ib(frec_c)
# U = Velocidad_full - Uc
# Velocidad_m = Velocidad_full/2
# p1 = np.polyfit(U[:npoints]**.5, Amplitud_full[:npoints]/2,1)
# fun_Amplitud = np.poly1d(p1)




UB = 1/L * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5

sigma = rhoa_b*L/(rho_papel*Papel_80.thickness)
# Contenido en frecuencia de la señal!!!!!!!

figb,axb = plt.subplots(1,2,figsize=(13,5))
ax0b,ax1b = axb

fig,ax = plt.subplots(1,2,figsize=(13,5))
ax0,ax1 = ax
# ax3.plot(Velocidad_full/2/UB,Frecuencia_full ,'o',fillstyle='none',markersize=10)
# ax3.plot(Velocidad_triang/2/UB,Frecuencia_triang ,'^',fillstyle='none',markersize=10)
# ax3.plot(Velocidad_rect/2/UB,Frecuencia_rect ,'s',fillstyle='none',markersize=10)

file_data = 'mediciones_2026.xlsx'
data_0 = pd.read_excel(file_data,header=1,sheet_name=0)  # lisa
data_1 = pd.read_excel(file_data,header=1,sheet_name=1)  # Aserrada
data_2 = pd.read_excel(file_data,header=1,sheet_name=2)  # Almenada
data_3 = pd.read_excel(file_data,header=1,sheet_name=3)  # Almenada_old
data_4 = pd.read_excel(file_data,header=1,sheet_name=4)  # Almenada2_old
data_5 = pd.read_excel(file_data,header=1,sheet_name=5)  # Aserrada_old
data_6 = pd.read_excel(file_data,header=1,sheet_name=6)  # Plana_old


u_lisa  = data_0[2:19]['Velocidad'].to_numpy()
dx_lisa = data_0[2:19]['Dx [mm]'].to_numpy()
freq_lisa = data_0[2:19]['Frecuencia'].to_numpy()

u_aserrada  = data_1[2:15]['Velocidad'].to_numpy()
dx_aserrada = data_1[2:15]['Dx [mm]'].to_numpy()
freq_aserrada = data_1[2:15]['Frecuencia'].to_numpy()

u_c_almenada = veloc_tunel_ib(data_2['Variador[Hz]'][10])
data_2 = data_2.drop(10)

u_almenada  = data_2[2:13]['Velocidad'].to_numpy()
dx_almenada = data_2[2:13]['Dx [mm]'].to_numpy()
freq_almenada = data_2[2:13]['Frecuencia'].to_numpy()

u_almenada = np.append(u_almenada,u_c_almenada)
dx_almenada = np.append(dx_almenada,0)
freq_almenada = np.append(freq_almenada,0)

u_almenada_old = data_3[1:12]['Velocidad'].to_numpy()
dx_almenada_old = data_3[1:12]['Dx [mm]'].to_numpy()
# freq_almenada_old = data_3[1:12]['Frecuencia'].to_numpy()



lin0, = ax0.plot(u_lisa/2/UB,dx_lisa/Lbandera/2,'o',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
lin1, = ax0.plot(u_aserrada/2/UB,dx_aserrada/Lbandera/2,'^',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
lin2, = ax0.plot(u_almenada/2/UB,dx_almenada/Lbandera/2,'s',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)

ax1.plot(u_lisa/2/UB,freq_lisa/u_lisa*2*Lbandera*1e-3,'o',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
ax1.plot(u_aserrada/2/UB,freq_aserrada/u_aserrada*2*Lbandera*1e-3,'^',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
ax1.plot(u_almenada/2/UB,freq_almenada/u_almenada*2*Lbandera*1e-3,'s',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)

ax0.set_ylabel('$A/2L$')
ax0.set_xlabel(r'$u^*=u_\infty/2 L\sqrt{\rho_s e/B}$')

for axi in [ax0,ax1]:
    axi.grid('gray',which='both')
    axi.set_xlabel(r'$u^*$')
ax1.set_ylabel(r'$A/2L$')
ax1.set_ylabel(r'$f_{foil}L/u_m$')
ax1.set_ylim([0.,0.5])
ax1.set_ylim([0.1,0.7])


ax0b.plot(u_lisa,dx_lisa,'o',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
ax0b.plot(u_aserrada,dx_aserrada,'^',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
ax0b.plot(u_almenada,dx_almenada,'s',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)

ax1b.plot(u_lisa,freq_lisa,'o',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
ax1b.plot(u_aserrada,freq_aserrada,'^',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
ax1b.plot(u_almenada,freq_almenada,'s',linestyle='none',fillstyle='none',markersize=10,markeredgewidth=2)
fig.savefig(dirw+'amplitudes_frecs_all_v2.pdf',dpi=300, bbox_inches='tight')
#lin3, = ax0.plot(u_almenada_old/2/UB,dx_almenada_old/Lbandera/2,'s',linestyle='none',fillstyle='none',color=lin2.get_color())

# fig1.savefig(dirw+'amplitudes_all_v2.pdf',dpi=300, bbox_inches='tight')
#fig.savefig(dirw+'amplitudes_frecs_all_v2.pdf',dpi=300, bbox_inches='tight')

