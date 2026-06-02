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

UB = 1/(Lbandera*1e-3) * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
data = []

data_0 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=0)  # lisa
data_1 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=1)  # Aserrada
data_2 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=2)  # Almenada
data_3 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=3)  # Almenada_old
data_4 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=4)  # Almenada2_old
data_5 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=5)  # Aserrada_old
data_6 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=6)  # Plana_old


u_lisa  = data_0[2:19]['Velocidad'].to_numpy()
dx_lisa = data_0[2:19]['Dx [mm]'].to_numpy()

u_aserrada  = data_1[2:15]['Velocidad'].to_numpy()
dx_aserrada = data_1[2:15]['Dx [mm]'].to_numpy()

data_2 = data_2.drop(10)
u_almenada  = data_2[2:13]['Velocidad'].to_numpy()
dx_almenada = data_2[2:13]['Dx [mm]'].to_numpy()

u_almenada_old = data_3[1:12]['Velocidad'].to_numpy()
dx_almenada_old = data_3[1:12]['Dx [mm]'].to_numpy()


fig,ax = plt.subplots()
lin0, = ax.plot(u_lisa/2/UB,dx_lisa/Lbandera/2,'o',linestyle='none')
lin1, = ax.plot(u_aserrada/2/UB,dx_aserrada/Lbandera/2,'^',linestyle='none')
lin2, = ax.plot(u_almenada/2/UB,dx_almenada/Lbandera/2,'s',linestyle='none')
lin3, = ax.plot(u_almenada_old/2/UB,dx_almenada_old/Lbandera/2,'s',linestyle='none',fillstyle='none',color=lin2.get_color())
ax.grid()
