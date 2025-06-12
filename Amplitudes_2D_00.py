import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
#%matplotlib widget
import serial,socket,os,glob,sys
#import atexit
import numpy as np
import pandas as pd
import time, threading,sys,glob
from ipywidgets import interact, interact_manual,interactive,widgets,Layout
colores = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
import tifffile as tif
import skimage as sk
from IPython.display import Latex
from funciones_flag import *
from scipy.signal import find_peaks
mks = ['s','o','>','p','v','^','*']
from skimage.filters import threshold_otsu


dir_data = '/home/juan/data/balseiro/vid_2025-02-24_19-58-59/'
dir_data = '/home/juan/data/balseiro/vid_2025-02-24_20-06-35/'
files_list = np.sort(glob.glob(dir_data+'*.tiff'))

nsnapshots = 1000
Amp_i = np.zeros((nsnapshots))

for i,filei in enumerate(files_list[:nsnapshots]):
    A = tif.imread(filei)
    if i==0:
        A_total = np.tile(np.zeros_like(A),[nsnapshots,1,1])
        A_curva_i = np.zeros((nsnapshots,len(A.T)))
    A_total[i] = A
    A_max_j = np.zeros(len(A.T))
    Int_j = np.zeros(len(A.T))
    umbral_intensidad = sk.filters.threshold_otsu(A)/3

    for j,Aj in enumerate(A.T):
        A_max_j[j] = Aj.argmax()
        Int_j[j] = Aj.max()
    Adiff = np.diff(A_max_j)
    Adiff[np.abs(Adiff)>50] = 1000
    nmax  = np.nonzero(Int_j>umbral_intensidad)[0].max()
    A_max_j = A_max_j[:nmax]
    A_curva_i[i,:nmax] = A_max_j
    Amp_i[i] = A_max_j[-5:].mean()
Astd, Amean = (A_total.std(0),A_total.mean(0))
del(A_total)


n_snapshot_view = 223
filei = files_list[n_snapshot_view]
Ai = tif.imread(filei)
fig1,ax1 = plt.subplots(1,2)
ax1a,ax1b =ax1
ax1a.imshow(Astd)
npaso = 20

A_max_j = A_curva_i[n_snapshot_view]
A_max_j = A_max_j[A_max_j>0]
nx = range(len(A_max_j))[::npaso]
ax1b.plot(nx,A_max_j[::npaso],'yo',fillstyle='none',markersize=7)

ax1b.imshow(Ai);
ax1b.plot([0,A.shape[1]],[Amp_i[n_snapshot_view],Amp_i[n_snapshot_view]],'r')
fig.savefig('plots_py/stats_snapshot_fit.png')

fig2,ax2 = plt.subplots()
for i in range(1000)[::50]:
    ax2.plot(A_curva_i[i],'o',linestyle='none',fillstyle='none')

fig2.savefig('plots_py/perfiles_t.png')
