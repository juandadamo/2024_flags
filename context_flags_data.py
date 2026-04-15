#!/usr/bin/env python
# coding: utf-8
from IPython.display import display, HTML


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import glob
from funciones_flag import *
from matplotlib.colors import to_rgba
from tikzplotlib import save as tikz_save
# set tex fonts in plots with legible sizes
plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
mks = ['s','o','v']
plt.close('all')
rhoa = 1.2
rhoa_b = 1.0888  #densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa

plt.close('all')
IM = plt.imread('raynaud_mulleners.png')

fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(IM,extent=[0,4,0,30],aspect='auto')



data_uoffset = pd.read_csv('datos_flutter_ustop_f.csv',decimal=',')
data_uonset = pd.read_csv('datos_flutter_uonset.csv',decimal=',')



Uoffset = data_uoffset['Ustop']/2
Uonset = data_uonset['Uonset']/2
L = data_uoffset['L']*1e-3
L2 = data_uonset['L']*1e-3

UB = 1/L * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
UB2 = 1/L2 * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
sigma = rhoa_b*L/(rho_papel*Papel_80.thickness)
sigma2 = rhoa_b*L2/(rho_papel*Papel_80.thickness)

ax.plot(sigma,Uoffset/UB,'s',markersize=10,fillstyle='none',markeredgewidth=3)
ax.set_xlabel(r'$m^* = \rho_f L / \rho_s e$')
ax.set_ylabel(r'$u^*_{\rm{offset}}=u_{\rm{offset}} L\sqrt{\rho_s e/B}$')

#fig.savefig('/home/juan/Documents/Publicaciones/2025_euromech/flag_shear/article/figures/instability.png')
#tikz_save('/home/juan/Documents/Publicaciones/2025_euromech/flag_shear/article/figures/instability_1.tikz')

# ax.plot(sigma2,Uonset/UB2,'o',markersize=10,fillstyle='none')

fig1,ax1 = plt.subplots(figsize=(8,8))
ax1.plot(L*1e3,Uoffset*2,'s',fillstyle='none')

L2_f = np.unique(L2)
uonset_f,uonset_f_e = np.tile(np.zeros_like(L2_f),[2,1])
for i,L2i in enumerate(L2_f):
    uonset_f[i] = Uonset[L2==L2i].mean()*2
    uonset_f_e[i] = Uonset[L2==L2i].std()*2
# lin, = ax1.plot(L2*1e3,Uonset*2,'o',fillstyle='none',markersize=3)
# ax1.errorbar(L2_f*1e3,uonset_f,uonset_f_e,marker=mks[1],linestyle='none',
#                 markersize=10,fillstyle='none',capsize=10,elinewidth=1,color=lin.get_color())


ax1.errorbar(L2_f*1e3,uonset_f,uonset_f_e,marker=mks[1],linestyle='none',
                markersize=10,fillstyle='none',capsize=10,elinewidth=1)
ax1.grid()
ax1.set_xlabel(r'$L$ [mm]')
ax1.set_ylabel(r'$u_\infty$ [m/s]')
# ax2=ax1.twinx()
# ax2.plot(L2_f*1e3,Uoffset/UB,marker=mks[1],linestyle='none',
#                 markersize=1,fillstyle='none')
#tikz_save('/home/juan/Documents/Publicaciones/2025_euromech/flag_shear/article/tikzs/u_L_flutter.tikz')
