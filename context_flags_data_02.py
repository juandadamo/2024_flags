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
import pickle

# set tex fonts in plots with legible sizes
plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
mks = ['s','o','v']
plt.close('all')

rhoa = 1.2
rhoa_b = 1.0888  # densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa

# --- 1. CREAMOS ÚNICAMENTE LA FIGURA COMBINADA ---
fig0, ax0 = plt.subplots(1, 2, figsize=(12, 5.5))
ax0b,ax0a = ax0
# --- 2. PROCESAMIENTO PARA EL PRIMER GRÁFICO (ax0[0]) ---
IM = plt.imread('raynaud_mulleners.png')

with open('plot_stability/fig_stability.pickle', 'rb') as f:
    fig_pickle = pickle.load(f)
ax_pickle = fig_pickle.get_axes()[0]

# Pasamos los elementos del gráfico del pickle al ax0[0]
# Copiamos las líneas/datos que ya existían en el pickle
ax0a.plot([],[],color='w',label=r'$\mathcal A_r$')
for line in ax_pickle.get_lines():
    if len(line.get_label())<4:
        ax0a.plot(line.get_xdata(), line.get_ydata(),
                label=f'${float(line.get_label()):.1f}$', color=line.get_color(),
                linestyle=line.get_linestyle(), marker=line.get_marker(),fillstyle = line.get_fillstyle(),markersize=line.get_markersize(),markeredgewidth=line.get_markeredgewidth())
        print(line.get_label())
        print(f'${float(line.get_label()):.1f}$')
    else:
        ax0a.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                linestyle=line.get_linestyle(), marker=line.get_marker(),fillstyle = line.get_fillstyle(),markersize=line.get_markersize(),linewidth=line.get_linewidth())


# Copiamos también las imágenes de fondo si el pickle tenía el imshow de raynaud_mulleners
for img in ax_pickle.get_images():
    ax0a.imshow(img.get_array(), extent=img.get_extent(), aspect=img.get_aspect())

plt.close(fig_pickle) # Ya extrajimos lo que necesitábamos, cerramos esta figura

# Cargamos datos nuevos
data_uoffset = pd.read_csv('datos_flutter_ustop_f.csv', decimal=',')
data_uonset = pd.read_csv('datos_flutter_uonset.csv', decimal=',')

Uoffset = data_uoffset['Ustop']/2
Uonset = data_uonset['Uonset']/2
L = data_uoffset['L']*1e-3
L2 = data_uonset['L']*1e-3

UB = 1/L * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
UB2 = 1/L2 * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
sigma = rhoa_b*L/(rho_papel*Papel_80.thickness)
sigma2 = rhoa_b*L2/(rho_papel*Papel_80.thickness)

# Graficamos los datos nuevos DIRECTAMENTE en ax0a
ax0a.plot(sigma, Uoffset/UB, marker=r'$\star$', markersize=15, markeredgewidth=2, label=r'$0.7$'+'\n'+r'$\mathrm{pres.~data}$', linestyle='none',color='tab:red',fillstyle='none')
ax0a.set_xlabel(r'$m^* = \rho_f L / \rho_s e$')
ax0a.set_ylabel(r'$u^*_{\rm{offset}}=u_{\rm{offset}} L\sqrt{\rho_s e/B}$')

ax0a.legend(title=r'$\mathcal A_r$', fontsize=14)
ax0a.grid(which='both')

# Ajuste de las etiquetas de la leyenda en ax0a
handles, labels = ax0a.get_legend_handles_labels()
# if len(labels) > 1:
#     labels[1] = '1.0'
ax0a.legend(handles, labels,fontsize=14)


# --- 3. PROCESAMIENTO PARA EL SEGUNDO GRÁFICO (ax0b) ---
# Graficamos DIRECTAMENTE en ax0b en lugar de crear fig1
ax0b.plot(L*1e3, Uoffset*2, marker=r'$\star$', color='tab:red', fillstyle='none', linestyle='none', markersize=15, label=r'$u_{\mathrm{offset}}$',markeredgewidth = line.get_markeredgewidth())

L2_f = np.unique(L2)
uonset_f, uonset_f_e = np.tile(np.zeros_like(L2_f), [2, 1])
for i, L2i in enumerate(L2_f):
    uonset_f[i] = Uonset[L2==L2i].mean()*2
    uonset_f_e[i] = Uonset[L2==L2i].std()*2

ax0b.errorbar(L2_f*1e3, uonset_f, uonset_f_e, marker=mks[1], linestyle='none',
                markersize=10, capsize=10, elinewidth=1, fillstyle='none', label=r'$u_{\mathrm{onset}}$',markeredgewidth = line.get_markeredgewidth())
ax0b.grid()
ax0b.set_xlabel(r'$L ~ [\mathrm{mm}]$')
ax0b.set_ylabel(r'$u_\infty~[\mathrm{m/s}]$')
ax0b.legend()
ax0a.set_xlim([0,4])
ax0a.set_ylim([0,30])
# --- 4. GUARDAR LA FIGURA COMPUESTA FINAL ---
fig0.tight_layout() # Para que no se superpongan los textos de ambos gráficos

ax0a.text(0.25,27,r'$\mathrm{b)}$',fontsize=20)
ax0b.text(65,31,r'$\mathrm{a)}$',fontsize=20)
fig0.savefig('/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/combined_stability_flutter.pdf', dpi=150, bbox_inches='tight')

# plt.show()
