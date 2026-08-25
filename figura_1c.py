
import gc
import glob
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from skimage import exposure
from tikzplotlib import save as tikz_save

from funciones_flag import *

# --- CONFIGURACIÓN DE REPOSITORIOS Y SALIDAS ---
dirout = '/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/'
dirout2 = '/home/juan/Documents/Publicaciones/2026_shear_flutter/tikzs/'

# --- CONFIGURACIÓN ESTILÍSTICA DE PLOTS (PAPER) ---
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

plt.close('all')
gc.collect()

# --- CONSTANTES FÍSICAS Y GEOMÉTRICAS ---
fsampling = 1000   # Hz
Lbandera = 128.5   # mm
escalax = 1 / 0.138 * 1.27  # px/mm
dt_muestreo = 1 / fsampling  # s
nyorigin = 491
nxorigin = 41

# --- CARGA DE DATOS ---
A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']

# --- PROCESAMIENTO DE IMAGEN Y MALLADO ---
xmin, xmax = np.array([0, Asum.shape[1]]) / escalax - nxorigin / escalax
ymin, ymax = np.array([0, Asum.shape[0]]) / escalax - nyorigin / escalax

X, Y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) / Asum.shape[1]),
                   np.arange(ymin, ymax, (ymax - ymin) / Asum.shape[0]))

Asum_normalized = (Asum - Asum.min()) / (Asum.max() - Asum.min())
Asum_eq = exposure.equalize_adapthist(Asum_normalized, clip_limit=0.03)

# --- SVD (MODOS RETENIDOS) ---
n1, n2 = [46, 1045]
U, S, Vh = np.linalg.svd((YT - YT.mean(0))[:, n1:n2])
s = 12
YT_r = np.dot(U[:, :s], np.dot(Vh[:s].T, np.diag(S[:s])).T) + YT.mean(0)[n1:n2]

# --- INICIALIZACIÓN DE LA FIGURA ---
fig, ax0 = plt.subplots(1, 1, figsize=(6.5, 5), layout='constrained')

ax0.set_xlabel(r'$x~ [\mathrm{mm}]$')
ax0.set_ylabel(r'$y~ [\mathrm{mm}]$')
ax0.set_ylim([-35, 35])
ax0.set_xlim([0, 160])
ax0.grid(True)

# --- PARÁMETROS DEL LOOP TEMPORAL ---
T_flap = 1/12
dt = 0.001 / T_flap * 2 * np.pi

i0 = 50
paso_i = 10
indices_efectivos = np.array([i for i in range(i0, 1000) if (i < i0 + 90) and (i % paso_i == 0)])
N = len(indices_efectivos)

# --- MAPA DE COLORES SEGURO (HEX) ---
cmap = plt.colormaps['viridis']
rgba_raw = cmap(np.linspace(0, 1, N))
rgba_colors = [mcolors.to_hex(c) for c in rgba_raw]
cmap_discreto = mcolors.ListedColormap(rgba_colors)

tiempos_en_pi = ((indices_efectivos - i0) * dt) / np.pi

# --- BUCLE DE PLOTEO ---
# Listas para acumular la nube gris y vectorizar en un único plot definitivo
x_gris_total = []
y_gris_total = []

ii = 0
for i in range(50, 1000, 1):
    # Guardamos los puntos de fondo sin llamar a ax0.plot individualmente
    x_gris_total.append(X[0] * 1.06 / 1)
    y_gris_total.append((YT[i] - nyorigin) / escalax)

    # Curvas de color seleccionadas
    if i < i0 + 90:
        xdata = X[0] * 1.06 / 1
        ydata = (YT[i] - nyorigin) / escalax
        if i % paso_i == 0:
            ax0.plot(xdata, ydata, marker='.', color=rgba_colors[ii],
                     markersize=1, linestyle='none', zorder=1)
            ii += 1

# Un solo ploteo unificado para toda la nube de fondo gris (¡Rápido de renderizar en el PDF!)
ax0.plot(np.ravel(x_gris_total), np.ravel(y_gris_total), marker='.', color='lightgray',
         markersize=.1, linestyle='none', zorder=0,rasterized=True)

# --- GENERACIÓN DEL COLORBAR ---
sm = plt.cm.ScalarMappable(cmap=cmap_discreto)
sm.set_clim(-0.5, N - 0.5)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax0, ticks=np.arange(N))
cbar_labels = [r'$' + f'{t:.2f}' + r'\pi$' for t in tiempos_en_pi]
cbar.ax.set_yticklabels(cbar_labels)
cbar.ax.set_title(r'$\mathrm{phase}$', fontsize=nfont - 2, pad=10)


# --- COTA DE AMPLITUD (A) ---
x_cota = 145
y_max = 21
y_min = -30

ax0.hlines(y=[y_min, y_max], xmin=110, xmax=x_cota, colors='tab:blue', linestyles='--', linewidth=1.5)
ax0.annotate('', xy=(x_cota, y_min), xytext=(x_cota, y_max),
             arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=1.5, shrinkA=0, shrinkB=0))
ax0.text(x_cota + 3, (y_max + y_min) / 2, r'$\mathrm{A}$', color='tab:blue', fontsize=nfont, va='center', ha='left')


# --- PERFIL DE VELOCIDADES EN LA ENTRADA ---
y_perfil = np.linspace(-35, 35, 200)
delta = 12.0
y_suelo = 0

u_perfil = np.zeros_like(y_perfil)
for idx, y_val in enumerate(y_perfil):
    dist_suelo = y_val - y_suelo
    if dist_suelo <= 0:
        u_perfil[idx] = 0
    else:
        u_perfil[idx] = 1.0 - np.exp(-3.7 * dist_suelo / delta)

ancho_perfil = 25
x_perfil = u_perfil * ancho_perfil

ax0.plot(x_perfil, y_perfil, color='black', linewidth=2)
ax0.vlines(x=0, ymin=-35, ymax=35, colors='black', linewidth=1.5)

for y_flecha in np.linspace(-30, 25, 7):
    dist_suelo = y_flecha - y_suelo
    u_val = 1.0 - np.exp(-3.7 * dist_suelo / delta) if dist_suelo > 0 else 0
    x_flecha_max = u_val * ancho_perfil
    if x_flecha_max > 0:
        ax0.annotate('', xy=(x_flecha_max, y_flecha), xytext=(0, y_flecha),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1))

ax0.text(ancho_perfil + 2, 25, r'$u_\infty$', color='black', fontsize=nfont, va='center')
ax0.text(ancho_perfil + 2, y_suelo + 12, r'$\delta$', color='black', fontsize=nfont, va='center')
# ax0.text(ancho_perfil + 2, y_suelo + 12, r'$\delta \equiv \theta_0$', color='black', fontsize=nfont, va='center')


# --- GUARDADO FINAL ÚNICO ---
fig.savefig(dirout + 'ref_image_sum_full_v2.pdf', dpi=300)
# fig.savefig(dirout + 'ref_image_sum_full_v2.png', dpi=300)

