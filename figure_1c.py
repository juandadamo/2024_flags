#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import exposure
# from tikzplotlib import save as tikz_save
from matplotlib import rcParams
from funciones_flag import *
from tikzplotlib import save as tikz_save
import glob, gc
import matplotlib.colors as mcolors
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
fig, ax0 = plt.subplots(1, 1, figsize=(6.5, 5))#, layout='constrained')

ax0.set_xlabel(r'$x~ [\mathrm{mm}]$')
ax0.set_ylabel(r'$y~ [\mathrm{mm}]$')
ax0.set_ylim([-35, 35])
ax0.set_xlim([0, 160])
ax0.grid(True)

# --- PARÁMETROS DEL LOOP TEMPORAL ---
T_flap = 1/12
dt = 0.001 / T_flap * 2 * np.pi  # Paso temporal en radianes

i0 = 50
paso_i = 10
# Sincronización matemática estricta: evaluamos cuántos múltiplos reales de 10 entran
indices_efectivos = np.array([i for i in range(i0, 1000) if (i < i0 + 90) and (i % paso_i == 0)])
N = len(indices_efectivos)  # N dinámico real basado en tus condiciones (Dará 9 curvas)

# --- MAPA DE COLORES Y LÍMITES DISCRETOS ---
cmap = plt.colormaps['viridis']
rgba_colors = cmap(np.linspace(0, 1, N))  # Fuerza a viridis a espaciarse perfecto entre 0 y 1 en N bloques
cmap_discreto = mcolors.ListedColormap(rgba_colors)

# Tiempos reales asociados a cada paso en múltiplos de pi
tiempos_en_pi = ((indices_efectivos - i0) * dt) / np.pi

# Seteo de parches de color discretos rodeando simétricamente cada tick
step = tiempos_en_pi[1] - tiempos_en_pi[0]
boundaries = np.linspace(tiempos_en_pi[0] - step/2, tiempos_en_pi[-1] + step/2, N + 1)
norm = mcolors.BoundaryNorm(boundaries, cmap_discreto.N)

# --- BUCLE DE PLOTEO ---
ii = 0
for i in range(50, 1000, 1):
    # Fondo en gris (todas las curvas)
    ax0.plot(X[0] * 1.06 / 1, (YT[i] - nyorigin) / escalax, marker='.', color='lightgray',
             markersize=.1, linestyle='none', alpha=0.5, zorder=0)

    # Curvas de color seleccionadas
    if i < i0 + 90:  # Ajustado al límite exacto de tus curvas deseadas
        xdata = X[0] * 1.06 / 1
        ydata = (YT[i] - nyorigin) / escalax
        if i % paso_i == 0:
            ax0.plot(xdata, ydata, marker='.', color=rgba_colors[ii],
                     markersize=1, linestyle='none', zorder=1)
            ii += 1

# --- GENERACIÓN DEL COLORBAR ---
sm = plt.cm.ScalarMappable(cmap=cmap_discreto, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax0, ticks=tiempos_en_pi, spacing='proportional')

# Formateo fino con estilo matemático \mathrm para el paper
cbar_labels = [r'$' + f'{t:.2f}' + r'\pi$' for t in tiempos_en_pi]
cbar.ax.set_yticklabels(cbar_labels)
cbar.ax.set_title(r'$\mathrm{phase}$', fontsize=nfont - 2, pad=10)

# --- GUARDADO EN ALTA CALIDAD ---



fig.savefig(dirout + 'ref_image_sum_full_v4.pdf', dpi=150,bbox_inches='tight')
# --- COTA DE AMPLITUD (A) ---
x_cota = 145  # Posición en X donde se ubicará la cota (fuera de la punta libre)
y_max = 21    # Deflexión máxima aproximada de la punta
y_min = -30   # Deflexión mínima aproximada de la punta

# Líneas de trazos horizontales (límites)
ax0.hlines(y=[y_min, y_max], xmin=110, xmax=x_cota, colors='tab:blue', linestyles='--', linewidth=1.5)

# Flecha doble vertical para la amplitud A
ax0.annotate('', xy=(x_cota, y_min), xytext=(x_cota, y_max),
             arrowprops=dict(arrowstyle='<->', color='tab:blue', lw=1.5, shrinkA=0, shrinkB=0))

# Texto "A" al lado de la flecha
ax0.text(x_cota + 3, (y_max + y_min) / 2, r'$\mathrm{A}$', color='tab:blue', fontsize=nfont, va='center', ha='left')


# --- PERFIL DE VELOCIDADES EN LA ENTRADA ---
# Generamos un eje Y local para la capa límite (desde el "suelo" en y=-35 hasta la corriente libre)
y_perfil = np.linspace(-35, 35, 200)
delta = 12.0 # mm (espesor de capa límite)
y_suelo = 0 # El límite inferior de tu gráfico

# Perfil analítico: u = 0 en el suelo, u = 1 de forma exponencial
# A y = y_suelo + 15, la distancia al suelo es 15, dando u ≈ 1 - e^(-3.7*15/12) = 0.99
u_perfil = np.zeros_like(y_perfil)
for idx, y_val in enumerate(y_perfil):
    dist_suelo = y_val - y_suelo
    if dist_suelo <= 0:
        u_perfil[idx] = 0
    else:
        u_perfil[idx] = 1.0 - np.exp(-3.7 * dist_suelo / delta)

# Escalamos el perfil para que quepa visualmente en la entrada del gráfico (ej: ancho de 30 mm)
ancho_perfil = 25
x_perfil = u_perfil * ancho_perfil

# 1. Dibujamos la línea del perfil de velocidades
ax0.plot(x_perfil, y_perfil, color='black', linewidth=2)

# 2. Dibujamos la línea base vertical (u = 0)
ax0.vlines(x=0, ymin=-35, ymax=35, colors='black', linewidth=1.5)

# 3. Dibujamos algunas flechas de vectores de velocidad (cada tanto en Y)
for y_flecha in np.linspace(-30, 25, 7):
    dist_suelo = y_flecha - y_suelo
    u_val = 1.0 - np.exp(-3.7 * dist_suelo / delta) if dist_suelo > 0 else 0
    x_flecha_max = u_val * ancho_perfil
    if x_flecha_max > 0:
        ax0.annotate('', xy=(x_flecha_max, y_flecha), xytext=(0, y_flecha),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1))

# 4. Etiquetas de texto en formato mathrm
ax0.text(ancho_perfil + 2, 25, r'$u_\infty$', color='black', fontsize=nfont, va='center')
ax0.text(ancho_perfil + 2, y_suelo + 12, r'$\delta \equiv \theta_0$', color='black', fontsize=nfont, va='center')

fig.savefig(dirout + 'ref_image_sum_full_v2.png', dpi=300)
fig.savefig(dirout + 'ref_image_sum_full_v3.pdf', dpi=150)
