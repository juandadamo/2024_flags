import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import exposure
# from tikzplotlib import save as tikz_save   
from matplotlib import rcParams

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

# Cerrar todas las figuras existentes
plt.close('all')

# Opcional: Forzar la recolección de basura (gc) para liberar memoria
import gc
gc.collect()

fsampling = 1000  # Hz
Lbandera = 128.5  # mm
escalax = 1 / 0.138  * 1.27 # px/mm 
dt = 1 / fsampling  # s
tmin = 0
tmax = 1
nyorigin = 491
nxorigin = 41

 
A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']
ny0 = 491

fig0,ax0 = plt.subplots()
fig1,ax1 = plt.subplots()

xmin,xmax = np.array([0  , Asum.shape[1]]) / escalax - nxorigin / escalax
# Plot the image with the correct aspect ratio
ymin,ymax = np.array([0, Asum.shape[0]]) / escalax - nyorigin / escalax


X,Y = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(ymin, ymax, (ymax-ymin)/Asum.shape[0]))
Asum_normalized = (Asum - Asum.min()) / (Asum.max() - Asum.min())
Asum_eq = exposure.equalize_adapthist(Asum_normalized, clip_limit=0.03)
#ax0.contourf(X/1,Y,Asum_eq**.4, cmap='inferno',levels=100,edgecolor='none')

cm0 = ax0.imshow(Asum_eq**.4, cmap='inferno',extent=(xmin, xmax/1, ymin, ymax),origin='lower')
ax0.set_xlabel('$x$ (mm)')
ax0.set_ylabel('$y$ (mm)')

ax0.set_ylim([-50,50])
ax0.set_aspect('equal')
for i in range(0,1000,100):
    ax0.plot(X[0]/1,(YT[i]-nyorigin)/escalax,'w.',markersize=.1)
ax0.plot(X[0]/1,(YT[3]-nyorigin)/escalax,'w.',markersize=1)
X,T = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(tmin, tmax, dt))
cm1 = ax1.contourf(T,X/1,(YT-nyorigin)/escalax/1/Lbandera, cmap='binary',levels=20)
ax1.set_xlabel('$t$ (s)')
ax1.set_ylabel('$x$ (mm)')
plt.colorbar(cm1, ax=ax1, label='$y/L$')

fig0.tight_layout()
fig1.tight_layout()
fig0.savefig('figures/ref_image_sum_full.png', dpi=300, bbox_inches='tight')
fig1.savefig('figures/ref_spatio_temporal.png', dpi=300, bbox_inches='tight')
figb, axb = plt.subplots()
YT_Fourier = np.fft.fft(YT, axis=0)
frec_YT = np.fft.fftfreq(YT.shape[0], d=1/fsampling)
FYT = np.abs(YT_Fourier).sum(axis=1)

axb.semilogy(frec_YT, FYT)

peak_freqs, _ = find_peaks(FYT[frec_YT>2],height=0.5*np.max(FYT[frec_YT>2]))
Frecuencia = frec_YT[frec_YT>2][peak_freqs][0]
print(f"Frecuencia de la señal: {Frecuencia:.2f} Hz")

axb.plot(Frecuencia, FYT[frec_YT>2][peak_freqs][0], 'ro')
y1,y2 = axb.get_ylim()
axb.plot([Frecuencia,Frecuencia] , [y1,FYT[frec_YT>2][peak_freqs][0]], 'r', linewidth=3,linestyle='dashed')
# axb.set_xticks([*axb.get_xticks(), Frecuencia])  # Agrega posición
axb.set_xlim([0,100])
xticks1 = axb.get_xticks()
axb.set_xticks(np.append(xticks1, Frecuencia))  # Agrega posición
axb.set_xticklabels([f"{tick:.0f}" for tick in axb.get_xticks()])  # Formatea etiquetas

 
axb.grid()
axb.set_xlabel('Frecuency (Hz)')
axb.set_ylabel('PSD')
figb.savefig('figures/Fourier_YT_full.png', dpi=300, bbox_inches='tight')