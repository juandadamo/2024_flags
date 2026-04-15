import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import exposure
# from tikzplotlib import save as tikz_save   
from matplotlib import rcParams
from funciones_flag import *
from tikzplotlib import save as tikz_save
dirout = '/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/'
dirout2 = '/home/juan/Documents/Publicaciones/2026_shear_flutter/tikzs/'
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
#
# cm0 = ax0.imshow(Asum_eq**.4, cmap='gist_gray_r',extent=(xmin, xmax/1, ymin, ymax),origin='lower',vmin=0.1,vmax=1)
#
# mask = Asum_eq > 0.4  # ajustá el umbral
# Asum_eq_masked = np.where(mask, Asum_eq**.4, 0)
# cm0 = ax0.imshow(Asum_eq_masked, cmap='gist_gray_r',
#                  extent=(xmin, xmax/1, ymin, ymax),
#                  origin='lower', vmin=0, vmax=1)

# cm0b = ax0.imshow(Asum_eq, cmap='gist_gray_r',
#                   extent=(xmin, xmax/1, ymin, ymax),
#                   origin='lower', vmin=0.075, vmax=1,alpha=1)
ax0.set_xlabel('$x$ (mm)')
ax0.set_ylabel('$y$ (mm)')

# ax0.set_ylim([-50,50])
# ax0.set_aspect('equal')

cmap = plt.colormaps['viridis']


# Generate N evenly spaced values from 0 to 1
N = 11
gradient_values = np.linspace(0, 1, N)

# Get the RGBA colors
rgba_colors = cmap(gradient_values[::])



for ii,i in enumerate(range(0,1000,1)):
    ax0.plot(X[0]*1.06/1,(YT[i]-nyorigin)/escalax,marker='.',color='lightgray'
             ,markersize=.1,linestyle='none',alpha=0.5)

for ii,i in enumerate(range(50,999,100)):
    if ii==9:
        ax0.plot(X[0]*1.06/1,(YT[i]-nyorigin)/escalax,marker='.',color=rgba_colors[ii]
             ,markersize=5,linestyle='none')
    else:
        ax0.plot(X[0]*1.06/1,(YT[i]-nyorigin)/escalax,marker='.',color=rgba_colors[ii]
             ,markersize=1,linestyle='none')
    print(YT[i].sum())

# ax0.plot(X[0]/1,(YT[i]-nyorigin)/escalax,rgba_colors[ii+1],markersize=3,linestyle='none',marker='s')
X,T = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(tmin, tmax, dt))
cm1 = ax1.contourf(T,X/1*1.06,(YT-nyorigin)/escalax/1/Lbandera, cmap='viridis',levels=20)
ax1.set_xlabel('$t$ (s)')
ax1.set_ylabel('$x$ (mm)')
plt.colorbar(cm1, ax=ax1, label='$y/L$')
ax0.set_ylim([-35,25])
ax0.set_xlim([0,140])
ax0.grid()

fig0.tight_layout()
fig1.tight_layout()
# fig0.savefig(dirout+'ref_image_sum_full.png', dpi=300, bbox_inches='tight')
# fig1.savefig(dirout+'spatio_temporal.png', dpi=300, bbox_inches='tight')




#figb.savefig('figures/Fourier_YT_full.png', dpi=300, bbox_inches='tight')


A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']
Imagen_sum = A['Imagen_sum']
A_curva_i = A['A_curva_i']
n1,n2 = [46,1045]#1146
U,S,Vh = np.linalg.svd((A_curva_i-A_curva_i.mean(0))[:,n1:n2])

fig2,ax2 = plt.subplots(figsize=(5*1.2,5))

#
# for ii,i in enumerate(range(0,1000,1)):
#     ax0.plot(X[0]*1.06/1,(YT[i]-nyorigin)/escalax,marker='.',color='lightgray'
#              ,markersize=.1,linestyle='none',alpha=0.5)

nstep= 4
nfinal = 100
for ii,i in enumerate(range(0,nfinal,nstep)):
    ax2.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT[i][n1:n2]-nyorigin)/escalax/Lbandera,color='k',
             linestyle='-',linewidth=2)

s = 100 #modos retenidos
YT_r = np.dot(U[:,:s],np.dot(Vh[:s].T,np.diag(S[:s])).T)

fig3,ax3 = plt.subplots(figsize=(5*1.2,5))
for ii,i in enumerate(range(0,nfinal,nstep)):
    ax3.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT_r[i] )/escalax/Lbandera,color='k',
             linestyle='-',linewidth=2)

for axi in [ax2,ax3]:
    axi.set_ylim([-.25,.15])
    axi.grid('gray')
    axi.set_ylabel('$y/L$')
    axi.set_xlabel('$x/L$')

fig2.tight_layout()
fig3.tight_layout()
fig2.savefig(dirout+'full134_snapshots.pdf')
fig3.savefig(dirout+'full134_snapshots_r.pdf')


fig4, ax4 = plt.subplots()
YT_r_Fourier = np.fft.fft(YT_r, axis=0)
frec_YT_r = np.fft.fftfreq(YT_r.shape[0], d=1/fsampling)
FYT_r = np.abs(YT_r_Fourier).sum(axis=1)

ax4.semilogy(frec_YT_r, FYT_r)





fig5, ax5 = plt.subplots()
YT_Fourier = np.fft.fft(YT-YT.mean(0), axis=0)
frec_YT = np.fft.fftfreq(YT.shape[0], d=1/fsampling)
FYT = np.abs(YT_Fourier).sum(axis=1)

ax5.semilogy(frec_YT, FYT)

peak_freqs, _ = find_peaks(FYT[frec_YT>2],height=0.5*np.max(FYT[frec_YT>2]))
Frecuencia = frec_YT[frec_YT>2][peak_freqs][0]
print(f"Frecuencia de la señal: {Frecuencia:.2f} Hz")

ax5.plot(Frecuencia, FYT[frec_YT>2][peak_freqs][0], 'ro')
y1,y2 = ax5.get_ylim()
ax5.plot([Frecuencia,Frecuencia] , [y1,FYT[frec_YT>2][peak_freqs][0]], 'r', linewidth=3,linestyle='dashed')
# axb.set_xticks([*axb.get_xticks(), Frecuencia])  # Agrega posición
ax5.set_xlim([0,100])
xticks1 = ax5.get_xticks()
ax5.set_xticks(np.append(xticks1, Frecuencia))  # Agrega posición
ax5.set_xticklabels([f"{tick:.0f}" for tick in ax5.get_xticks()])  # Formatea etiquetas


for axi in [ax4,ax5]:
    axi.set_xlim([0,100])
    axi.set_ylim([5.5e5,1e8])
    axi.grid()
    axi.set_xlabel('Frecuency (Hz)')
    axi.set_ylabel('PSD')
fig4.tight_layout()
fig5.tight_layout()


fig6,ax6 = plt.subplots()
ax6.semilogy(S[:10],marker='o',fillstyle='none',linestyle='none',markersize=10)
ax6.grid(color='gray',linestyle='dotted',which='both')
ax6.set_ylabel(r'eigenvalues $\lambda$')
ax6.set_xlabel('mode number')
fig6.tight_layout()
tikz_save(dirout2+'eigenvalues_pod.tikz')
fig6.savefig(dirout+'eigenvalues_pod.pdf')


fig7,ax7 = plt.subplots()
Nsnapshots = len(U[:,0])
t_s = np.arange(0,Nsnapshots/fsampling,1/fsampling)
ax7.plot(t_s,U[:,0]*S[0]*Vh[0].max()/escalax/Lbandera,label='mode 1')
ax7.plot(t_s,U[:,1]*S[1]*Vh[1].max()/escalax/Lbandera,label='mode 2',linestyle='dashed')

s=2
YT_r = np.dot(U[:,:s],np.dot(Vh[:s].T,np.diag(S[:s])).T)

# ax7.plot(YT_r[:,980]/escalax/Lbandera)
# ax7.plot((YT[:,980]-YT.mean(0)[980])/escalax/Lbandera)
ax7.grid(color='gray',linestyle='dotted',which='both')
ax7.legend(ncols=2,fontsize=12)
ax7.set_ylabel(r'Amplitude')
ax7.set_xlabel('time [s]')
fig7.tight_layout()
tikz_save(dirout2+'modes_t_pod.tikz')
fig7.savefig(dirout+'modes_t_pod.pdf')


fig8,ax8 = plt.subplots()
amp_0 = U[:,0].max()*S[0]/escalax/Lbandera*(-2)
amp_1 = U[:,1].max()*S[1]/escalax/Lbandera*(-2)
ax8.plot(Vh[0]*amp_0 ,label='pod mode 1')
ax8.plot(amp_1*Vh[1],label='pod mode 2',linestyle='dashed')
# ax8.plot(S[2]*Vh[2]/escalax/Lbandera,label='mode 2',linestyle='dashdot')
x_s = np.linspace(0,1,200)
n_s = np.linspace(0,1000,200)
a_1 = w_n (BnL[0],x_s,A1=amp_0*0.5*Vh[0].max(),L=1)
a_2 = w_n (BnL[1],x_s,A1=amp_1*Vh[1].max(),L=1)
ax8.plot(n_s,a_1,linestyle='dotted',label='cantilever mode 1')
ax8.plot(n_s,a_2,linestyle='dashdot',label='cantilever mode 2')
ax8.legend(fontsize=12)
ax8.grid()
for axi in [ax8]:
    axi.set_ylim([-.25,.15])
    axi.grid('gray')
    axi.set_ylabel('$y/L$')
    axi.set_xlabel('$x/L$')
fig8.tight_layout()
tikz_save(dirout2+'modes_x_pod.tikz')
fig8.savefig(dirout+'modes_x_pod.pdf')



U0 = U[:,0]

U1 = U[:,1]
S_T = (YT.mean(1)-YT.mean())/YT.mean()
# Aplicar ventana de Hanning

ventana = np.hanning(len(U0))
U_ventaneada = U0 * ventana
U1_v = U1 * ventana
S_v = S_T * ventana
FU = np.fft.fft(U_ventaneada)
FU1 = np.fft.fft(U1_v)
FS = np.fft.fft(S_v)
frecuencias = np.fft.fftfreq(len(U), d=1/1000)  # ajusta 'd' según tu sampling

# Magnitud (solo frecuencias positivas)
magnitud = np.abs(FU[:len(U)//2])
magnitud1 = np.abs(FU1[:len(U1)//2])
magnitud_s = np.abs(FS[:len(FS)//2])
frecs_pos = frecuencias[:len(U)//2]
fig9,ax9 = plt.subplots()
ax9.semilogy(frecs_pos, magnitud,label = 'mode 1')
ax9.semilogy(frecs_pos, magnitud1,label = 'mode 2')
ax9.semilogy(frecs_pos, magnitud_s,label = 'full signal')
ax9.set_xlim([0,100])
ax9.set_ylim(bottom=1e-3)
ax9.legend(fontsize=12)
ax9.grid('gray',which='both')
ax9.set_xlabel('frequency')
ax9.set_ylabel('PSD')
ax9.plot([12,12],[1e-3,30.5],linestyle='dashed',color='k')
fig9.tight_layout()
# tikz_save(dirout2+'modes_fourier_pod.tikz')
fig9.savefig(dirout+'modes_fourier_pod.pdf')
#
# from scipy import signal
# fs=1000
# # Ejemplo comparativo
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#
# # Segmento pequeño (menos ruido, baja resolución)
# f, P = signal.welch(U0, fs=fs, nperseg=64)
# axes[0,0].semilogy(f, P)
# axes[0,0].set_title('nperseg = 64 (suave pero baja resolución)')
#
# # Segmento mediano
# f, P = signal.welch(U0, fs=fs, nperseg=256)
# axes[0,1].semilogy(f, P)
# axes[0,1].set_title('nperseg = 256 (buen equilibrio)')
#
# # Segmento grande (alta resolución, más ruido)
# f, P = signal.welch(U0, fs=fs, nperseg=512)
# axes[1,0].semilogy(f, P)
# axes[1,0].set_title('nperseg = 512 (alta resolución, más ruido)')
#
# # FFT clásica para comparar
# FU0 = np.fft.fft(U0)
# freqs = np.fft.fftfreq(len(U0), 1/fs)
# magnitud = np.abs(FU0[:len(U0)//2])
# axes[1,1].plot(freqs[:len(U0)//2], magnitud)
# axes[1,1].set_title('FFT clásica (máxima resolución, máximo ruido)')
# axes[1,1].set_yscale('log')
#
# plt.tight_layout()
#
