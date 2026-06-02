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

fig0,ax0 = plt.subplots(figsize=(6.75,5.5))
fig1,ax1 = plt.subplots(figsize=(6.75,5.5))

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
ax1.set_xlabel('$t$ (s)' )
ax1.set_ylabel('$x$ (mm)' )
cbar1 = plt.colorbar(cm1, ax=ax1, label='$y/L$')
cbar1.set_label('$y/L$')
cbar1.ax.tick_params(labelsize=14)
# cbar1.set_label('$y/L$', rotation=0, labelpad=15)
# cbar1.set_label('$y/L$', rotation=0, labelpad=15, ha='left')
# cbar1.ax.xaxis.set_label_position('top')
ax0.set_ylim([-35,25])
ax0.set_xlim([0,140])
ax0.grid()



# fig0.savefig(dirout+'ref_image_sum_full.png', dpi=300, bbox_inches='tight')
fig1.savefig(dirout+'spatio_temporal.pdf',dpi=150, bbox_inches='tight')




#figb.savefig('figures/Fourier_YT_full.png', dpi=300, bbox_inches='tight')


A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']
Imagen_sum = A['Imagen_sum']
A_curva_i = A['A_curva_i']
n1,n2 = [46,1045]#1146
U,S,Vh = np.linalg.svd((A_curva_i-A_curva_i.mean(0))[:,n1:n2])

fig2,ax2 = plt.subplots(figsize=(6.75,5.5))

#
# for ii,i in enumerate(range(0,1000,1)):
#     ax0.plot(X[0]*1.06/1,(YT[i]-nyorigin)/escalax,marker='.',color='lightgray'
#              ,markersize=.1,linestyle='none',alpha=0.5)

nstep= 4
nfinal = 100
for ii,i in enumerate(range(0,nfinal,nstep)):
    ax2.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT[i][n1:n2]-nyorigin)/escalax/Lbandera,color='k',
             linestyle='-',linewidth=2)

ax2.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT.mean(0)[n1:n2]-nyorigin)/escalax/Lbandera,
             linestyle='dashed',linewidth=4,color='tab:orange')

s = 2 #modos retenidos
YT_r = np.dot(U[:,:s],np.dot(Vh[:s].T,np.diag(S[:s])).T)+A_curva_i.mean(0)[n1:n2]


fig3,ax3 = plt.subplots(figsize=(6.75,5.5))
for ii,i in enumerate(range(0,nfinal,nstep)):
    ax3.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT_r[i] -nyorigin)/escalax/Lbandera,color='k',
             linestyle='-',linewidth=2)
ax3.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT.mean(0)[n1:n2]-nyorigin)/escalax/Lbandera,
             linestyle='dashed',linewidth=4,color='tab:orange')
for axi in [ax2,ax3]:
    axi.set_ylim([-.25,.15])
    axi.set_xlim([0,.9])
    axi.grid('gray')
    axi.set_ylabel('$y/L$')
    axi.set_xlabel('$x/L$')

# fig2.tight_layout()
# fig3.tight_layout()
fig2.savefig(dirout+'full134_snapshots.pdf',dpi=150, bbox_inches='tight')
fig3.savefig(dirout+'full134_snapshots_r.pdf',dpi=150, bbox_inches='tight')


fig4, ax4 = plt.subplots(figsize=(3,2.5))
YT_r_Fourier = np.fft.fft(YT_r, axis=0)
frec_YT_r = np.fft.fftfreq(YT_r.shape[0], d=1/fsampling)
FYT_r = np.abs(YT_r_Fourier).sum(axis=1)

ax4.semilogy(frec_YT_r, FYT_r)





fig5, ax5 = plt.subplots(figsize=(6.75,5.5))
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
    axi.set_xlabel('Frequency (Hz)')
    axi.set_ylabel('PSD')
# fig4.tight_layout()

fig5.savefig('/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/Fourier_YT_full.pdf',dpi=150, bbox_inches='tight')
energia = S / np.sum(S)
energia_acum = np.cumsum(energia) * 100  # porcentaje
fig6,ax6 = plt.subplots(figsize=(6.75,6))
ax6.semilogy(S[:10],marker='o',fillstyle='none',linestyle='none',markersize=10)
ax6.grid(color='gray',linestyle='dotted',which='both')
ax6.set_ylabel(r'eigenvalues $\lambda$')
ax6.set_xlabel('mode number')

ax6b = ax6.twinx()
ax6b.plot(energia_acum[:10], marker='s', fillstyle='none', linestyle='--',
          color='k', markersize=8, label='Cumulative energy')
ax6b.set_ylabel('cumulative energy (\%)')
ax6b.tick_params(axis='y', labelcolor='k')
ax6b.set_ylim(0, 110)

tikz_save(dirout2+'eigenvalues_pod.tikz')
fig6.savefig(dirout+'eigenvalues_pod.pdf',dpi=150, bbox_inches='tight')


fig7,ax7 = plt.subplots(figsize=(6.75,5.5))
Nsnapshots = len(U[:,0])
t_s = np.arange(0,Nsnapshots/fsampling,1/fsampling)
scale_mod = Vh[0].max()/escalax/Lbandera
scale_mod_1 = 1/escalax/Lbandera
scale_mod_2 = 1/escalax/Lbandera
ax7.plot(t_s,U[:,0]*S[0]*scale_mod_1,label='mode 1')
ax7.plot(t_s,U[:,1]*S[1]*scale_mod_2,label='mode 2',linestyle='dashed')

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
fig7.savefig(dirout+'modes_t_pod.pdf',dpi=150, bbox_inches='tight')


fig8,ax8 = plt.subplots(figsize=(6.75,5.5))
amp_0 = U[:,0].max()*S[0]/escalax/Lbandera*(-2)
amp_1 = U[:,1].max()*S[1]/escalax/Lbandera*(-2)
xn = X[0][n1:n2]*1.06/1/Lbandera
ax8.plot(xn,Vh[0]*amp_0 ,label='pod mode 1')
ax8.plot(xn,amp_1*Vh[1],label='pod mode 2',linestyle='dashed')
# ax8.plot(S[2]*Vh[2]/escalax/Lbandera,label='mode 2',linestyle='dashdot')
x_s = np.linspace(0,1,200)
n_s = np.linspace(0,1000,200)

a_1 = w_n (BnL[0],x_s,A1=amp_0*0.5*Vh[0].max(),L=1)
a_2 = w_n (BnL[1],x_s,A1=amp_1*Vh[1].max(),L=1)
ax8.plot(x_s,a_1,linestyle='dotted',label='cantilever mode 1')
ax8.plot(x_s,a_2,linestyle='dashdot',label='cantilever mode 2')
ax8.legend(fontsize=12)



ax8.grid()
for axi in [ax8]:
    axi.set_ylim([-.3,.2])
    axi.set_xlim([0,.9])
    axi.grid('gray')
    axi.set_ylabel('$y/L$')
    axi.set_xlabel('$x/L$')
fig8.tight_layout()
tikz_save(dirout2+'modes_x_pod.tikz')
fig8.savefig(dirout+'modes_x_pod.pdf',dpi=150, bbox_inches='tight')



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
fig9,ax9 = plt.subplots(figsize=(6.75,5.5))
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

# fig9.tight_layout()
# tikz_save(dirout2+'modes_fourier_pod.tikz')
fig9.savefig(dirout+'modes_fourier_pod.pdf',dpi=150, bbox_inches='tight')

