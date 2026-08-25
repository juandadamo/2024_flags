import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import exposure
# from tikzplotlib import save as tikz_save   
from matplotlib import rcParams
from funciones_flag import *
from tikzplotlib import save as tikz_save
import glob
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

fig,ax = plt.subplots(1,2,figsize=(13,5))
ax2,ax6 = ax

figb,axb = plt.subplots(1,2,figsize=(13,5))
ax8,ax7 = axb
figc,axc = plt.subplots(1,2,figsize=(13,5))
ax9,ax3 = axc
A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']
Imagen_sum = A['Imagen_sum']
A_curva_i = A['A_curva_i']
n1,n2 = [46,1045]#1146
U,S,Vh = np.linalg.svd((A_curva_i-A_curva_i.mean(0))[:,n1:n2])
xmin,xmax = np.array([0  , Asum.shape[1]]) / escalax - nxorigin / escalax
# Plot the image with the correct aspect ratio
ymin,ymax = np.array([0, Asum.shape[0]]) / escalax - nyorigin / escalax
X,T = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(tmin, tmax, dt))

energia = S / np.sum(S)
energia_acum = np.cumsum(energia) * 100
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


for ii,i in enumerate(range(0,nfinal,nstep)):
    ax3.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT_r[i] -nyorigin)/escalax/Lbandera,color='k',
             linestyle='-',linewidth=2)
ax3.plot(X[0][n1:n2]*1.06/1/Lbandera,(YT.mean(0)[n1:n2]-nyorigin)/escalax/Lbandera,
             linestyle='dashed',linewidth=4,color='tab:orange')
for axi in [ax2,ax3]:
    axi.set_ylim([-.25,.25])
    axi.set_xlim([0,.9])
    axi.grid('gray')
    axi.set_ylabel('$y/L$')
    axi.set_xlabel('$x/L$')


ax6.semilogy(S[:10],marker='o',fillstyle='none',linestyle='none',markersize=10)
ax6.grid(color='gray',linestyle='dotted',which='both')
ax6.set_ylabel(r'$\mathrm{eigenvalues~} \lambda$')
ax6.set_xlabel(r'$\mathrm{mode~ number}$')
ax6b = ax6.twinx()
ax6b.plot(energia_acum[:10], marker='s', fillstyle='none', linestyle='--',
          color='k', markersize=8, label=r'$\mathrm{Cumulative~ energy}$')
ax6b.set_ylabel(r'$\mathrm{cumulative~ energy}~ (\%)$')
ax6b.tick_params(axis='y', labelcolor='k')
ax6b.set_ylim(0, 110)



Nsnapshots = len(U[:,0])
t_s = np.arange(0,Nsnapshots/fsampling,1/fsampling)
scale_mod = Vh[0].max()/escalax/Lbandera
scale_mod_1 = 1/escalax/Lbandera
scale_mod_2 = 1/escalax/Lbandera
ax7.plot(t_s,U[:,0]*S[0]*scale_mod_1,label=r'$\mathrm{mode~ 1}$')
ax7.plot(t_s,U[:,1]*S[1]*scale_mod_2,label=r'$\mathrm{mode~ 2}$',linestyle='dashed')

s=2
YT_r = np.dot(U[:,:s],np.dot(Vh[:s].T,np.diag(S[:s])).T)

# ax7.plot(YT_r[:,980]/escalax/Lbandera)
# ax7.plot((YT[:,980]-YT.mean(0)[980])/escalax/Lbandera)
ax7.grid(color='gray',linestyle='dotted',which='both')
ax7.legend(ncols=2,fontsize=12)
ax7.set_ylabel(r'$\mathrm{Amplitude}$')
ax7.set_xlabel(r'$\mathrm{time~} [\mathrm{s}]$')


amp_0 = U[:,0].max()*S[0]/escalax/Lbandera*(-2)
amp_1 = U[:,1].max()*S[1]/escalax/Lbandera*(-2)
xn = X[0][n1:n2]*1.06/1/Lbandera
ax8.plot(xn,Vh[0]*amp_0 ,label=r'$\mathrm{pod~ mode~ 1}$')
ax8.plot(xn,amp_1*Vh[1],label=r'$\mathrm{pod~ mode~ 2}$',linestyle='dashed')
# ax8.plot(S[2]*Vh[2]/escalax/Lbandera,label='mode 2',linestyle='dashdot')
x_s = np.linspace(0,1,200)
n_s = np.linspace(0,1000,200)

a_1 = w_n (BnL[0],x_s,A1=amp_0*0.5*Vh[0].max(),L=1)
a_2 = w_n (BnL[1],x_s,A1=amp_1*Vh[1].max(),L=1)
ax8.plot(x_s,a_1,linestyle='dotted',label=r'$\mathrm{cantilever~ mode~ 1}$')
ax8.plot(x_s,a_2,linestyle='dashdot',label=r'$\mathrm{cantilever~ mode~ 2}$')
ax8.legend(fontsize=12)



ax8.grid()
for axi in [ax8]:
    axi.set_ylim([-.3,.2])
    axi.set_xlim([0,.9])
    axi.grid('gray')
    axi.set_ylabel('$y/L$')
    axi.set_xlabel('$x/L$')



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

ax9.semilogy(frecs_pos, magnitud,label = r'$\mathrm{mode~ 1}$')
ax9.semilogy(frecs_pos, magnitud1,label = r'$\mathrm{mode~ 2}$')
ax9.semilogy(frecs_pos, magnitud_s,label = r'$\mathrm{full~ signal}$')
ax9.set_xlim([0,100])
ax9.set_ylim(bottom=1e-3)
ax9.legend(fontsize=12)
ax9.grid('gray',which='both')
ax9.set_xlabel(r'$\mathrm{Frequency}~ [\mathrm{Hz}]$')
ax9.set_ylabel(r'$\mathrm{PSD\ [(m/s)^2/Hz]}$')
ax9.plot([12,12],[1e-3,30.5],linestyle='dashed',color='k')
fig.savefig(dirout+'snapshots_eigenvalues_pod.pdf',dpi=150, bbox_inches='tight')
figb.savefig(dirout+'modes_x_pod_modes_t.pdf',dpi=150, bbox_inches='tight')
figc.savefig(dirout+'modes_fourier_pod_snapshots_r.pdf',dpi=150, bbox_inches='tight')
