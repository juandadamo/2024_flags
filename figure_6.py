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
ax3,ax4 = axb
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
fig.savefig(dirout+'snapshots_eigenvalues_pod.pdf',dpi=150, bbox_inches='tight')
