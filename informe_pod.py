import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage import exposure
# from tikzplotlib import save as tikz_save   
from matplotlib import rcParams
from funciones_flag import *
from tikzplotlib import save as tikz_save
import tikzplotlib


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
xmin,xmax = np.array([0  , Asum.shape[1]]) / escalax - nxorigin / escalax
# Plot the image with the correct aspect ratio
ymin,ymax = np.array([0, Asum.shape[0]]) / escalax - nyorigin / escalax


X,Y = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(ymin, ymax, (ymax-ymin)/Asum.shape[0]))
A = np.load('data_out/full_freq_13.4.npz')
Asum = A['Imagen_sum']
YT = A['A_curva_i']
Imagen_sum = A['Imagen_sum']
A_curva_i = A['A_curva_i']
n1,n2 = [46,1045]#1146
U,S,Vh = np.linalg.svd((A_curva_i-A_curva_i.mean(0))[:,n1:n2])


nstep= 1
nfinal = 100

s = 2 #modos retenidos
YT_r = np.dot(U[:,:s],np.dot(Vh[:s].T,np.diag(S[:s])).T)+A_curva_i.mean(0)[n1:n2]






fig1a, ax1a = plt.subplots(figsize=(6,5))
fig1b, ax1b = plt.subplots(figsize=(6,5))

nstep = 1
for i,Vhi in enumerate(Vh[:4]):
    ax1a.plot(Vhi[::nstep],linewidth=2)
    ax1b.plot(U[::nstep,i]*S[i],linewidth=2)
for axi in [ax1a,ax1b]:
    axi.grid()


U0,S0,Vh0 = np.linalg.svd(A_curva_i[:,n1:n2])
fig2a, ax2a = plt.subplots(figsize=(6,5))
fig2b, ax2b = plt.subplots(figsize=(6,5))

for i,Vhi in enumerate(Vh0[:5]):
    if i ==0:
        ax2a.plot(Vhi[::],linewidth=2,color='k')
        ax2b.plot(U0[::nstep,i]*S0[i],linewidth=2,color='k')
    else:
        ax2a.plot(Vhi[::],linewidth=2)
        ax2b.plot(U0[::nstep,i]*S0[i],linewidth=2)




for axi in [ax2a,ax2b]:
    axi.grid()
ax2a.set_ylim([-0.08,0.08])
ax1a.set_ylim([-0.08,0.08])
ax1a.set_ylabel(r'$\phi_i(x)$')
ax2a.set_ylabel(r'$\phi_i(x)$')
ax1a.set_xlabel(r'$x/L$')
ax2a.set_xlabel(r'$x/L$')

ax1b.set_ylabel(r'$a_i(t_i)$')
ylim_at = ax1b.get_ylim()
ax1b.set_ylim(ax2b.get_ylim())
ax2b.set_ylabel(r'$a_i(t_i)$')
ax1b.set_xlabel(r'$n_i$')
ax2b.set_xlabel(r'$n_i$')



for i,figi in enumerate([fig1a,fig1b,fig2a,fig2b]):
    figi.tight_layout()
    figi.savefig(dirout+f'analisis_pod_0{i:0d}.pdf')
    if i==1:
        ax1b.set_ylim(ylim_at)
        figi.savefig(dirout+f'analisis_pod_at.pdf')

fig3,ax3 = plt.subplots()


x1 = np.linspace(0,1,len(A_curva_i[0][n1:n2]))
Am = (A_curva_i.mean(0)-nyorigin)/escalax/Lbandera
for Ai in A_curva_i[:75:3]:
    Ai = (Ai-nyorigin)/escalax/Lbandera
    ax3.plot(x1,Ai[n1:n2],color='tab:blue',marker='o',markersize=.5,linestyle='none')
ax3.plot(x1,Am[n1:n2],marker='o',markersize=2,linestyle='none',color='tab:orange')
#tikz_save(dirout2+'analisis_pod_02.tikz')
ax3.grid()
ax3.set_xlabel(r'$x/L$')
ax3.set_ylabel(r'$y/L$')
fig3.tight_layout()
fig3.savefig(dirout+'snaps_media.pdf')

fig4,ax4  = plt.subplots()
ax4.semilogy(S[:50],linestyle='none',marker='o',fillstyle='none',markersize=10,markeredgewidth=2,label=r'$\Lambda$')
ax4.grid(which= 'both')


ax4.semilogy(S0[:50],linestyle='none',marker='s',fillstyle='none',markersize=10,markeredgewidth=2,label=r'$\Lambda_0$')
ax4.set_xlim([-.5,20])
ax4.set_ylabel(r'$\lambda$')
ax4.set_xlabel(r'n modo')
ax4.legend()
fig4.tight_layout()
fig4.savefig(dirout+'eigen_pod_00.pdf')

fig5,ax5 = plt.subplots(figsize=(6,5))

x_s = np.linspace(0,1,999)
n_s = np.arange(0,999,1)
a_s = np.zeros((4,len(x_s)))

for i  in range(4):
    a_s[i] = w_n (BnL[i],x_s,A1=1,L=1)
    a_s[i] = a_s[i] / np.linalg.norm(a_s[i])



    ax5.plot(n_s,a_s[i],linestyle='-',label=f' mode {i:0d}',linewidth=2)
ax5.set_ylim([-0.08,0.08])
ax5.grid(which='both')
ax5.set_xlabel(r'$x/L$')
ax5.set_ylabel(r'$\phi_i(x)$')
fig5.tight_layout()
fig5.savefig(dirout+'cantilever.pdf')
