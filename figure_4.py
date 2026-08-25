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
ax0,ax1 = ax

xmin,xmax = np.array([0  , Asum.shape[1]]) / escalax - nxorigin / escalax
# Plot the image with the correct aspect ratio
ymin,ymax = np.array([0, Asum.shape[0]]) / escalax - nyorigin / escalax


X,Y = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(ymin, ymax, (ymax-ymin)/Asum.shape[0]))
Asum_normalized = (Asum - Asum.min()) / (Asum.max() - Asum.min())
Asum_eq = exposure.equalize_adapthist(Asum_normalized, clip_limit=0.03)


# ax0.set_ylim([-50,50])
# ax0.set_aspect('equal')

cmap = plt.colormaps['viridis']




X,T = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(tmin, tmax, dt))
# cm1 = ax1.contourf(T,X/1*1.06,(YT-nyorigin)/escalax/1/Lbandera, cmap='viridis',levels=20)
# pcolormesh con rasterized=True baja el peso de 3MB a menos de 100KB y mantiene la calidad intacta
cm1 = ax1.pcolormesh(T, X/1*1.06, (YT-nyorigin)/escalax/1/Lbandera, cmap='viridis', shading='auto', rasterized=True)
ax1.set_xlabel(r'$\mathrm{time~[s]}$' )
ax1.set_ylabel(r'$x~ [\mathrm{mm}]$' )
cbar1 = plt.colorbar(cm1, ax=ax1, label='$y/L$')
cbar1.set_label(r'$y/L$')
cbar1.ax.tick_params(labelsize=14)


rhoa = 1.2
rhoa_b = 1.0888  #densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa
Uinf = 12
delta_cl = 18e-3 # espesor de capa limite para velocidad 12m/s

#longitud caracteristica de la placa plana (tunel) en base a la medicion en Balseiro
x_carac = longitud_equivalente_capa_limite_turbulenta(delta_cl,Uinf,nu)
U = 12
delta_U12 = delta_turb(x_carac,U,nu)



fsampling = 1000 # Hz
escalax = 1/0.138 # px/mm
escalax = 1/0.130
Lbandera = 138.5 # mm
Lbandera = 130
dx = 1/escalax


Papel_80.L = Lbandera*1e-3  # Convertir a metros
Papel_80.freq_nat()
fn = np.zeros((3,1))
for i in range(3):
    fn[i] = Papel_80.fn[i]


f_offset  = [11.9,  11.4,16.6]
f_onset  = [16.3,20.0, 25.9 ]
u_offset, u_onset = np.tile(np.zeros_like(f_offset),[2,1])
for i,f_offi, f_oni in zip (range(len(f_offset)),f_offset,f_onset):
    u_offset[i] = veloc_tunel_ib(f_offi)
    u_onset[i] = veloc_tunel_ib(f_oni)


caso = 'rect'
caso = 'triang'
caso = 'full'

if caso == 'full':
    npoints = 5
    frec_c = 12.7
elif caso == 'triang':
    npoints = 5
    frec_c = 13


lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*.npz'))

lista_caso_2d = np.delete(lista_caso_2d,[2,7,8])



Velocidad_full, Amplitud_full, Frecuencia_full = np.zeros((3,len(lista_caso_2d)))

Velocidad_full, Amplitud_full, Frecuencia_full = lee_datos_foil(lista_caso_2d,Velocidad_full,Amplitud_full,Frecuencia_full)
plt.close('All')

gc.collect()


Amplitud_full = Amplitud_full/Lbandera
Uc = veloc_tunel_ib(frec_c)
U = Velocidad_full - Uc
Velocidad_m = Velocidad_full/2
p1 = np.polyfit(U[:npoints]**.5, Amplitud_full[:npoints]/2,1)
fun_Amplitud = np.poly1d(p1)




UB = 1/L * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5

sigma = rhoa_b*L/(rho_papel*Papel_80.thickness)


i = 0

lin4, = ax0.plot(Velocidad_full/2 /UB, Amplitud_full/2, marker='s', fillstyle='none',markeredgewidth=2,markersize=8,linestyle='none')
ax0.set_ylim([0,.5])

c4 = lin4.get_color()
ax0.plot(u_onset [i]/2/UB,0,'o',markersize=10,fillstyle='none',color=c4 )
ax0.plot(u_offset[i]/2/UB,0,'o',markersize=10,fillstyle='none',color=c4 )
ax0.annotate("",xytext=(u_offset[i]/2/UB,0 ),xy=(u_offset[i]/2/UB ,Amplitud_full[0]/2), arrowprops=dict(arrowstyle="<-", lw=1.5, mutation_scale=20,shrinkA=5,color=c4,ls='dashed'),color=c4)
ax0.annotate("",xytext=(u_onset [i]/2/UB,0 ),xy=(u_onset [i]/2/UB ,Amplitud_full[-3]/2), arrowprops=dict(arrowstyle="->", lw=1.5, mutation_scale=20,shrinkA=5,color=c4),color=c4)


ax0.text(u_offset[i]/2/UB,0.1,r'$u^*_{\mathrm{off}}$'+f'$ = {np.round(u_offset[i]/2/UB,1):.2f}$',fontsize=16,bbox=dict(boxstyle="round,pad=0.3", fc="tab:blue", alpha=0.2, ec="b", lw=.2))
ax0.text(u_onset[i]/2/UB+0.5,0.3,r'$u^*_{\mathrm{on}}$'+f'$ = {np.round(u_onset [i]/2/UB,1):.2f}$',fontsize=16,bbox=dict(boxstyle="round,pad=0.3", fc="tab:blue", alpha=0.2, ec="b", lw=.2))
ax0.grid(which='both')
ax0.set_ylabel('$A/2L$')
ax0.set_xlabel(r'$u^*=u_\infty/2 L\sqrt{\rho_s e/B}$')
# fig10.tight_layout()

# plt.close('all')
fig.savefig(dirout+'amplitude_full&spatio_temporal.pdf',dpi=150, bbox_inches='tight')

