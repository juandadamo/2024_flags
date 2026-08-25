import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
#%matplotlib widget
import serial,socket,os,glob,sys
#import atexit
import numpy as np
import pandas as pd
import time, threading,sys,glob
colores = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
import tifffile as tif
import skimage as sk
from IPython.display import Latex
from funciones_flag import *
from scipy.signal import find_peaks
mks = ['o','^','s','p','v','^','*']
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from tikzplotlib import save as tikz_save
if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2026_shear_flutter/figures/'
    dir_tik = '/home/juan/Documents/Publicaciones/2026_shear_flutter/tikzs/'

plt.close('all')
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
# Opcional: Forzar la recolección de basura (gc) para liberar memoria
import gc
gc.collect()

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
# Contenido en frecuencia de la señal!!!!!!!


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig4,ax4 = plt.subplots(figsize=(6.75,5.5))
fig4.canvas.setWindowTitle('Freq')
# deltaw = delta_turb(x_carac,Velocidad,nu)
ax4.plot(Velocidad_full/UB/2, Frecuencia_full*Lbandera*1e-3/(Velocidad_m), marker='s',markersize=10, fillstyle='none',
         linestyle='none')
ax4.grid()
#ax4.set_ylim([0.045, 0.065])
ax4.set_ylabel(r'St = $f_{foil}L/u_m$')
# ax4.set_xlabel(r'$\sqrt{U-U_c}$ [m/s$^{1/2}$]')
ax4.set_xlabel(r'$u^*$')
# ax4.set_xlim([0,2.5])
axins4 = inset_axes(ax4, width="70%", height="70%",bbox_to_anchor=(0.5, 0.12, 0.65, 0.65),bbox_transform=ax4.transAxes,loc="lower left")
axins4.plot(Velocidad_full,Frecuencia_full, 'ks', fillstyle='none')
axins4.grid(which='both')
axins4.set_xlabel(r'$U$ [m/s]',fontsize=14)
axins4.set_ylabel(r'$f_{foil}$ [hz]',fontsize=14)
axins4.set_ylim([9,26])
# fig4.tight_layout()

#axins4.grid(which='both', visible=True)
#fig4.canvas.draw()

fig4.savefig(dirw+'full_frequency.pdf', dpi=150, bbox_inches='tight')




fig6,ax6 = plt.subplots()


ax6.plot(Frecuencia_full*Lbandera*1e-3/(Velocidad_m),Amplitud_full/2, 'ks', fillstyle='none')
ax6.grid()
ax6.set_ylabel(r'$A/2L$')
ax6.set_xlabel(r'$f_{foil}L/u_\infty$')
ax6.set_ylim([0,0.4])
ax6.set_yticks(np.arange(0, 0.9, 0.1))    
fig6.tight_layout()
# fig6.savefig(dirw+'Freq_Amp'+caso+'.png',dpi=300, bbox_inches='tight')

((Papel_80.E*Papel_80.thickness**3) / (rhoa*Papel_80.L**3))**0.5


fig7,ax7 = plt.subplots()
ax7.plot(Velocidad_full,Frecuencia_full, 'ks', fillstyle='none')
ax7.grid()
ax7.set_xlabel(r'$U$ [m/s]')
ax7.set_ylabel(r'$f_{foil}$ [hz]')
ax7.set_ylim([9,26])
ax7.set_xlim([6,15])
# ax7.set_yticks(np.arange(0, 0.9, 0.1))    
fig7.tight_layout()

fig8,ax8 = plt.subplots()

ax8.plot(Velocidad_full/UB/2,Amplitud_full/2, 'ks', fillstyle='none')

ax8.grid()
ax8.set_ylabel(r'$A/2L$')
ax8.set_xlabel(r'$u^*$')
ax8.set_ylim([0,0.4])
#ax6.set_yticks(np.arange(0, 0.9, 0.1))
fig8.tight_layout()
# fig6.savefig(dirw+'Freq_Amp'+caso+'.png',dpi=300, bbox_inches='tight')
# fig7.savefig(dirw+'Freq_Veloc_'+caso+'.png',dpi=300, bbox_inches='tight')

fig9,ax9 = plt.subplots()

ax9.plot(Velocidad_full/UB/2,(Amplitud_full/2)**2, 'ks', fillstyle='none')

ax9.grid()
ax9.set_ylabel(r'$(A/2L)^2$')
ax9.set_xlabel(r'$ u^*$')
ax9.set_ylim([0,0.2])
#ax6.set_yticks(np.arange(0, 0.9, 0.1))
fig9.tight_layout()


i = 0
fig10,ax10 = plt.subplots(figsize=(6,5))
lin4, = ax10.plot(Velocidad_full/2 /UB, Amplitud_full/2, marker='s', fillstyle='none',markeredgewidth=2,markersize=8,linestyle='none')
ax10.set_ylim([0,.5])

c4 = lin4.get_color()
ax10.plot(u_onset [i]/2/UB,0,'o',markersize=10,fillstyle='none',color=c4 )
ax10.plot(u_offset[i]/2/UB,0,'o',markersize=10,fillstyle='none',color=c4 )
ax10.annotate("",xytext=(u_offset[i]/2/UB,0 ),xy=(u_offset[i]/2/UB ,Amplitud_full[0]/2), arrowprops=dict(arrowstyle="<-", lw=1.5, mutation_scale=20,shrinkA=5,color=c4,ls='dashed'),color=c4)
ax10.annotate("",xytext=(u_onset [i]/2/UB,0 ),xy=(u_onset [i]/2/UB ,Amplitud_full[-3]/2), arrowprops=dict(arrowstyle="->", lw=1.5, mutation_scale=20,shrinkA=5,color=c4),color=c4)


ax10.text(u_offset[i]/2/UB,0.1,f'$u^*_{{offset}}$ = {np.round(u_offset[i]/2/UB,1):.2f}',fontsize=16,bbox=dict(boxstyle="round,pad=0.3", fc="tab:blue", alpha=0.2, ec="b", lw=.2))
ax10.text(u_onset[i]/2/UB+0.5,0.3,f'$u^*_{{onset}}$ = {np.round(u_onset [i]/2/UB,1):.2f}',fontsize=16,bbox=dict(boxstyle="round,pad=0.3", fc="tab:blue", alpha=0.2, ec="b", lw=.2))
ax10.grid(which='both')
ax10.set_ylabel('$A/2L$')
ax10.set_xlabel(r'$u^*=u_\infty/2 L\sqrt{\rho_s e/B}$')
# fig10.tight_layout()

# plt.close('all')
tikz_save(dir_tik+'bistability_full.tikz')
fig10.savefig(dirw+'bistability_full.pdf',dpi=150, bbox_inches='tight')

caso = 'triang'

lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*.npz'))
Velocidad_triang, Amplitud_triang, Frecuencia_triang = np.zeros((3,len(lista_caso_2d)))

Velocidad_triang, Amplitud_triang, Frecuencia_triang = lee_datos_foil(lista_caso_2d,Velocidad_triang,Amplitud_triang,Frecuencia_triang)

caso = 'rect'

lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*.npz'))
Velocidad_rect, Amplitud_rect, Frecuencia_rect = np.zeros((3,len(lista_caso_2d)))

Velocidad_rect, Amplitud_rect, Frecuencia_rect = lee_datos_foil(lista_caso_2d,Velocidad_rect,Amplitud_rect,Frecuencia_rect)

Amplitud_rect = Amplitud_rect/Lbandera
Amplitud_triang = Amplitud_triang/Lbandera

fig,ax = plt.subplots(1,2,figsize=(13,5))
ax1,ax2 = ax
# fig1, ax1 = plt.subplots(figsize=(6.75,5.5))
# fig2, ax2 = plt.subplots(figsize=(6.75,5.5))
lin1, = ax1.plot(Velocidad_full/2/UB,Amplitud_full/2,'o',fillstyle='none',markersize=10,markeredgewidth=2,zorder=10)
lin2, = ax1.plot(Velocidad_triang/2/UB,Amplitud_triang/2,'^',fillstyle='none',markersize=10,markeredgewidth=2,zorder=10)
lin3, = ax1.plot(Velocidad_rect/2/UB,Amplitud_rect/2,'s',fillstyle='none',markersize=10,markeredgewidth=2,zorder=10)
color_i = []
for lin_i in [lin1,lin2,lin3]:
    color_i.append(lin_i.get_color())


ax2.plot(Velocidad_full/2/UB,Frecuencia_full*Lbandera*1e-3/(Velocidad_full/2),'o',fillstyle='none',markersize=10,markeredgewidth=2)

ax2.plot(Velocidad_triang/2/UB,Frecuencia_triang*Lbandera*1e-3/(Velocidad_triang/2),'^',fillstyle='none',markersize=10,markeredgewidth=2)

ax2.plot(Velocidad_rect/2/UB,Frecuencia_rect*Lbandera*1e-3/(Velocidad_rect/2),'s',fillstyle='none',markersize=10,markeredgewidth=2)


for axi in [ax1,ax2]:
    axi.grid('gray',which='both')
    axi.set_xlabel(r'$u^*$')
ax1.set_ylabel(r'$A/2L$')
ax2.set_ylabel(r'$f_{foil}L/u_m$')
ax2.set_ylim([0.,0.5])
ax2.set_ylim([0.1,0.7])

# tikz_save('/home/juan/Documents/Publicaciones/2026_shear_flutter/tikzs/bistability_full.tikz')
# fig10.savefig('/home/juan/Documents/Publicaciones/2026_shear_flutter/tikzs/bistability_full.pdf')
data_all = ([Velocidad_full, Amplitud_full, Frecuencia_full],[Velocidad_triang, Amplitud_triang, Frecuencia_triang],[Velocidad_rect, Amplitud_rect, Frecuencia_rect])



for i in np.arange(len(f_offset)):

    amp_i = data_all[i][1][-3:].mean()
    amp_i0 = data_all[i][1][0]
    ax1.plot(u_onset[i]/2/UB,0,marker=mks[i],markersize=10,fillstyle='none',color=color_i[i] )
    ax1.plot(u_offset[i]/2/UB,0,marker=mks[i],markersize=10,fillstyle='none',color=color_i[i],markeredgewidth=2 )
    ax1.annotate("",xytext=(u_offset[i]/2/UB,0 ),xy=(u_offset[i]/2/UB ,amp_i0/3), arrowprops=dict(arrowstyle="<-", lw=3.5, mutation_scale=20,shrinkA=5,color=color_i[i],ls='dashed'),color=color_i[i])
    if i<2:
            ax1.annotate("",xytext=(u_onset [i]/2/UB,0 ),xy=(u_onset[i]/2/UB ,amp_i/2), arrowprops=dict(arrowstyle="->", lw=3.5, mutation_scale=20,shrinkA=5,color=color_i[i]),color=color_i[i],zorder=-1)
    elif i==2:
            ax1.annotate("",xytext=(u_onset [i]/2/UB,0 ),xy=(u_onset[i]/2/UB ,0.25), arrowprops=dict(arrowstyle="->", lw=3.5, mutation_scale=20,shrinkA=5,color=color_i[i]),color=color_i[i],zorder=-1)

fig.savefig(dirw+'amplitudes_frecs_all.pdf',dpi=300, bbox_inches='tight')
#fig2.savefig(dirw+'frecs_all.pdf',dpi=300, bbox_inches='tight')

fig3,ax3 = plt.subplots()

ax3.plot(Velocidad_full/2/UB,Frecuencia_full ,'o',fillstyle='none',markersize=10)
ax3.plot(Velocidad_triang/2/UB,Frecuencia_triang ,'^',fillstyle='none',markersize=10)
ax3.plot(Velocidad_rect/2/UB,Frecuencia_rect ,'s',fillstyle='none',markersize=10)


data_0 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=0)  # lisa
data_1 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=1)  # Aserrada
data_2 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=2)  # Almenada
data_3 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=3)  # Almenada_old
data_4 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=4)  # Almenada2_old
data_5 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=5)  # Aserrada_old
data_6 = pd.read_excel('almenado_data.xlsx',header=1,sheet_name=6)  # Plana_old


u_lisa  = data_0[2:19]['Velocidad'].to_numpy()
dx_lisa = data_0[2:19]['Dx [mm]'].to_numpy()

u_aserrada  = data_1[2:15]['Velocidad'].to_numpy()
dx_aserrada = data_1[2:15]['Dx [mm]'].to_numpy()

data_2 = data_2.drop(10)
u_almenada  = data_2[2:13]['Velocidad'].to_numpy()
dx_almenada = data_2[2:13]['Dx [mm]'].to_numpy()

u_almenada_old = data_3[1:12]['Velocidad'].to_numpy()
dx_almenada_old = data_3[1:12]['Dx [mm]'].to_numpy()



lin0, = ax1.plot(u_lisa/2/UB,dx_lisa/Lbandera/2,'o',linestyle='none')
lin1, = ax1.plot(u_aserrada/2/UB,dx_aserrada/Lbandera/2,'^',linestyle='none')
lin2, = ax1.plot(u_almenada/2/UB,dx_almenada/Lbandera/2,'s',linestyle='none')
lin3, = ax1.plot(u_almenada_old/2/UB,dx_almenada_old/Lbandera/2,'s',linestyle='none',fillstyle='none',color=lin2.get_color())

# fig1.savefig(dirw+'amplitudes_all_v2.pdf',dpi=300, bbox_inches='tight')
fig.savefig(dirw+'amplitudes_frecs_all_v2.pdf',dpi=300, bbox_inches='tight')

