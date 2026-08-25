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


# ax1.set_ylim([-50,50])
# ax1.set_aspect('equal')

cmap = plt.colormaps['viridis']




X,T = np.meshgrid(np.arange(xmin, xmax, (xmax-xmin)/Asum.shape[1]),
                  np.arange(tmin, tmax, dt))


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



from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# deltaw = delta_turb(x_carac,Velocidad,nu)
ax1.plot(Velocidad_full/UB/2, Frecuencia_full*Lbandera*1e-3/(Velocidad_m), marker='s',markersize=10, fillstyle='none',
         linestyle='none')
ax1.grid()
#ax1.set_ylim([0.045, 0.065])
ax1.set_ylabel(r'$\mathrm{St} = f_{\mathrm{foil}}L/u_m$')
# ax1.set_xlabel(r'$\sqrt{U-U_c}$ [m/s$^{1/2}$]')
ax1.set_xlabel(r'$u^*$')
# ax1.set_xlim([0,2.5])
axins4 = inset_axes(ax1, width="70%", height="70%",bbox_to_anchor=(0.5, 0.12, 0.65, 0.65),bbox_transform=ax1.transAxes,loc="lower left")
axins4.plot(Velocidad_full,Frecuencia_full, 'ks', fillstyle='none')
axins4.grid(which='both')
axins4.set_xlabel(r'$U~ [\mathrm{m/s}]$',fontsize=14)
axins4.set_ylabel(r'$f_{\mathrm{foil}}~ [\mathrm{Hz}]$',fontsize=14)
axins4.set_ylim([9,26])





YT_Fourier = np.fft.fft(YT-YT.mean(0), axis=0)
frec_YT = np.fft.fftfreq(YT.shape[0], d=1/fsampling)
FYT = np.abs(YT_Fourier).sum(axis=1)

ax0.semilogy(frec_YT, FYT)

peak_freqs, _ = find_peaks(FYT[frec_YT>2],height=0.5*np.max(FYT[frec_YT>2]))
Frecuencia = frec_YT[frec_YT>2][peak_freqs][0]
print(f"Frecuencia de la señal: {Frecuencia:.2f} Hz")

ax0.plot(Frecuencia, FYT[frec_YT>2][peak_freqs][0], 'ro')
y1,y2 = ax0.get_ylim()
ax0.plot([Frecuencia,Frecuencia] , [y1,FYT[frec_YT>2][peak_freqs][0]], 'r', linewidth=3,linestyle='dashed')
# axb.set_xticks([*axb.get_xticks(), Frecuencia])  # Agrega posición
ax0.set_xlim([0,100])
xticks1 = ax0.get_xticks()
xticks1 = np.arange(0,125,25)
ax0.set_xticks(np.append(xticks1, Frecuencia))  # Agrega posición
ax0.set_xticklabels([f"{tick:.0f}" for tick in ax0.get_xticks()])  # Formatea etiquetas



ax0.set_xlim([0,100])
ax0.set_ylim([5.5e5,1e8])
ax0.grid()
#ax0.set_xlabel('Frequency (Hz)')
ax0.set_xlabel(r'$\mathrm{Frequency}~ [\mathrm{Hz}]$')
# ax0.set_ylabel('PSD')
ax0.set_ylabel(r'$\mathrm{PSD\ [(m/s)^2/Hz]}$')
# fig4.tight_layout()# fig4.tight_layout()

#ax0ns4.grid(which='both', visible=True)
#fig4.canvas.draw()
# fig10.tight_layout()

# plt.close('all')
fig.savefig(dirout+'full_frequency_2.pdf',dpi=150, bbox_inches='tight')

