"""
figura_2f.py
Genera la Figura 2 del paper (FTP - deformación de la bandera).
Tres paneles: (a) imagen cruda con líneas de corte, (b) mapa h(x,z), (c) perfiles h(x).
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.restoration import unwrap_phase
from skimage.transform import ProjectiveTransform, warp
from skimage.transform import rotate
from skimage.filters import threshold_otsu


nfont = 16
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

# rotacion / correcion de imagen
alfa = 1.14
src = np.array([
    [  83,  120],
    [1253,76],
    [1277.62987013,  900],
    [  83,  900],
])
xmin,xmax,ymin,ymax = [src[:,0].min(),src[:,0].max(),src[:,1].min(),src[:,1].max()]
dst = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])


def corregir_trapecio(img, src, dst):
    rot_img = rotate(img,1.14)
    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    #print(warp(src,tform.inverse))
    return warp(rot_img, tform.inverse)


# --- rutas ---
dird  = '/home/juan/data/balseiro/vid_2025-02-24_16-04-16/'
dird = '/home/juan/data/balseiro/vid_2025-02-24_15-48-03/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_16-00-23/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'

fn_ref = dird0 + "frame_000001.tiff"
maxdphi_file = 'maxdphi13.txt'  # ajustar si está en otro lado

# --- ROI ---
rmin, rmax = 80, 900
cmin, cmax = 200, 1100

rmin,rmax = 100,900
cmin,cmax = 1,1275
# rmin, rmax  = 0, 750
# cmin, cmax = 130, 1100


# L = 130 mm <-> 700 px  (cols 100 a 800 en la ROI)
escala = 130.0 / (rmax-rmin)      # mm/px
H_mm = 200.0

# --- parámetros FTP ---
p  = 5.0        # mm, período de las franjas
Lp = 1000.0     # mm
D  = 400.0      # mm
factor = p * Lp / (2 * np.pi * D)   # mm/rad, ~1.9894

# --- elegir frame cercano a 27 rad ---
maxdphi = np.loadtxt(maxdphi_file)
target = 23.3
idx = np.argmin(np.abs(maxdphi - target))
frame_idx = idx + 1
print(f"Frame elegido: {frame_idx}  (maxdphi = {maxdphi[idx]:.2f} rad)")

# --- FTP ---
def imagestodphasemap2(dY, dY0, ns, cu):
    nx, ny = dY.shape
    fY0 = np.fft.fft2(dY0)
    fY  = np.fft.fft2(dY)
    maxF, imax = np.max(np.abs(fY0[0, 10:ny//2])), np.argmax(np.abs(fY0[0, 10:ny//2]))
    fx = np.arange(nx); fy = np.arange(ny)
    fmax = imax + 9
    f2x, f2y = np.meshgrid(fx, fy)
    s = fmax / ns
    gausfilt = np.exp(-((f2y - fmax)**2 + f2x**2)/s**2) + \
               np.exp(-((f2y - fmax)**2 + (f2x - f2x[-1])**2)/s**2)
    Ny0 = np.fft.ifft2(fY0 * gausfilt.T)
    Ny  = np.fft.ifft2(fY  * gausfilt.T)
    phase0 = np.unwrap(np.angle(Ny0))
    phase  = np.unwrap(np.angle(Ny))
    return phase - phase0

# --- cargar imágenes ---
ref_full = imread(fn_ref).astype(np.double)
ref_full_c = np.copy(ref_full)
ref_full = corregir_trapecio(ref_full, src, dst)

ref_roi  = ref_full[rmin:rmax, cmin:cmax]

fn_def   = dird + f"frame_{frame_idx:06d}.tiff"
def_full = imread(fn_def).astype(np.double)
def_full = corregir_trapecio(def_full, src, dst)
def_roi  = def_full[rmin:rmax, cmin:cmax]
gri_roi  = np.mean(ref_roi) * np.ones_like(ref_roi)

# --- fase y altura ---
dphi = imagestodphasemap2(def_roi - gri_roi, ref_roi - gri_roi, 3, 100)
dphi = unwrap_phase(dphi)
dphi = dphi - dphi[0, 0]
h_map = -factor * dphi   # mm


h_tot = np.zeros_like(def_full)

h_tot[rmin:rmax,cmin:cmax] = h_map


thres = threshold_otsu(def_full)
zerox,zeroy = np.nonzero(def_full<thres/6)
h_cur = np.copy(h_tot)
h_tot = np.ma.masked_where(def_full<0.01,h_tot)
# --- tres líneas de corte (filas de la ROI) ---
filas = {'H/3': 250, 'H/2': 450, '2H/3': 650}
filas = {'$z_3$': 430, '$z_2$': 680, '$z_1$': 930}
# --- figura ---
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2),gridspec_kw={'width_ratios': [1, 1, 1.4]})

# (a) imagen cruda deformada (ROI) con líneas
ax[0].imshow(def_full.T, cmap='gray')
for nombre, f in filas.items():
    ax[0].axhline(f+cmin, color='tab:orange', lw=2.0,linestyle='dashed')
ax[0].set_title('(a) FTP Image')
ax[0].set_xlabel('col [px]'); ax[0].set_ylabel('row [px]')

# (b) mapa h(x,z)

im = ax[1].imshow(h_tot.T, cmap='viridis',  extent=[0, (h_tot.shape[0]-0)*escala, 0, h_tot.shape[1]*escala])

x_im = np.arange(h_tot.shape[0])*escala
y_im = np.arange(h_tot.shape[1])*escala
X, Y = np.meshgrid(x_im,y_im)
px0 = 89
im = ax[1].imshow(h_tot.T,extent=[-px0*escala, (h_tot.shape[0]-px0)*escala, 0, h_tot.shape[1]*escala],vmax=40)


    #X,Y,h_tot.T,levels=np.linspace(0,40,20))
ax[1].plot(zerox*escala-px0*escala,1279*escala-zeroy*escala,'ks')
#ax[0].plot(zerox,zeroy,'ws')
#im = ax[1].plot(h_tot.T, cmap='Greys',  extent=[0, h_map.shape[0]*escala, 0, h_map.shape[1]*escala])
for nombre, f in filas.items():
    ax[1].axhline(1279*escala-f*escala,  color='tab:orange', lw=2.0,linestyle='dashed')

ax[1].set_title('(b) Deformation Map $h(x,z)$ [mm]')
ax[1].set_xlabel('$x$ [mm]'); ax[1].set_ylabel('$z$ [mm]')
ax[1].set_yticks(1279*escala-np.array(list(filas.values()))*escala,list(filas.keys()))
plt.colorbar(im, ax=ax[1], label='$h$ [mm]')

#perfiles y  mascara

x_mm = np.arange(h_tot.shape[0]) * escala - 89*escala
for nombre, f in filas.items():
    perfil = h_cur[:, 1280-f]              # ya es masked array
    indice_fin = np.nonzero(np.diff(perfil[-300:])<0)[0][0] + len(perfil)-300-5

    ax[2].plot(x_mm[:indice_fin], perfil[:indice_fin], label=f'$z$ = {nombre}', marker='', ls='-')
ax[2].set_title('(c) $h(x)$ profiles')
ax[2].set_xlabel('$x$ [mm]'); ax[2].set_ylabel('$y$ [mm]')
ax[2].set_ylim([-40,40])
ax[2].legend(fontsize=14); ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figura_2f.png', dpi=200)
plt.show()

# --- guardar datos ---
np.savez('figura_2f.npz',
         frame_idx=frame_idx,
         maxdphi_val=maxdphi[idx],
         h_map=h_map,
         x_mm=x_mm,
         escala=escala,
         factor_mm_rad=factor,
         p=p, Lp=Lp, D=D)
#print("Guardado: figura_2f.png y figura_2f.npz")
#print(f"Factor de conversión: {factor:.4f} mm/rad")
