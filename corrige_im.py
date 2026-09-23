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

plt.close('all')

#correcion de imagen

src = np.array([
    [  68.04545455,  110.62987013],
    [1279.13636364,   86.12987013],
    [1277.62987013,  915.99350649],
    [  73,  890],
])


src = np.array([
    [  83,  120],
    [1253,76],
    [1277.62987013,  900],
    [  83,  900],
])




xmin,xmax,ymin,ymax = [src[:,0].min(),src[:,0].max(),src[:,1].min(),src[:,1].max()]
ancho = src[:,0].max() - src[:,0].min()
alto  = src[:,1].max() - src[:,1].min()

dst = np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])
#dst = np.array([[0,0],[ancho,0],[ancho,alto],[0,alto]])

def corregir_trapecio(img, src, dst, ancho, alto):
    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    return warp(img, tform.inverse, output_shape=(int(alto), int(ancho)),
                preserve_range=True)


# --- rutas ---
dird  = '/home/juan/data/balseiro/vid_2025-02-24_16-04-16/'
dird = '/home/juan/data/balseiro/vid_2025-02-24_15-48-03/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_16-00-23/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'

fn_ref = dird0 + "frame_000001.tiff"
maxdphi_file = 'maxdphi13.txt'  # ajustar si está en otro lado

# --- ROI ---
rmin, rmax = 150, 800
cmin, cmax = 150, 1200

rmin, rmax  = 0, 750
cmin, cmax = 130, 1100


# L = 130 mm <-> 700 px  (cols 100 a 800 en la ROI)
escala = 130.0 / 700.0      # mm/px
H_mm = 200.0

# --- parámetros FTP ---
p  = 5.0        # mm, período de las franjas
Lp = 1000.0     # mm
D  = 400.0      # mm
factor = p * Lp / (2 * np.pi * D)   # mm/rad, ~1.9894

# --- elegir frame cercano a 27 rad ---
maxdphi = np.loadtxt(maxdphi_file)
target = 22.0
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
rot_full = rotate(ref_full,1.14)


tform = ProjectiveTransform()
tform.estimate(src, dst)

corr_ref = warp(rot_full,tform.inverse,preserve_range=True)


fig,ax = plt.subplots(1,2)
ax0,ax1 = ax
ax0.imshow(ref_full_c.T)
ax1.imshow(rot_full.T)

fig2,ax2 = plt.subplots(1,2)
ax2a,ax2b = ax2
ax2a.imshow(rot_full.T)
ax2b.imshow(corr_ref.T)
# ref_full = corregir_trapecio(ref_full, src, dst, ancho, alto)
#
#
# ref_roi  = ref_full[rmin:rmax, cmin:cmax]
# fn_def   = dird + f"frame_{frame_idx:06d}.tiff"
# def_full = imread(fn_def).astype(np.double)
# def_full = corregir_trapecio(def_full, src, dst, ancho, alto)
# def_roi  = def_full[rmin:rmax, cmin:cmax]
# gri_roi  = np.mean(ref_roi) * np.ones_like(ref_roi)
