import glob
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.restoration import unwrap_phase
from pathlib import Path

from skimage.transform import ProjectiveTransform, warp
from skimage.transform import rotate
from skimage.filters import threshold_otsu



plt.ion()
plt.close("all")


import numpy as np


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
    # tform.from_estimate(src, dst)
    #print(warp(src,tform.inverse))
    return warp(rot_img, tform.inverse)


def imagestodphasemap2(dY, dY0, ns, cu):
    # Basic FTP treatment.
    # This function takes a deformed and a reference image and calculates the phase difference map between the two.

    # INPUTS:
    # dY    = deformed image
    # dY0   = reference image
    # ns    = size of gaussian filter
    # cu    = reference column for unwrapping

    # OUTPUT:
    # dphase    = phase difference map between images

    # Basic definitions
    nx, ny = dY.shape
    fY0 = np.fft.fft2(dY0)
    fY = np.fft.fft2(dY)

    # Search for the maximum frequency in the spectrum
    maxF, imax = np.max(np.abs(fY0[0, 10 : ny // 2])), np.argmax(
        np.abs(fY0[0, 10 : ny // 2])
    )

    if imax == 0:
        # Uncomment the following line if you want to stop the execution if imax is 0
        # stop()
        pass

    # Sampling at nx, ny points
    fx = np.arange(nx)
    fy = np.arange(ny)

    fmax = imax + 9
    # Axis of Fourier transform
    f2x, f2y = np.meshgrid(fx, fy)

    # Definition of the gaussian filter
    s = fmax / ns
    gausfilt = np.exp(-((f2y - fmax) ** 2 + f2x**2) / s**2) + np.exp(
        -((f2y - fmax) ** 2 + (f2x - f2x[-1]) ** 2) / s**2
    )

    # Multiplication by the filter
    Nfy0 = fY0 * gausfilt.T
    Nfy = fY * gausfilt.T


    # Inverse Fourier transform of both images
    Ny0 = np.fft.ifft2(Nfy0)
    Ny = np.fft.ifft2(Nfy)

    # Very basic phase unwrapping follows!
    # First step
    phase0 = np.unwrap(np.angle(Ny0))
    phase = np.unwrap(np.angle(Ny))
    # Second step
    # p0 = phase0[:, cu]
    # up0 = np.unwrap(p0)
    # p = phase[:, cu]
    # up = np.unwrap(p)
    # phase0 = phase0 + (up0 - p0)[:, np.newaxis]
    # phase = phase + (up - p)[:, np.newaxis]

    # Definition of the phase difference map
    dphase = phase - phase0

    return dphase


# loading of images:
# (a) a reference image
# (one with fringes at fixed k)
#
# (b) a deformed image
# (one with deformed fringes)
#
# (c) a gray image
# (an image of the center intensity level projected onto the scene)
dird = '/home/juan/data/balseiro/vid_2025-02-24_16-04-16/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_16-00-23/'
dird0b = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'
fn_ref = dird0+"frame_000001.tiff"
#fn_def = dird+"frame_000010.tiff"



archivos = sorted(Path(dird0b).glob("*.tiff"))


imagen_m = 0
n_imagen = 10 -1

# --- ROI ---
rmin, rmax = 80, 900
cmin, cmax = 200, 1100

rmin,rmax = 100,900
cmin,cmax = 1,1275



# L = 130 mm <-> 700 px  (cols 100 a 800 en la ROI)
escala = 130.0 / (rmax-rmin)      # mm/px
H_mm = 200.0

# --- parámetros FTP ---
p  = 5.0        # mm, período de las franjas
Lp = 1000.0     # mm
D  = 400.0      # mm
factor = p * Lp / (2 * np.pi * D)   # mm/rad, ~1.9894









# --- fase y altura ---




ref_full = imread(fn_ref).astype(np.double)
ref_roi  = ref_full[rmin:rmax, cmin:cmax]
gri_roi  = np.mean(ref_roi) * np.ones_like(ref_roi)
h_map_serie = 0

for i in range(1,n_imagen):
    fn_i = dird0b+ f"frame_{i:06d}.tiff"

    def_full = imread(fn_i).astype(np.double)
    def_full = corregir_trapecio(def_full, src, dst)
    def_roi  = def_full[rmin:rmax, cmin:cmax]
    dphi = imagestodphasemap2(def_roi - gri_roi, ref_roi - gri_roi, 3, 100)
    dphi = unwrap_phase(dphi)
    dphi = dphi - dphi[0, 0]
    h_map = -factor * dphi   # mm


    imagen_m += def_full
    h_map_serie += h_map

imagen_m = imagen_m / n_imagen
h_map_serie = h_map_serie / n_imagen

imagen_std = 0
h_map_serie_std = 0
for i in range(1,n_imagen):
    fn_i = dird0b+ f"frame_{i:06d}.tiff"
    def_full = imread(fn_i).astype(np.double)
    def_full = corregir_trapecio(def_full, src, dst)
    def_roi  = def_full[rmin:rmax, cmin:cmax]
    dphi = imagestodphasemap2(def_roi - gri_roi, ref_roi - gri_roi, 3, 100)
    dphi = unwrap_phase(dphi)
    dphi = dphi - dphi[0, 0]
    h_map = -factor * dphi   # mm


    h_map_serie_std += (h_map - h_map_serie)**2
    imagen_std += (imread(fn_i).astype(np.double) - imagen_m)**2
imagen_std = (imagen_std/n_imagen)**.5
h_map_serie_std = (h_map_serie_std/n_imagen)**.5


plt.imshow(h_map_serie_std)
