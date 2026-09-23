import glob
import numpy as np
from skimage.io import imread
from skimage.restoration import unwrap_phase

dird = '/home/juan/data/balseiro/vid_2025-02-24_16-04-16/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_16-00-23/'
dird  = '/home/juan/data/balseiro/vid_2025-02-24_16-04-16/'
dird = '/home/juan/data/balseiro/vid_2025-02-24_15-48-03/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_16-00-23/'
dird0 = '/home/juan/data/balseiro/vid_2025-02-24_15-56-03/'
fn_ref = dird0 + "frame_000001.tiff"

cmin, cmax = 150, 1200
rmin, rmax = 150, 800

# --- función FTP (idéntica a la tuya) ---
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

# --- referencia ---
ref_full = imread(fn_ref).astype(np.double)[rmin:rmax, cmin:cmax]
gri = np.mean(ref_full) * np.ones_like(ref_full)

# --- loop rápido ---
N = 1000   # ajustá según cuántos frames tengas
maxdphi = np.zeros(N)

for i in range(1, N+1):
    fn = dird + f"frame_{i:06d}.tiff"
    try:
        def_full = imread(fn).astype(np.double)[rmin:rmax, cmin:cmax]
    except FileNotFoundError:
        print("faltan frames a partir de", i)
        maxdphi = maxdphi[:i-1]
        break
    dphi = imagestodphasemap2(def_full - gri, ref_full - gri, 3, 100)
    dphi = unwrap_phase(dphi)
    dphi = dphi - dphi[0, 0]
    maxdphi[i-1] = np.max(np.abs(dphi))
    if i % 50 == 0:
        print(i, maxdphi[i-1])

imax = np.argmax(maxdphi)
print("Máximo global:", maxdphi[imax], "en frame", imax+1)


np.savetxt('maxdphi13.txt',maxdphi)

