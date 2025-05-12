import glob
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.restoration import unwrap_phase

plt.ion()
plt.close("all")

import numpy as np


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

fn_ref = "/Users/pablo/Sandbox/Bandera/19mar2025/vid_2025-02-24_16-00-23/frame_000001.tiff"
fn_def = "/Users/pablo/Sandbox/Bandera/19mar2025/vid_2025-02-24_16-04-16/frame_000010.tiff"

ref_im = imread(fn_ref).astype(np.double)


# Obtener la lista de archivos .tif en la carpeta actual

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

archivos = sorted(glob.glob("*.tif"))
for i in range(1,100):
    fn_def = f"/Users/pablo/Sandbox/Bandera/19mar2025/vid_2025-02-24_16-04-16/frame_{i:06d}.tiff"
    ref_im = imread(fn_ref).astype(np.double)
    def_im = imread(fn_def).astype(np.double)

    cmin = 150
    cmax = 1200
    rmin = 150
    rmax = 800

    ref_im = ref_im[rmin:rmax, cmin:cmax]
    def_im = def_im[rmin:rmax, cmin:cmax]
    gri_im = np.mean(ref_im) * np.ones_like(ref_im)

    dphi = imagestodphasemap2(def_im - gri_im, ref_im - gri_im, 3, 100)
    dphi = unwrap_phase(dphi)

    xvec = np.linspace(0, 1, dphi.shape[0])
    yvec = np.linspace(0, 1, dphi.shape[1])
    x, y = np.meshgrid(xvec, yvec)
    x = x.T
    y = y.T

    dphi_old = dphi[0,0]
    dphi = dphi-dphi_old

    varplot = ax.plot_surface(x, y, dphi, cmap=cm.coolwarm,
                    antialiased=False)

    ax.set_title(str(i))
    ax.view_init(elev=23, azim=-136)
    ax.set_zlim(-30, 30)

    plt.draw()
    plt.show()

    out_name = f"/Users/pablo/Sandbox/Bandera/19mar2025/vid_2025-02-24_16-04-16/dphi_{i:06d}.png"

    plt.savefig(out_name)
    plt.cla()
    del varplot
    print(i)

import imageio
with imageio.get_writer(f'demo_primeros_resultados_crudos.gif', mode='I', duration=0.1) as writer:
    for i in range(1,100):
        print(i)
        out_name = f"/Users/pablo/Sandbox/Bandera/19mar2025/vid_2025-02-24_16-04-16/dphi_{i:06d}.png"
        image = imageio.imread(out_name)
        writer.append_data(image)
