import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

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

ref_im = imread("init0.jpg").astype(np.double)
def_im = imread("objet0.jpg").astype(np.double)

# creating a gray image synthetically (usually one captures one such image)
gri_im = np.mean(ref_im) * np.ones_like(ref_im)

# we calculate the phase map
dphi = imagestodphasemap2(def_im - gri_im, ref_im - gri_im, 3, 100)

fig, ax = plt.subplots(2, 4)
ax[0, 0].imshow(gri_im)
ax[0, 1].imshow(ref_im)
ax[0, 2].imshow(def_im)
ax[0, 3].imshow(dphi)

ax[1, 0].plot(gri_im[100, :])
ax[1, 0].grid(True)
ax[1, 1].plot(ref_im[100, :])
ax[1, 1].grid(True)
ax[1, 2].plot(def_im[100, :])
ax[1, 2].grid(True)
ax[1, 3].plot(dphi[100, :], ".-")
ax[1, 3].grid(True)

x = np.linspace(0, 1024, 1024)

# ax[1, 3].plot(x, -0.01 * x)
