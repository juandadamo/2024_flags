import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure
import glob
from matplotlib import rcParams
from funciones_flag import *
import tifffile as tif
from skimage import filters, measure


plt.close('all')
dir_w = '/home/juan/data/full/41_0_full/'
file_list = np.sort(glob.glob(dir_w+'/*.tif'))
# nfile = 1845

ntime_1 = len(file_list)
# ntime_1 = 10

for i,filei in enumerate(file_list[:ntime_1]):
    A = tif.imread(filei)
    if i ==0:
        m,n = A.shape
        Y_total = np.zeros((ntime_1, n))
    block_size = 35  # debe ser impar, mayor que el ancho de la bandera (4 píxeles)
    thresh_local = filters.threshold_local(A, block_size, method='gaussian')
    #thresh = filters.threshold_otsu(A)
    binaria = A > thresh_local

    centroide_columna = np.zeros(n)
    for j in range(n):
        filas_bandera = np.where(binaria[:, j])[0]
        if len(filas_bandera) > 0:
            centroide_columna[j] = np.mean(filas_bandera)
        else:
            centroide_columna[j] = np.nan  # sin bandera en esa columna

    Y_total[i] = centroide_columna
    if i % 100 == 0:
        print(f'Procesada imagen {i}/{ntime_1}')

file_out = 'full_uniform/velocidad_41.npz'
dictsal = {'YT':Y_total}
np.savez(file_out,**dictsal)
# Después de tener Y_total (500 × n)
# Puedes hacer SVD de estas curvas

# Centrar los datos
# Y_mean = np.mean(Y_total, axis=0)
# Y_centered = Y_total - Y_mean
#
# # SVD
# U, s, Vt = np.linalg.svd(Y_centered, full_matrices=False)
#
# # Ver energía acumulada
# energia_acum = np.cumsum(s**2) / np.sum(s**2)
# print(f'3 modos: {100*energia_acum[2]:.1f}% de energía')
# print(f'5 modos: {100*energia_acum[4]:.1f}%')
#
# # Plot primeros modos espaciales (Vt)
# fig, ax = plt.subplots(1, 3, figsize=(12, 3))
# for i in range(3):
#     ax[i].plot(Vt[i])
#     ax[i].set_title(f'Modo espacial {i+1}')
#     ax[i].set_xlabel('Columna')
#     ax[i].set_ylabel('Desplazamiento vertical')
# fig,ax = plt.subplots()
# for Yi in Y_total:
#     ax.plot(Yi)
#
#
#
# # Amedia = Asum/ntime_1
# # U1, s, V= np.linalg.svd(A_total-Amedia.reshape((1,m*n)), full_matrices=False)
# # k = 10
# # U1 = U1[:,:k]
# # V = V[:k]
# # s = s[:k]
# # fig,ax = plt.subplots(1,2)
# # ax0,ax1 = ax
# # ax0.imshow(A)
# # ax1.imshow(Asum)
# #
# #
# # fig2,ax2 = plt.subplots(2,2)
# ax2 = ax2.flatten()
# for i,ax2i in enumerate(ax2):
#     ax2i.imshow(V[i].reshape(m,n))
#
# dirout = 'full_uniform/'
# dictsal = {'U1':U1,'s':s,'V':V,'Amedia':Amedia}
# np.savez(dirout+'full_41.npz',**dictsal)
