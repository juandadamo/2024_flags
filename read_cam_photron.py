import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure
import glob
from matplotlib import rcParams
from funciones_flag import *
import tifffile as tif


plt.close('all')
dir_w = '/home/juan/data/full/41_0_full/'
file_list = np.sort(glob.glob(dir_w+'/*.tif'))
nfile = 1845

ntime_1 = 500


for i,filei in enumerate(file_list[:ntime_1]):
    A = tif.imread(filei)
    A = np.uint8(A[320:760,210:1164])
    if i==0:
        Asum = A
        m,n = A.shape
        A_total = np.zeros((ntime_1,m*n))

    else:
        Asum += A
    A_total[i] = A.reshape(1,m*n)

U1, s, V= np.linalg.svd(A_total, full_matrices=False)

fig,ax = plt.subplots(1,2)
ax0,ax1 = ax
ax0.imshow(A)
ax1.imshow(Asum)


fig2,ax2 = plt.subplots(2,2)
ax2 = ax2.flatten()
for i,ax2i in enumerate(ax2):
    ax2i.imshow(V[i].reshape(m,n))

