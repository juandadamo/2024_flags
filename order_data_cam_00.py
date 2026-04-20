import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure
import glob
from matplotlib import rcParams
from funciones_flag import *
import tifffile as tif
from skimage import filters, measure


plt.close('all')
A = np.load('full_uniform/velocidad_41.npz')
YT = A['YT']

fig,ax = plt.subplots()
for yi in YT[:50]:
    ax.plot(yi,linestyle='none',marker='.',color='tab:blue')
