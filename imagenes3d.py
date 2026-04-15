#!/usr/bin/env python
# coding: utf-8
from IPython.display import display, HTML


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import glob
from funciones_flag import *
from matplotlib.colors import to_rgba
from tikzplotlib import save as tikz_save
import tifffile as tiff
# set tex fonts in plots with legible sizes
plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
mks = ['s','o','v']
plt.close('all')
rhoa = 1.2
rhoa_b = 1.0888  #densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa

plt.close('all')

dirdata = '/home/juan/data/balseiro/vid_2025-02-24_16-04-16/'
lista_im = np.sort(glob.glob(dirdata+'*.tiff'))
fig,ax = plt.subplots(2,4,figsize=(7,5 ),sharex=True,sharey=True ,
                       gridspec_kw={'wspace':0, 'hspace':0})
axs = ax.flatten()
for i,im_i in enumerate(lista_im[:32:4]):
    A = tiff.imread(im_i)
    axs[i].imshow(A.T)
    axs[i].axis('off')
    axs[i].set_xlim(0,900)
# fig.tight_layout()

#fig.savefig('/home/juan/Documents/Publicaciones/2025_euromech/flag_shear/article/figures/profiles_3d.pdf')
