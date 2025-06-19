from skimage import io, filters, measure, morphology
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
#%matplotlib widget
import serial,socket,os,glob,sys
#import atexit
import numpy as np
import pandas as pd
import time, threading,sys,glob
colores = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
import tifffile as tif
import skimage as sk
from IPython.display import Latex
from funciones_flag import *
from scipy.signal import find_peaks
mks = ['s','o','>','p','v','^','*']
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage import feature

escalax = 1/0.138 # px/mm


if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'


caso = 'rect'
caso = 'triang'
# caso = 'full'

if caso == 'full':
    npoints = 6
    frec_c = 12.2
elif caso == 'triang':
    npoints = 5
    frec_c = 13


lista_caso_2d = np.sort(glob.glob('data_out/'+caso+'_freq*'))

Velocidad, Amplitud = np.zeros((2,len(lista_caso_2d)))
for j, filej in enumerate(lista_caso_2d[:1]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']



    # Cargar imagen en escala de grises
    image = Asum**.12
    YT = A1['A_curva_i']
    frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
    Velocidad[j] =  veloc_tunel_ib(frec_j)
    # Detección de bordes con Canny
    edges = feature.canny(image, sigma=4)
    closed_edges = morphology.closing(edges, morphology.disk(radius=5))
    image[closed_edges] = 1
    image[np.logical_not(closed_edges)] = 0
    delta_coord = np.nonzero(image==1e5)[0].max()- np.nonzero(image==1e5)[0].min()
    # raise ValueError()
    Amplitud[j]  = delta_coord*1.0/ escalax  # mm
    if j==4:
        fig0,ax0 = plt.subplots()
        ax0.imshow(Asum)
        for YT_k in YT[100:150:10]:
            ax0.plot(YT_k,marker='o',color='tab:orange',markersize=0.5,linestyle='none')
        fig0.savefig(dirw+'image_sum_'+caso+'.png')
        #fig1,ax1 = plt.subplots()
        #ax1.imshow(label_image)
        ax0.plot(coord_amp[:,1],coord_amp[:,0],marker='o',fillstyle='none',linestyle='none',markersize=0.1,color='tab:orange')
        fig0.savefig(dirw+'image_label_'+caso+'.png')
        ax0.plot([1,len(YT.T)],[aux[:,0].max(),aux[:,0].max()],color='w',linewidth=3)
        ax0.plot([1,len(YT.T)],[aux[:,1].min(),aux[:,1].min()],color='w',linewidth=3)
        fig0.savefig(dirw+'image_label_amp_'+caso+'.png')
fig,ax = plt.subplots()
#Uc = Velocidad[0]


Uc = veloc_tunel_ib(frec_c)
U = Velocidad - Uc
ax.plot(Velocidad,Amplitud,'ks',fillstyle='none',linestyle='none')
ax.set_xlabel('$U$[m/s]')
ax.set_ylabel('$A$')
ax.grid()
fig.savefig(dirw+'Amplitudes_'+caso+'.png')
 