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
from skimage.filters import threshold_otsu, rank
from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte

if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
    dir_root = r'D:/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'
    dir_root = r'/home/juan/data/balseiro/'
 

lista_calibra_1 = np.sort(glob.glob(dir_root+'CalibraciónLaser/vid_2025-02-25_15-49-17/*.tiff'))

for i,filei in enumerate(lista_calibra_1[:]):
    A = tif.imread(filei)
    if i==0:
        A_total = np.tile(np.zeros_like(A),[len(lista_calibra_1),1,1])
        A_curva_i = np.zeros((len(lista_calibra_1),len(A.T)))
    A_total[i] = A

A_mean = A_total.mean(0)
A_norm = (A_mean - A_mean.min()) / (A_mean.max() - A_mean.min())  # Normaliza entre 0 y 1

img = img_as_ubyte(A_norm)

radius = 5
selem = disk(radius)

local_otsu = rank.otsu(img, selem)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu

   