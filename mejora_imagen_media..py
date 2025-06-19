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
for j, filej in enumerate(lista_caso_2d[:7]):
    A1 = np.load(filej)
    Asum = A1['Imagen_sum']



    # Cargar imagen en escala de grises
    image = Asum

    # Detección de bordes con Canny
    edges = filters.sobel(image)  # o filters.canny(image, sigma=1)