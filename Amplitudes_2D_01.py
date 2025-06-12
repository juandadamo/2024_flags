import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
#%matplotlib widget
import serial,socket,os,glob,sys
#import atexit
import numpy as np
import pandas as pd
import time,sys,glob
 
colores = (plt.rcParams['axes.prop_cycle'].by_key()['color'])
import tifffile as tif
import skimage as sk
from IPython.display import Latex
from funciones_flag import *
from scipy.signal import find_peaks
mks = ['s','o','>','p','v','^','*']
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.util import invert
from skimage import data
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis
plt.close('All')

if socket.gethostname() == 'CNRS304952':
    dir_root = r'D:/'
else:
    dir_root = r'/home/juan/data/balseiro/'
dir_data_2D = pd.read_csv(r'casos_2D_lista_archivos.csv')

index_full = dir_data_2D['medida']=='full'

casos_full = dir_data_2D[index_full]['nombre carpeta'].to_numpy()[:-2][::-1]

fmotor_full = dir_data_2D[index_full]['freq motor'].to_numpy()[:-2][::-1]

for j,caso_full_j in enumerate(casos_full):
    files_list = np.sort(glob.glob(dir_root+caso_full_j+'/*.tiff'))
    # raise ValueError()
    nsnapshots = np.min((1000,len(files_list)))

    Amp_i = np.zeros((nsnapshots))
    print(caso_full_j)
    print(fmotor_full[j])
    for i,filei in enumerate(files_list[:nsnapshots][:]):
        A = tif.imread(filei)
        if i==0:
            #A_total = np.tile(np.zeros_like(A),[nsnapshots,1,1])
            A_curva_i = np.zeros((nsnapshots,len(A.T)))
            Asum = np.zeros_like(A*0.1)
        Asum += A
        A_max_j = np.zeros(len(A.T))
        Int_j = np.zeros(len(A.T))
        umbral_intensidad = sk.filters.threshold_otsu(A)/5
        B = A>= umbral_intensidad

        kernel = disk(3)  # o square(3) para un kernel cuadrado
        B = B.astype(np.uint8)
        #B = invert(B)
        B_clean = binary_closing(B, square(3))  # Elimina píxeles aislados
        B_clean = binary_opening(B_clean, square(3))  # Suaviza bordes

        C = skeletonize(B_clean)

        ny  = C.astype(np.uint8).argmax(axis=0)

        Amp_i[i] = ny.max()
        A_curva_i[i]  = ny
        #if i ==100:raise ValueError()

    nx =  range(C.shape[1])
    dictsal = {'Imagen_sum':Asum,'A_curva_i':A_curva_i}
    nombre_out = f'data_out/full_freq_{fmotor_full[j]}.npz'
    #raise ValueError()
    print(nombre_out)
    np.savez(nombre_out,**dictsal)

#raise ValueError()
fig,ax = plt.subplots()
for i in range(1,1000,10):
    ax.plot(nx,A_curva_i[i],'o',linestyle='none',markersize=2,fillstyle='none')
