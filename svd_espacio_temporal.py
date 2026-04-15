import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import glob

caso = 'full_freq_13.4'

lista_casos = np.sort(glob.glob('data_out/'+'*.npz'))

for caso in lista_casos[2:3]:
    A = np.load(caso)

    Imagen_sum = A['Imagen_sum']
    A_curva_i = A['A_curva_i']
    U,S,Vh = np.linalg.svd((A_curva_i-A_curva_i.mean(0))[:,46:1146])

for Vhi in Vh[:6]:
    plt.plot(Vhi)
