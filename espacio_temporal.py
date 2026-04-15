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
    dictsal = {'Imagen_sum':Imagen_sum,'A_curva_i':A_curva_i}
    xs = np.linspace(0,1280,1280)
    #for Ai in A_curva_i:
    #    plt.plot(xs,Ai,'.')

    savemat(caso.replace('.npz','.mat'),dictsal)
