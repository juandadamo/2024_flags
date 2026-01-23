import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat


caso = 'full_freq_13.4'
A = np.load('data_out/'+caso+'.npz')

Imagen_sum = A['Imagen_sum']
A_curva_i = A['A_curva_i']
dictsal = {'Imagen_sum':Imagen_sum,'A_curva_i':A_curva_i}
xs = np.linspace(0,1280,1280)
#for Ai in A_curva_i:
#    plt.plot(xs,Ai,'.')

savemat('data_out/'+caso+'.mat',dictsal)
