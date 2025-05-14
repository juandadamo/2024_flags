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
plt.close('all')
dictflag = {'Caso':['Full','Segmented rectangular','Segmented Triangular'],'f_bistable':np.zeros((3)),'f_flutter':np.zeros((3)),'U_bistable':np.zeros((3)),'U_flutter':np.zeros((3))}
A = pd.DataFrame.from_dict(dictflag)

A.iloc[0,1] = 11.9
A.iloc[0,2]= 16.3
A.iloc[1,1] = 16.6
A.iloc[1,2] = 25.9
A.iloc[2,1] = 11.4
A.iloc[2,2] = 20
print('Resumen intevalos estabilidad')
display(A)


#identificacion de ensayos
dirlist = np.sort(glob.glob('/home/juan/data/balseiro/*'))
dirlist = np.array([list_i.split('/')[-1]  for list_i in dirlist])
dictfiles = {'measurement':dirlist,'folder name':dirlist}
styles = [dict(selector="caption",
                       props=[("text-align", "center"),
                              ("font-size", "150%"),
                              ("color", 'green') ])]

B = pd.DataFrame.from_dict(dictfiles)

# raise ValueError()
B = B.drop(range(15))
B = B.reset_index(drop=True)
C = B.copy()
C = C.drop(range(15))
C = C.reset_index(drop=True)
B.iloc[0,0] = 'Segmented rect - 0 - Reference 1'
B.iloc[1,0] = 'Segmented rect - 17.2 - Mode 0'
B.iloc[2,0] = 'Segmented rect - 17.2 - Mode 0 - b'
B.iloc[3,0] = 'Segmented rect - 17.2 - Mode 1'
B.iloc[4,0] = 'Segmented rect - 16.9 - Mode 1'
B.iloc[5,0] = 'Full  - 13.4 - Mode 1'
B.iloc[6,0] = 'Full  - 13.4 - Mode 0'
B.iloc[7,0] = 'Full  - 0 - Reference 1'
B.iloc[8,0] = 'Full  - 17.2 - Mode 1'
B.iloc[9,0] = 'Full  - 0 - Reference 2'
B.iloc[10,0] = 'Segmented triang - 13.4 - Mode 0'
B.iloc[11,0] = 'Segmented triang - 13.4 - Mode 1'
B.iloc[12,0] = 'Segmented triang - 17.2 - Mode 1'
B.iloc[13,0] = 'Segmented triang - 0 - Reference 1'
B = B.drop(range(14,42))
#display(B.style.set_caption("Resumen medidas 3D").set_table_styles(styles))

print('Resumen medidas 3D')
display(B)
C.iloc[0:10,0] = 'full'
C.iloc[10:15,0] = 'rect'
C.iloc[15:17,0] = 'full'
C.iloc[17:,0] = 'triang'
freqs1 = np.arange(20.,12,-1)
freqs1 = np.hstack((freqs1,[12.8,12.2]))
freqs2 = np.arange(20.,15,-1)
freqs3 = np.arange(20,11,-1)
freqs3 = np.hstack((freqs3,[11.5]))
freqs = np.hstack((freqs1,freqs2,[17.2,13.4],freqs3))
C['freq motor'] = freqs
print('Resumen medidas 2D')
display(C)


dictflag_L = {'Caso':['L0','L1','L2','L3','L4','L5'],'f_bistable':np.zeros((6)),'f_flutter':np.zeros((6)),'U_bistable':np.zeros((6)),'U_flutter':np.zeros((6))}
D = pd.DataFrame.from_dict(dictflag_L)

D.iloc[0,1] = 11.9
D.iloc[0,2]= 16.3
D.iloc[1,1] = 10.2
D.iloc[1,2] = 16.8
D.iloc[2,1] = 14.4
D.iloc[2,2] = 25.5
D.iloc[3,1] = 14.9
D.iloc[3,2] = 21.6
D.iloc[4,1] = 21
D.iloc[4,2] = 22
D.iloc[5,1] = 14.2
D.iloc[5,2] = 22.5
D['L'] = np.array([13,15.5,10.5,8,7,9])
print('Resumen intevalos estabilidad L variable')

D['U_bistable'] = veloc_tunel_ib(D['f_bistable'])
D['U_flutter'] = veloc_tunel_ib(D['f_flutter'])

A['U_bistable'] = veloc_tunel_ib(A['f_bistable'])
A['U_flutter'] = veloc_tunel_ib(A['f_flutter'])


# print(f'${{f_n}}_1= {Papel_80.fn[0]:.3f}$Hz')
# print(f'${{f_n}}_2= {Papel_80.fn[1]:.3f}$Hz')

rhoa = 1.2
rhoa_b = 1.0888  #densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa
Uinf = 12
delta_cl = 18e-3 # espesor de capa limite para velocidad 12m/s

#longitud caracteristica de la placa plana (tunel) en base a la medicion en Balseiro
x_carac = longitud_equivalente_capa_limite_turbulenta(delta_cl,Uinf,nu)

delta1 = delta_turb(x_carac,D['U_bistable'],nu)
delta2 = delta_turb(x_carac,D['U_flutter'],nu)
D['$f_{kh}1$'] =  0.032*D['U_bistable']/2/delta1
D['$f_{kh}2$'] =  0.032*D['U_flutter']/2/delta2

f_n1 = np.zeros_like(D['U_bistable'])
f_n2 = np.zeros_like(D['U_bistable'])
for i,Li in enumerate(D['L']):
       Papel_80.L = Li*1e-2
       Papel_80.freq_nat()
       f_n1[i] = Papel_80.fn[0]
       f_n2[i] = Papel_80.fn[1]
D['$f_n1$'] = f_n1
D['$f_n2$'] = f_n2
#theta0 = delta_cl
#Um = Uinf / 2
#f_kh = 0.032*Um/theta0
#theta0 = 0.032*Um/f_kh
D =  D.sort_values('L')

display(D)



fig,ax = plt.subplots()
for i,Li in enumerate(D['L']):
       l, = ax.plot([Li,Li],D.iloc[i,3:5],'-o')
       colork  = l.get_color()
       l.set_linewidth(10)
       l.set_markeredgewidth(2)
       l.set_markeredgecolor(colork)
       l.set_fillstyle('none')
       l.set_markersize(10)

       l.set_color(to_rgba(colork,0.2))

ax.set_xlabel('L [mm]')
ax.set_ylabel('U [m/s]')
ax.grid()

fig1,ax1 = plt.subplots()
for i,Li in enumerate(D['$f_n2$']):
       l, = ax1.plot([Li,Li],D.iloc[i,6:8],'-o')
       colork  = l.get_color()
       l.set_linewidth(10)
       l.set_markeredgewidth(2)
       l.set_markeredgecolor(colork)
       l.set_fillstyle('none')
       l.set_markersize(10)

       l.set_color(to_rgba(colork,0.2))

ax1.set_xlabel(r'$f_n$ [hz]')
ax1.set_ylabel(r'$f_{kh}$ [hz]')
ax1.grid()

A.to_csv('Intervalos estabilidad.csv')
B.to_csv('casos_3D_lista_archivos.csv')
C.to_csv('casos_2D_lista_archivos.csv')
D.to_csv('Estabilidad_L_variable.csv')
