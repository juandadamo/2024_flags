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
# set tex fonts in plots with legible sizes
plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)

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
# print(C.shape)
C = C.drop(range(5))
C = C.reset_index(drop=True)
# print(C.shape)
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

# raise ValueError()
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
D['$Ca_b$'] = rhoa_b*D['U_bistable']**2*D['L']**3 / (Papel_80.B) 
D['$Ca_f$'] = rhoa_b*D['U_flutter']**2*D['L']**3 / (Papel_80.B) 
D['$U_c$'] = (Papel_80.B/(rhoa_b*D['L']**3))**0.5

#theta0 = delta_cl
#f_kh = 0.032*Um/theta0
#theta0 = 0.032*Um/f_kh
D =  D.sort_values('L')

display(D)



fig,ax = plt.subplots()
for i,Li in enumerate(D['L']):
       l0, = ax.plot(Li,D.iloc[i,3],'o',linestyle='none',color='tab:blue' )
       colork  = l0.get_color()
       l1, = ax.plot(Li,D.iloc[i,4],'d',linestyle='none',color=colork )
       for li in [l0,l1]:
              li.set_markeredgecolor(colork)
              li.set_markersize(7)
              li.set_markeredgewidth(4)
              # li.set_color(to_rgba(colork,0.2))  
              # li.set_markerfacecolor(to_rgba(colork,0.2))
              li.set_fillstyle('none')
              if i == 4:
                     rectangle = plt.Rectangle((Li-0.3,D.iloc[i,3]-.4),0.6,D.iloc[i,4]-D.iloc[i,3]+1,linewidth=1,edgecolor='k',
                                               facecolor=to_rgba(colork,0.1))
                     ax.add_patch(rectangle)


       l2, = ax.plot([Li,Li],D.iloc[i,3:5],color=colork,
                     linewidth=8,linestyle='-')
       l2.set_alpha(0.3)


       
 


 


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

#tikz_save('figures/Intervalos_estabilidad_L_variable.tikz',figure=fig)



# Cauchy_flutter = rhoa_b*D['U_flutter']**2*(D['L']*.01)**3/(Papel_80.B)
# Cauchy_bistable = rhoa_b*D['U_bistable']**2*(D['L']*.01)**3/(Papel_80.B)

U_B = (Papel_80.B/(D['L']*0.01)**3/rhoa_b)**0.5
u_0 = D['U_bistable']/U_B

u_0_flutter = D['U_flutter']*0.5/U_B

mu_f_s = rhoa_b*D['L']*0.01/(Papel_80.rho*Papel_80.thickness)

omega_B = (rhoa_b*D['U_bistable']**2 / (Papel_80.rho*Papel_80.thickness*D['L']*0.01))**0.5
freq_B = (omega_B/(2*np.pi)).to_numpy()


data_argentina  = plt.imread('argentina_2004.png')


fig2,ax2 = plt.subplots()
# ax2.imshow(data_argentina,extent=[-1.6,10.9,-20,110],aspect='auto')
# ax2.set_axis_off()

l, = ax2.plot(mu_f_s,u_0,'o',fillstyle='none',markersize=7,markeredgewidth=2)
# raise ValueError()   
ax2.plot(mu_f_s,u_0_flutter,'d',fillstyle='none',
         markersize=7,markeredgewidth=2,color=l.get_color())
ax2.grid()
# U_bistable'],Cauchy_bistable,'-o',label='Bistable')
# ax2.plot(D['U_flutter'],Cauchy_flutter,'-o',label='Flutter')
# ax2.set_xlabel('U [m/s]')
# ax2.set_ylabel('Cauchy number')
# ax2.grid()
# ax2.legend()
# A.to_csv('Intervalos estabilidad.csv')
# B.to_csv('casos_3D_lista_archivos.csv')
# # C.to_csv('casos_2D_lista_archivos.csv')
# D.to_csv('Estabilidad_L_variable.csv')
# fig2.savefig('figures/Argentina_2004.png',dpi=300)


data_calculo = pd.read_csv('instability_values_flag.csv')


rho_vals = data_calculo['rho'].values
u0_crit = data_calculo['u0_crit'].values
omega_crit = data_calculo['omega_crit'].to_numpy()
omega_crit = omega_crit.astype(np.complex128)
freq_crit = np.real(omega_crit)/(2*np.pi)
ax2.plot(rho_vals, u0_crit, '*',fillstyle='none',
         markersize=7,markeredgewidth=2)


curva_00 = pd.read_csv('curve_argentina_00.csv',delimiter=';',decimal=',',header=None).to_numpy()
curva_01 = pd.read_csv('curve_argentina_01.csv',delimiter=';',decimal=',',header=None).to_numpy()
curva_02 = pd.read_csv('curve_argentina_02.csv',delimiter=';',decimal=',',header=None).to_numpy()
curva_03 = pd.read_csv('curve_argentina_03.csv',delimiter=';',decimal=',',header=None).to_numpy()
curva_exp = pd.read_csv('curve_argentina_exp.csv',delimiter=';',decimal=',',header=None).to_numpy()


estilos_linea = ['-','--','-.',':']
for i,curva in enumerate([curva_00,curva_01,curva_02,curva_03]):
       poly1 = np.polyfit(curva[:,0],1/curva[:,1],5)
       p1 = np.poly1d(poly1)
       x_fit = np.linspace(.001,10,100)
       y_fit = p1(x_fit)**-1
       if np.logical_or(i==0,i==3):
              ax2.plot(x_fit,y_fit,linestyle=estilos_linea[i],linewidth=3,color='k')
       else:
              ax2.plot(x_fit,y_fit,linestyle=estilos_linea[i],linewidth=1,color='k')
       # ax2.plot(curva[:,0],curva[:,1],marker='o',linestyle='none',color=lin.get_color(),fillstyle='none'
              #   ,markersize=7,markeredgewidth=2)



ax2.plot(curva_exp[:,0],curva_exp[:,1],'ks',markersize=3,markeredgewidth=2,linestyle='none',fillstyle='full')
# ax2.plot(curva_01[:,0],curva_01[:,1],marker='o',linestyle='none',color='tab:orange')
# ax2.plot(curva_02[:,0],curva_02[:,1],marker='o',linestyle='none',color='tab:green')
# ax2.plot(curva_03[:,0],curva_03[:,1],marker='o',linestyle='none',color='tab:blue')
# ax2.grid()
ax2.set_xlim([0,10])
ax2.set_ylim([0,100])
ax2.set_xlabel(r'$\sigma^*$')
ax2.set_ylabel(r'$u_0$')
#fig2.savefig('figures/Argentina_2004.png',dpi=300)
#tikz_save('figures/Argentina_2004.tikz',figure=fig2)

print(u0_crit[-2],freq_crit[-2]*freq_B[-2])





data_nuevo = [
    {'L': 60, 'b': 85.71428571, 'Uonset': 0, 'Ustop': 61.15185714},
    {'L': 65, 'b': 92.85714286, 'Uonset': 0, 'Ustop': 66.38542857},
    {'L': 70, 'b': 100, 'Uonset': 0, 'Ustop': 71.619},
    {'L': 75, 'b': 107.1428571, 'Uonset': 0, 'Ustop': 76.85257143},
    {'L': 80, 'b': 114.2857143, 'Uonset': 0, 'Ustop': 82.08614286},
    {'L': 85, 'b': 121.4285714, 'Uonset': 0, 'Ustop': 87.31971429},
    {'L': 90, 'b': 128.5714286, 'Uonset': 0, 'Ustop': 92.55328571},
    {'L': 95, 'b': 135.7142857, 'Uonset': 0, 'Ustop': 97.78685714},
    {'L': 100, 'b': 142.8571429, 'Uonset': 71.619, 'Ustop': 103.0204286},
    {'L': 110, 'b': 157.1428571, 'Uonset': 78.946, 'Ustop': 113.4875714},
    {'L': 120, 'b': 171.4285714, 'Uonset': 86.273, 'Ustop': 123.9547143},
    {'L': 130, 'b': 185.7142857, 'Uonset': 93.6, 'Ustop': 134.4218571},
    {'L': 140, 'b': 200, 'Uonset': 100.927, 'Ustop': 144.889},
    {'L': 150, 'b': 214.2857143, 'Uonset': 108.254, 'Ustop': 155.3561429},
    {'L': 160, 'b': 228.5714286, 'Uonset': 115.581, 'Ustop': 165.8232857}
]

df_nuevo = pd.DataFrame(data_nuevo)

#fig4,ax4 = plt.subplots()
#ax4.plot(df_nuevo['L'],df_nuevo['Uonset'],'o')


# Datos como lista de diccionarios

# Crear DataFrame de pandas
L_s1 = np.arange(60,105,5)
L_s2 = np.arange(110,170,10)
L_s = np.hstack((L_s1,L_s2))*1e-3
U_stop = np.array([15.71,13.95,12.56,12.56,11.90,11.31,10.95,10.07,9.92,9.48,8.16,7.21,6.77,6.77,6.40])


U_B = 1/L_s * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5

u_0 = U_stop/U_B
mu_f_s = rhoa_b*L_s/(Papel_80.rho*Papel_80.thickness)
fig5,ax5 = plt.subplots()
ax5.plot(L_s,U_stop,'o')
ax.plot(L_s*100,U_stop,'v')
ax2.plot(mu_f_s,u_0,'v',markersize=10,fillstyle='none')



data_ustop = pd.read_csv('datos_flutter_ustop.csv',decimal=',')
data_uonset = pd.read_csv('datos_flutter_uonset.csv',decimal=',')
fig5,ax5 = plt.subplots()

ax5.plot(data_ustop['L'],data_ustop['Ustop'],'o')
ax5.plot(data_uonset['L'],data_uonset['Uonset'],'s')


Ustop = data_ustop['Ustop']
Uonset = data_uonset['Uonset']
L = data_ustop['L']*1e-3
L2 = data_uonset['L']*1e-3
UB = 1/L * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
UB2 = 1/L2 * (Papel_80.B/Papel_80.rho/Papel_80.thickness)**.5
sigma = rhoa_b*L/(rho_papel*Papel_80.thickness)
sigma2 = rhoa_b*L2/(rho_papel*Papel_80.thickness)

fig6,ax6 = plt.subplots()
ax6.plot(sigma,Ustop/UB,'s')
ax2.plot(sigma,Ustop/UB,'s')


ax6.plot(sigma2,Uonset/UB2,'o')
ax2.plot(sigma2,Uonset/UB2,'o')


IM = plt.imread('raynaud_mulleners.png')
