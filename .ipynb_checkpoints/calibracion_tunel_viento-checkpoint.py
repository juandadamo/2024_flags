import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close('all')
csv_data= """Voltage,Vitesse_testo,Hauteur,Hauteur_2,Hauteur_3
60,4.92,0.6810,,
57.5,4.68,0.6903,,
55,4.44,0.6923,,
52.5,4.20,0.6965,0.7087,0.6975
50,3.99,0.7012,0.7006,
47.5,3.77,0.7106,,
45,3.56,0.7115,,
42.5,3.34,0.7206,,
40,3.12,0.7215,,
35,2.72,0.7314,,
30,2.30,0.7427,,
20,1.47,0.7567,,
10,0.58,0.7628,,
0,0, 0.7715,0.7688, 0.7772
"""
A = pd.read_csv('releves_soufflerie_may.csv',header=0,delimiter=';',skipfooter=2,engine='python')
# print(A)
Volt, U_testo, Y = A.iloc[:-1,:3].to_numpy().T
Yzeros = A.iloc[-1,2:].to_numpy()
Yzeros = Yzeros - 0.016
g = 9.8
peso_alcohol_propilico = 24#grf
vol_alcohol_propilico = 30#ml
masa_alcohol_propilico = peso_alcohol_propilico *1e-3
densidad_alcohol_propilico = masa_alcohol_propilico/  (vol_alcohol_propilico*1e-6)
rho_al = densidad_alcohol_propilico
rho_aire = 1.2

dY = np.tile(Y,[3,1])
for i,dYi in enumerate(dY):
    dY[i] = Yzeros[i] - Y
deltaP = dY*25.4*1e-3*g*rho_al
U_manom = np.sqrt(2*deltaP/rho_aire)

fig,ax = plt.subplots()

ax.plot(Volt,U_manom[0],'o',linestyle='none',fillstyle='none')
ax.plot(Volt,U_manom[1],'s',linestyle='none',fillstyle='none')
ax.plot(Volt,U_manom[2],'v',linestyle='none',fillstyle='none')
ax.plot(Volt,U_testo,'d',linestyle='none',fillstyle='none')
ax.grid()
ax.set_xlim([0,65])
ax.set_ylim([0,6])
