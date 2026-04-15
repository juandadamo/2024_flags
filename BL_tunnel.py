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
from scipy.optimize import curve_fit


def blasius_profile(y, U_inf, x, nu=1.5e-5):
    """
    Perfil de capa límite laminar de Blasius

    Parameters:
    y : array - distancia desde la pared (m)
    U_inf : float - velocidad de corriente libre (m/s)
    x : float - distancia desde el borde de ataque (m)
    nu : float - viscosidad cinemática (m²/s)

    Returns:
    U : array - perfil de velocidad (m/s)
    """
    # Coordenada de similaridad η = y * sqrt(U_inf/(ν*x))
    eta = y * np.sqrt(U_inf/(nu*x))

    # Solución numérica de Blasius (valores tabulados)
    # η vs f'(η) = U/U_inf
    eta_table = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8,
                          2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8,
                          4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0])

    f_prime_table = np.array([0.0, 0.0664, 0.1328, 0.1989, 0.2647, 0.3298, 0.3938, 0.4563,
                              0.5168, 0.5748, 0.6298, 0.6813, 0.7290, 0.7725, 0.8115,
                              0.8460, 0.8761, 0.9018, 0.9233, 0.9411, 0.9555, 0.9670,
                              0.9759, 0.9827, 0.9878, 0.9915, 0.9942, 0.9962, 0.9975,
                              0.9984, 0.9990])

    # Interpolar para los valores de η de nuestros datos
    U_Uinf = np.interp(eta, eta_table, f_prime_table, left=0, right=1)

    return U_inf * U_Uinf
def boundary_layer_thicknesses(y, U, U_inf):
    """
    Calcula espesores de capa límite

    Parameters:
    y : array - distancia desde la pared (m)
    U : array - perfil de velocidad (m/s)
    U_inf : float - velocidad de corriente libre

    Returns:
    delta_star : espesor de desplazamiento (m)
    theta : espesor de momento (m)
    """
    dy = np.gradient(y)

    # Espesor de desplazamiento: delta* = integral(1 - U/U_inf) dy
    delta_star = np.trapz(1 - U/U_inf, y)

    # Espesor de momento: theta = integral(U/U_inf * (1 - U/U_inf)) dy
    theta = np.trapz((U/U_inf) * (1 - U/U_inf), y)

    return delta_star, theta

def turbulent_bl(y, U_inf, delta):
    """Perfil de capa límite turbulenta ley 1/7"""
    return U_inf * (y/delta)**(1/7)

def log_law(y, u_tau, delta, nu=1.5e-5, kappa=0.41):
    """Perfil de capa límite turbulenta ley logarítmica"""
    mask = y < delta
    result = np.zeros_like(y)
    y_plus = y[mask] * u_tau / nu
    u_plus = (1/kappa) * np.log(y_plus) + 5.0
    result[mask] = u_tau * u_plus
    result[~mask] = u_tau * ((1/kappa) * np.log(delta * u_tau / nu) + 5.0)
    return result
mks = ['s','o','v']
plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)

plt.close('all')
rhoa = 1.2
rhoa_b = 1.0888  #densidad aire de bariloche
nu = 1.5e-5*rhoa_b / rhoa

plt.close('all')

fig,ax = plt.subplots(figsize=(8,8))
# fig2,ax2 = plt.subplots(figsize=(8,8))
A = pd.read_excel('Capa Límite Bandera .xlsx',decimal=',')
yi = A['y'].to_numpy().T
yi = yi.max()-yi

y_s = np.linspace(0,40,200)

V_i_s = A.iloc[:,1:6:2].to_numpy().T
delta_V_i_s = A.iloc[:,2:7:2].to_numpy().T

U_infs  =(5.88,15.7,28.4)
x_tunel = 0.5
for i,Vi in enumerate(V_i_s):
    deltai = delta_V_i_s[i]
    params, _ = curve_fit(turbulent_bl, yi, Vi,
                      p0=[np.max(Vi[-8:]), 12])  # estimación inicial
    if i==0:
        params, _ = curve_fit(blasius_profile, yi, Vi,
                      p0=[5, 9])
    if i==1:
        params, _ = curve_fit(blasius_profile, yi, Vi,
                      p0=[16, 13])
    if i==2:
        params, _ = curve_fit(turbulent_bl, yi, Vi,
                      p0=[27, 25])

    # params, _ = curve_fit(lambda y, u_tau, delta: log_law(y, u_tau, delta, nu=nu),
    #                   yi, Vi, p0=[0.5, 0.03])

    U_inf_fit, delta_fit = params
    lin = ax.errorbar(yi,Vi,deltai,marker=mks[i],linestyle='none',
                markersize=10,fillstyle='none',capsize=10,elinewidth=1)
    # if i ==0:
    #     ax.plot(y_s,blasius_profile(y_s, U_inf_fit, delta_fit,15e-6 ),color=lin[0].get_color())
    # elif i==1:
    #     ax.plot(y_s,turbulent_bl(y_s, 15.5, 19),color=lin[0].get_color())
    # elif i ==2:
    #     ax.plot(y_s,turbulent_bl(y_s, 28., 20),color=lin[0].get_color())

    # ax2.plot(yi,Vi/U_infs[i],marker=mks[i],linestyle='none',
                # markersize=10,fillstyle='none')
    print(U_inf_fit,delta_fit)
ax.grid()
ax.set_xlabel(r'$y$ [mm]')
ax.set_ylabel(r'$u$ [m/s]')

# Para cada perfil (10, 20, 30 Hz)
# Para cada perfil (10, 20, 30 Hz)
for i, Vi in enumerate(V_i_s):
    # Estimación de U_inf como promedio de últimos puntos
    U_inf = np.mean(Vi[-3:])


# ax2.grid()
tikz_save('/home/juan/Documents/Publicaciones/2025_euromech/flag_shear/article/tikzs/tunnel_BL.tikz')
