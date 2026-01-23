import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import pandas as pd

plt.close('all')
def C_theodorsen(k):
    """Theodorsen function approximation."""
    if k == 0:
        return 1.0
    else:
        H1 = scipy.special.hankel1(1, k)
        H0 = scipy.special.hankel1(0, k)
        return H1 / (H1 + 1j * H0)
    
def f(s):
    return 2*np.sqrt((1-s)/s) 
def n(s):
    return 2*np.sqrt(2*(1-s)*s)

s = np.linspace(0.01,0.99,100)
f_s = f(s)
n_s = n(s)  


rho_vals = np.linspace(0, 10, 20)
u0_vals = np.linspace(10, 50, 50)
# u_0 = U / U_c
#  reduced velocity for an aeroelastic problem
#  U_c = (B/(rho_a*L^3))^0.5
# rho = rho_a*L/(rho_s*h)
# rho represents the added mass effect of the fluid on the structure

# M = 1 + rho*n(s)

# M * d2 eta/dt^2 + d4 eta/dx^4 = rho*u_0*C(q)*f(s)*(d eta/dt + u_0*d eta/dx)
# q = omega/(2*u_0)

# eta(t,0) = 0
# d eta/dx(t,0) = 0
# d2 eta/dx^2(t,L) = 0
# d3 eta/dx^3(t,L) = 0
# at the onset of instability Re(sigma=0), sigma = i*omega
# eta (s,t) = xi*exp(i*omega*t)

# a nonlinear eigenvalue problem for omega and xi


def residual_xi(s, xi, omega, u0, rho):
    """
    Residuo de la ecuación de autovalores no lineal para xi(s) y omega.
    xi: vector de desplazamientos modales en s
    omega: frecuencia compleja
    u0: velocidad reducida
    rho: masa adimensional
    """
    n_s = n(s)
    f_s = f(s)
    M_s = 1 + rho * n_s
    q = omega / (2 * u0)
    Cq = C_theodorsen(q)
    dxi_ds = np.gradient(xi, s)
    d4xi_ds4 = np.gradient(np.gradient(np.gradient(np.gradient(xi, s), s), s), s)
    term1 = d4xi_ds4
    term2 = -omega**2 * M_s * xi
    term3 =  rho * u0 * Cq * f_s * (1j*omega * xi + u0 * dxi_ds)
    return term1 + term2 + term3




def matriz_sistema(s, omega, u0, rho):
    N = len(s)
    ds = s[1] - s[0]
    A = np.zeros((N, N), dtype=complex)

    n_s = n(s)
    f_s = f(s)
    M_s = 1 + rho * n_s
    q = omega / (2 * u0)
    Cq = C_theodorsen(q)

    # Nodos interiores: ecuación diferencial
    for i in range(2, N-2):
        # Derivada cuarta central
        A[i, i-2] = 1 / ds**4
        A[i, i-1] = -4 / ds**4
        A[i, i]   = 6 / ds**4 - omega**2 * M_s[i] - 1j * rho * u0 * Cq * f_s[i] * omega
        A[i, i+1] = -4 / ds**4 + rho * u0**2 * Cq * f_s[i] / (2*ds)
        A[i, i+2] = 1 / ds**4 - rho * u0**2 * Cq * f_s[i] / (2*ds)
        # El término de la derivada primera se aproxima por diferencias centradas
        # Aquí se incluye en los coeficientes de i+1 e i-1

    # Condiciones de borde en s=0
    A[0, 0] = 1  # xi(0) = 0
    A[1, 0] = -1/ds  # xi'(0) ≈ (xi[1] - xi[0])/ds = 0
    A[1, 1] = 1/ds

    # Condiciones de borde en s=1
    A[-2, -3] = 1 / ds**2  # xi''(N-1) ≈ (xi_{N-1} - 2xi_{N-2} + xi_{N-3})/ds^2 = 0
    A[-2, -2] = -2 / ds**2
    A[-2, -1] = 1 / ds**2

    A[-1, -4] = -1 / ds**3  # xi'''(N-1) ≈ (x        else:i_{N-1} - 3xi_{N-2} + 3xi_{N-3} - xi_{N-4})/ds^3 = 0
    A[-1, -3] = 3 / ds**3
    A[-1, -2] = -3 / ds**3
    A[-1, -1] = 1 / ds**3

    return A

 
 



import time


u0_crit = []
rho_vals = np.linspace(3.72666666, 5, 4)
omega_rs = np.linspace(0.001, 30, 50)
omega_is = np.linspace(-2, 2, 20)
umbral = 15e-3  # Umbral para considerar que hay autovalor

time_0 = time.time()


u0_crit = []
omega_crit_anterior = None  # Para la continuación
omega_crit_list = []
xi_crit_list = []


omega_lista1 = np.array([(8.76923076923077+0.05263157894736836j),
 (9.047008547008549-0.014035087719298317j),
 (8.547008547008549+0.1859649122807017j),
 (8.824786324786327+0.11929824561403501j),
 (9.102564102564106+0.14152046783625724j),(9.76923076923077+0.1578947368421053j)])
rho_lista1 = np.array([1.18      , 1.60444444, 2.02888889, 2.45333333, 2.87777778,3.30222222])       

u0_crit_lista1 = np.zeros_like(rho_lista1)

for j,rho_j in enumerate(rho_lista1):
    min_singular = np.zeros_like(u0_vals)
    for i,u0_i in enumerate(u0_vals[::-1]):
        A = matriz_sistema(s, omega_lista1[j], u0_i, rho_j)
        U = np.linalg.svd(A, compute_uv=False)
        min_singular[i] = np.min(U)
    # raise ValueError()
    min_global = np.min(min_singular)
    idx_min = np.unravel_index(np.argmin(min_singular, axis=None), min_singular.shape)
    if (min_global < umbral) :
            u0_crit.append( u0_vals[::-1][idx_min])



u0_vals = np.linspace(1,25,20)
for irho, rho_j in enumerate(rho_vals[::]):
    encontrado = False
    for u0_i in u0_vals[::-1]:
        time_1 = time.time()
        print(f"rho={rho_j:.2f}, u0={u0_i:.2f}, tiempo transcurrido: {time_1 - time_0:.2f} s")
        # Si es el primer rho, barrido amplio; luego, barrido local
        if irho == 0 or omega_crit_anterior is None:
            omega_rs = np.linspace(8, 11, 50)
            omega_is = np.linspace(-1, 1, 20)
        else:
            # Barrido local alrededor del último omega_crit
            delta_r = 1
            delta_i = 1
            omega_rs = np.linspace(omega_crit_anterior.real - delta_r, omega_crit_anterior.real + delta_r, 50)
            omega_is = np.linspace(omega_crit_anterior.imag - delta_i, omega_crit_anterior.imag + delta_i, 50)
        min_singular = np.zeros((len(omega_rs), len(omega_is)))
        for kr, omega_rs_i in enumerate(omega_rs):
            for ki, omega_is_i in enumerate(omega_is):
                omega_k = omega_rs_i + omega_is_i*1j
                A = matriz_sistema(s, omega_k, u0_i, rho_j)
                U = np.linalg.svd(A, compute_uv=False)
                min_singular[kr, ki] = np.min(U)
        min_global = np.min(min_singular)
        idx_min = np.unravel_index(np.argmin(min_singular, axis=None), min_singular.shape)
        omega_crit = omega_rs[idx_min[0]] + 1j*omega_is[idx_min[1]]
        # Criterio: valor singular mínimo suficientemente pequeño y parte imaginaria de omega cerca de cero
        print(f'Singularidad: {min_global:.05f}, omega:{omega_crit:.03g}')
        # print(omega_crit)
        if (min_global < umbral) and (np.abs(omega_crit.imag) < 0.4):
            u0_crit.append(u0_i)
            omega_crit_list.append(omega_crit)
            u, lam, vh = np.linalg.svd(matriz_sistema(s, omega_crit, u0_i, rho_j))
            xi_crit = vh[-1, :]
            xi_crit_list.append(xi_crit)
            omega_crit_anterior = omega_crit  # Continuar desde aquí
            encontrado = True
            print(f"  --> Encontrado u0_crit={u0_i:.2f} para rho={rho_j:.2f} con omega={omega_crit:.3f} y valor singular mínimo={min_global:.2e}")
            if len(u0_crit) > 1:
                salto = np.abs(u0_crit[-1] - u0_crit[-2])
                print(salto)
                if np.logical_or(salto > 10,salto==np.nan):  # Ajusta este umbral según tu curva esperada
                    print(f"Salto detectado en rho={rho_j:.2f}: refinando...")
            
            
            break
        # else:
        #     #refine

        #     raise ValueError()  # Probar siguiente u0_i


    if not encontrado:
        u0_crit.append(np.nan)
        omega_crit_anterior = None  # Reinicia si no encontró



# # Después de agregar u0_crit.append(u0_i)
# if len(u0_crit) > 1:
#     salto = np.abs(u0_crit[-1] - u0_crit[-2])
#     if salto > 2:  # Ajusta este umbral según tu curva esperada
#         print(f"Salto detectado en rho={rho_j:.2f}: refinando...")
#         # Barrido más amplio y fino
#         delta_r_fino = 3
#         delta_i_fino = 1
#         omega_rs_fino = np.linspace(omega_crit_anterior.real - delta_r_fino, omega_crit_anterior.real + delta_r_fino, 30)
#         omega_is_fino = np.linspace(omega_crit_anterior.imag - delta_i_fino, omega_crit_anterior.imag + delta_i_fino, 30)
#         min_singular_fino = np.zeros((len(omega_rs_fino), len(omega_is_fino)))
#         for kr, omega_rs_i in enumerate(omega_rs_fino):
#             for ki, omega_is_i in enumerate(omega_is_fino):
#                 omega_k = omega_rs_fino[kr] + omega_is_fino[ki]*1j
#                 A = matriz_sistema(s, omega_k, u0_i, rho_j)
#                 U = np.linalg.svd(A, compute_uv=False)
#                 min_singular_fino[kr, ki] = np.min(U)
#         min_global_fino = np.min(min_singular_fino)
#         idx_min_fino = np.unravel_index(np.argmin(min_singular_fino, axis=None), min_singular_fino.shape)
#         omega_crit_fino = omega_rs_fino[idx_min_fino[0]] + 1j*omega_is_fino[idx_min_fino[1]]
#         if (min_global_fino < umbral) and (np.abs(omega_crit_fino.imag) < 0.2):
#             # Reemplaza el valor anterior por el refinado
#             u0_crit[-1] = u0_i
#             omega_crit_anterior = omega_crit_fino
#             print(f"  --> Refinado: u0_crit={u0_i:.2f} para rho={rho_j:.2f} con omega={omega_crit_fino:.3f} y valor singular mínimo={min_global_fino:.2e}")



rho_total = np.hstack((rho_lista1,rho_vals))
omega_total = np.hstack((omega_lista1, omega_crit_list))
fig,ax = plt.subplots()

ax.plot(rho_total, u0_crit, 'o',linestyle='',fillstyle='none')
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$u_{0,crit}$')
# ax.title('Curva de inestabilidad (Argentina & Mahadevan 2005)')
ax.grid()
 

omega_total = np.hstack((omega_lista1, np.asarray(omega_crit_list)))

data_out = pd.DataFrame({
    'rho': rho_total,
    'u0_crit': u0_crit,
    'omega_crit': omega_total})
#data_out.to_csv('instability_values_flag.csv', index=False)