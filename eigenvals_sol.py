import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

def C_theodorsen(k):
    if k == 0:
        return 1.0
    else:
        H1 = scipy.special.hankel1(1, k)
        H0 = scipy.special.hankel1(0, k)
        return H1 / (H1 + 1j * H0)

def f(s):
    return 2 * np.sqrt((1 - s) / s)

def n(s):
    return 2 * np.sqrt(2 * (1 - s) * s)

def matriz_sistema(s, omega, u0, rho):
    N = len(s)
    ds = s[1] - s[0]
    A = np.zeros((N, N), dtype=complex)
    n_s = n(s)
    f_s = f(s)
    M_s = 1 + rho * n_s
    q = omega / (2 * u0)
    Cq = C_theodorsen(q)
    for i in range(2, N-2):
        A[i, i-2] = 1 / ds**4
        A[i, i-1] = -4 / ds**4
        A[i, i]   = 6 / ds**4 - omega**2 * M_s[i] - 1j * rho * u0 * Cq * f_s[i] * omega
        A[i, i+1] = -4 / ds**4 + rho * u0**2 * Cq * f_s[i] / (2*ds)
        A[i, i+2] = 1 / ds**4 - rho * u0**2 * Cq * f_s[i] / (2*ds)
    A[0, 0] = 1
    A[1, 0] = -1/ds
    A[1, 1] = 1/ds
    A[-2, -3] = 1 / ds**2
    A[-2, -2] = -2 / ds**2
    A[-2, -1] = 1 / ds**2
    A[-1, -4] = -1 / ds**3
    A[-1, -3] = 3 / ds**3
    A[-1, -2] = -3 / ds**3
    A[-1, -1] = 1 / ds**3
    return A

# --- Parámetros y valores manuales ---
s = np.linspace(0.01, 0.99, 100)
rho_vals = np.linspace(3.72666666, 5, 4)
u0_vals = np.linspace(1, 25, 20)
umbral = 15e-3

# Valores manuales para evitar cálculo total en algunos puntos
omega_lista1 = np.array([
    (8.76923076923077+0.05263157894736836j),
    (9.047008547008549-0.014035087719298317j),
    (8.547008547008549+0.1859649122807017j),
    (8.824786324786327+0.11929824561403501j),
    (9.102564102564106+0.14152046783625724j),
    (9.76923076923077+0.1578947368421053j)
])
rho_lista1 = np.array([1.18, 1.60444444, 2.02888889, 2.45333333, 2.87777778, 3.30222222])
u0_crit_lista1 = np.zeros_like(rho_lista1)

# --- Cálculo para valores manuales ---
for j, rho_j in enumerate(rho_lista1):
    min_singular = np.zeros_like(u0_vals)
    for i, u0_i in enumerate(u0_vals[::-1]):
        A = matriz_sistema(s, omega_lista1[j], u0_i, rho_j)
        U = np.linalg.svd(A, compute_uv=False)
        min_singular[i] = np.min(U)
    min_global = np.min(min_singular)
    idx_min = np.unravel_index(np.argmin(min_singular, axis=None), min_singular.shape)
    if min_global < umbral:
        u0_crit_lista1[j] = u0_vals[::-1][idx_min]

# --- Cálculo automático para el resto ---
u0_crit = []
omega_crit_anterior = None
omega_crit_list = []
xi_crit_list = []

for irho, rho_j in enumerate(rho_vals):
    encontrado = False
    for u0_i in u0_vals[::-1]:
        # Barrido local o global según corresponda
        if irho == 0 or omega_crit_anterior is None:
            omega_rs = np.linspace(8, 11, 50)
            omega_is = np.linspace(-1, 1, 20)
        else:
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
        if (min_global < umbral) and (np.abs(omega_crit.imag) < 0.3):
            u0_crit.append(u0_i)
            omega_crit_list.append(omega_crit)
            u, lam, vh = np.linalg.svd(matriz_sistema(s, omega_crit, u0_i, rho_j))
            xi_crit = vh[-1, :]
            xi_crit_list.append(xi_crit)
            omega_crit_anterior = omega_crit
            encontrado = True
            # Refinamiento automático si hay salto grande
            if len(u0_crit) > 1:
                salto = np.abs(u0_crit[-1] - u0_crit[-2])
                if salto > 10:
                    print(f"Salto detectado en rho={rho_j:.2f}: refinando...")
                    # Aquí puedes agregar un rebarrido más fino si lo deseas
            break
    if not encontrado:
        u0_crit.append(np.nan)
        omega_crit_anterior = None

# --- Unir resultados manuales y automáticos si lo deseas ---
# (opcional, según cómo quieras graficar)

# --- Graficar curva de inestabilidad ---
plt.figure()
plt.plot(rho_lista1, u0_crit_lista1, 'o', label='Manual')
plt.plot(rho_vals, u0_crit, 'o-', label='Automático')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$u_{0,crit}$')
plt.title('Curva de inestabilidad (Argentina & Mahadevan 2005)')
plt.grid()
plt.legend()
plt.show()