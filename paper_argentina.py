import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt

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

def sistema_completo(s, xi, omega, u0, rho):
    """
    Sistema completo: ecuación diferencial + condiciones de borde.
    """
    N = len(s)
    res = np.zeros(N, dtype=complex)

    # Ecuación diferencial en nodos interiores
    res[2:-2] = residual_xi(s[2:-2], xi[2:-2], omega, u0, rho)

    # Condiciones de borde en s=0
    res[0] = xi[0]  # xi(0) = 0
    res[1] = (xi[1] - xi[0]) / (s[1] - s[0])  # xi'(0) ≈ (xi[1]-xi[0])/ds = 0

    # Condiciones de borde en s=1
    res[-2] = (xi[-1] - 2*xi[-2] + xi[-3]) / ((s[1] - s[0])**2)  # xi''(1) ≈ 0
    res[-1] = (xi[-1] - 3*xi[-2] + 3*xi[-3] - xi[-4]) / ((s[1] - s[0])**3)  # xi'''(1) ≈ 0

    return res




from scipy.optimize import root

# Parámetros de la grilla
s = np.linspace(0, 1, 100)
N = len(s)

# Rango de parámetros
rho_vals = np.linspace(0, 10, 20)
u0_vals = np.linspace(0.1, 200, 200)

# Guardar resultados
u0_crit = []

for rho in rho_vals:
    found = False
    for u0 in u0_vals:
        # Supón que la inestabilidad ocurre para omega real (crecimiento marginal)
        omega_guess = 10.0  # Valor inicial, puedes ajustar
        xi_guess = np.sin(np.pi * s)  # Modo inicial aproximado

        def F(omega):
            xi = xi_guess
            res = sistema_completo(s, xi, omega, u0, rho)
            return np.linalg.norm(res)

        sol = root(F, omega_guess)
        omega_sol = sol.x[0]
        # Criterio: si el residuo es suficientemente pequeño, hay solución
        if sol.success and F(omega_sol) < 1e-3:
            u0_crit.append(u0)
            found = True
            break
    if not found:
        u0_crit.append(np.nan)

plt.plot(rho_vals, u0_crit, 'o-')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$u_{0,crit}$')
plt.title('Curva de inestabilidad')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

s = np.linspace(0, 1, 100)
N = len(s)
rho_vals = np.linspace(0, 10, 20)
u0_vals = np.linspace(0.1, 200, 200)
u0_crit = []

for rho in rho_vals:
    found = False
    for u0 in u0_vals:
        # Barre omega en un rango razonable
        omega_range = np.linspace(0.1, 30, 50)
        min_sigma = 1e6
        omega_crit = None
        for omega in omega_range:
            # Modo inicial arbitrario (no importa la escala)
            xi_guess = np.sin(np.pi * s)
            res = sistema_completo(s, xi_guess, omega, u0, rho)
            sigma = np.linalg.norm(res)
            if sigma < min_sigma:
                min_sigma = sigma
                omega_crit = omega
        # Si el mínimo residuo es suficientemente pequeño, hay solución
        if min_sigma < 1e-2:
            u0_crit.append(u0)
            found = True
            break
    if not found:
        u0_crit.append(np.nan)

plt.plot(rho_vals, u0_crit, 'o-')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$u_{0,crit}$')
plt.title('Curva de inestabilidad')
plt.show()



rho = 2
u0 = 30

res = sistema_completo(s, xi_guess, omega, u0, rho)
