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

# Parámetros físicos (ajusta según el paper)
L = 1.0
B = 1.0
m = 1.0
U_star_range = np.linspace(0.1, 10, 100)

# Ecuación característica (simplificada, debes ajustar según el modelo)
def char_eq(omega, U_star):
    k = omega * L / U_star
    Ck = C_theodorsen(k)
    # Ejemplo: ecuación característica genérica
    return omega**2 + B * omega**4 - U_star * Ck * omega

# Búsqueda de raíces
omegas_real = []
omegas_imag = []
for U_star in U_star_range:
    sol = scipy.optimize.root(lambda w: np.real(char_eq(w + 0j, U_star)), 1.0)
    omegas_real.append(np.real(sol.x[0]))
    omegas_imag.append(np.imag(sol.x[0]))

plt.figure()
plt.plot(U_star_range, omegas_real, label='Re(omega)')
plt.plot(U_star_range, omegas_imag, label='Im(omega)')
plt.xlabel('Reduced velocity $U^*$')
plt.ylabel('Frequency / Growth rate')
plt.legend()
plt.title('Figura 3 - Argentina & Madhevan (2004)')
plt.show()