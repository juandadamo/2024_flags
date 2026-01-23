import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros
omega = 1.0  # Frecuencia angular fija

# Función para la dinámica radial (dr/dt)
def radial_dynamics(r, mu):
    return mu * r + r**3 - r**5

# Encontrar puntos fijos y ciclos límite (analíticamente)
def find_equilibria(mu):
    r_equilibria = []
    # Punto fijo trivial
    r_equilibria.append(0.0)
    # Soluciones no triviales (resolver mu + r^2 - r^4 = 0)
    if mu >= -0.25:
        r_squared = np.roots([-1, 1, 0, mu])
        for root in r_squared:
            if root > 0 and np.isreal(root):
                r_equilibria.append(np.sqrt(root))
    return np.sort(np.real(r_equilibria))

# Rango del parámetro mu
mu_values = np.linspace(-0.5, 0.5, 500)

# Calcular equilibrios para cada mu
stable = []
unstable = []
for mu in mu_values:
    r_eq = find_equilibria(mu)
    for r in r_eq:
        # Estabilidad: derivada de dr/dt respecto a r
        stability = mu + 3 * r**2 - 5 * r**4
        if r == 0:
            if stability < 0:
                stable.append((mu, r))
            else:
                unstable.append((mu, r))
        else:
            if stability < 0:
                stable.append((mu, r))
            else:
                unstable.append((mu, r))

# Convertir a arrays para graficar
stable = np.array(stable)
unstable = np.array(unstable)

# Graficar
# plt.figure(figsize=(10, 6))
# plt.plot(stable[:, 0], stable[:, 1], 'b-', linewidth=2, label='Estable')
# plt.plot(unstable[:, 0], unstable[:, 1], 'r--', linewidth=2, label='Inestable')
# plt.xlabel('Parámetro $\mu$', fontsize=14)
# plt.ylabel('Amplitud $r$', fontsize=14)
# plt.title('Bifurcación Andronov-Hopf Subcrítica', fontsize=16)
# plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=12)
# plt.show()

mu = np.linspace(-0.5, 0.5, 100) # Valor de mu para el ejemplo
rama_i = np.zeros_like(mu)
rama_s = np.zeros_like(mu)
for i,mui in enumerate(mu):
    i_stable = np.abs(stable[:, 0] -mui).argmin()
    i_unstable = np.abs(unstable[:, 0] -mui).argmin()
    val_stable = stable[i_stable, 1]
    val_unstable = unstable[i_unstable, 1]
    rama_s[i] = val_stable
    rama_i[i] = val_unstable

fig,ax = plt.subplots(figsize=(10, 6))
ax.plot(mu, rama_s, 'ko', linewidth=2, label='Rama de Amplitud')

