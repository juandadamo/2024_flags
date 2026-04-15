import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# DATOS
# ==================================================
L = 350.0          # mm
delta_L = 100.0 # mm (porque 100 mm de diámetro -> 50 mm de radio)



def y_cubico(x):
    x_norm = x / L
    return -2*delta_L * x_norm**3 + 3*delta_L * x_norm**2



# ==================================================
# CALCULAR PUNTOS
# ==================================================
x = np.linspace(0, L, 200)

y_cub = y_cubico(x)






# ==================================================
# GRÁFICOS
# ==================================================
fig, ax1 = plt.subplots( figsize=(10, 8))


ax1.plot(x, y_cub, 'r-', label='Cúbico', linewidth=2)

ax1.set_xlabel('x (mm)')
ax1.set_ylabel('Desplazamiento de la pared (mm)')
ax1.set_title('Perfiles de contracción (y(0)=0, reducción final = -50 mm)')
ax1.legend()
ax1.grid(True)


# Gráfico 2: Ángulos

