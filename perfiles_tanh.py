import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
# $U_m\left[1+R(\tanh (2 y/\delta_\omega+\eta)\right]$
U1 = 1.0
U2 = 0.0
Um = ( U1 + U2) / 2
deltaU = U1 - U2
R = deltaU / (2*Um)
delta_omega = 0.5
eta = 0.1
y = Um * (1 + R * np.tanh(2 * y / delta_omega + eta))
# y = np.tanh(x)

fig,ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label='Tanh Function', color='blue')
ax.axhline(0, color='black', lw=0.5, ls='--')