#tunel viento modificacion

# y = ax^2 + bx + c
import matplotlib.pyplot as plt
import numpy as np
Umax = 5.5
Lz,Ly = [465,470]

Lx = 350
Lreduccion = 100


A =np.array([[3*Lx**2, 2*Lx],[Lx**3,Lx**2]])
b = (0,Lreduccion)

(a0,a1) = np.dot(np.mat(A)**-1,np.array([0,Lreduccion]).T).T

fig,ax = plt.subplots()
x = np.linspace(0,Lx,200)
y = (a0*x**3+a1*x**2).T
y2 = 0
ax.plot(x,y)
ax.plot(x,y2*x)
ax.plot(np.ones_like(y),y)


S_actual  = Lz*Ly



Lz2,Ly2 = [Lz-Lreduccion,Ly-Lreduccion]

S_2 = Lz2*Ly2

U_2 = Umax * S_actual / S_2
