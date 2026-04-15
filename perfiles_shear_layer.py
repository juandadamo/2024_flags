#/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
U1 = 1

U2 = 0

deltaU = U1-U2
Um = (U1+U2)/2
theta = 0.1
R = deltaU/(2*Um)

y = np.linspace(-1,1,200)

Uinf = 1*y

Utanh = Um*(1+R*np.tanh(y/(2*theta)))
Ulin = Um * (1+R*(y/(2*theta)))
Ulin[y>2*theta] = U1
Ulin[y<-2*theta] = 0
fig,ax = plt.subplots()
ax.plot(Utanh,y)
ax.plot(Ulin,y)
ax.grid()
ax.set_ylabel('$y$')
ax.set_xlabel('$u(y)$')
