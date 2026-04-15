import numpy as np
import matplotlib.pyplot as plt
from tikzplotlib import save as savetikz
plt.rc('font', family='serif', size=18)
plt.rc('text', usetex=True)
plt.close('all')
x = np.linspace(0, 1, 200)
scale = 1/2.54
fig,ax = plt.subplots(figsize=(10*scale,7*scale))


def modo2(x):
    return np.cosh(4.694*x) - np.cos(4.694*x) - 1.0185*(np.sinh(4.694*x) - np.sin(4.694*x))

# plt.plot(x, y1, 'b-', label='Modo 1')
y2 = modo2(x)
for ni,yi in enumerate(np.linspace(0,5,7)):
    ax.plot(x, -y2-yi,color='gray',linewidth=1,linestyle='dashed',dashes=[4,4,4])
    if ni ==3:
        ax.plot(x, -y2-yi,color='tab:blue',linewidth=3)


for ni,xi in enumerate(np.linspace(0,1,7)):
    yi = (np.cosh(4.694*xi) - np.cos(4.694*xi)) - 1.0185*(np.sinh(4.694*xi) - np.sin(4.694*xi))
    ax.plot([xi,xi],[-yi,-yi-5],color='gray',linewidth=1,linestyle='dashed',dashes=[4,4,4])

ax.plot(x, -y2,color='black',linewidth=3)
ax.plot(x, -y2-5,color='black',linewidth=3)
ax.plot(x, -y2-5.5,color='black',linewidth=1)
dy = (-y2[51] +y2[50])/(x[1])
ax.annotate("",xytext=(x[0]+.01,-y2[0]-5.5),xy=(-0.025+0.01,-y2[0]-5.5+.01), arrowprops=dict(arrowstyle="->"))
ax.annotate("",xytext=(x[-1],-y2[-1]-5.5),xy=(x[-1]+0.025,-y2[-1]-5.5+.18), arrowprops=dict(arrowstyle="->"))
ax.text(x[100],-y2[100]-5.5-.1,'$L$',bbox=dict(facecolor='white', alpha=1,edgecolor='none'))


ax.plot([x[0],x[0]],[-y2[0],-y2[0]-5],color='black',linewidth=3)
ax.plot([x[-1],x[-1]],[-y2[-1],-y2[-1]-5],color='black',linewidth=3)



ax.annotate("", xytext=(0, 1.5), xy=(0, -6.5),
            arrowprops=dict(arrowstyle="->"))
ax.annotate("", xytext=(0,y2[0]-2.5), xy=(.2, y2[0]-2.5),arrowprops=dict(arrowstyle="->"))

diry = -1.5/-.1
xi,yi = [0,y2[0]-2.5]
dx = -.1
ax.annotate("", xytext=(xi,yi), xy=(xi+dx, yi+dx*diry),arrowprops=dict(arrowstyle="->"))
ax.text(xi+dx*1.9,yi+dx*diry-.1,'$y$')
ax.text(.2, y2[0]-2.7,'$x$')
ax.text(0.05, -6.5,'$z$')

xi = .5
yi = -modo2(xi)
ax.annotate("",xytext=(xi,yi-2.5),xy=(xi+.1,yi+.15-2.45),arrowprops=dict(arrowstyle="->",color='tab:blue'))
ax.text(xi+.05,yi-2.1,'$s$',color='tab:blue')

dxh = 0.05
xh = x[-1]+dxh
yh = -y2[-1]+.3
ax.plot([xh,xh],[yh,yh-5],color='black',linewidth=1)
ax.annotate("",xytext=(xh,yh-.1),xy=(xh,yh), arrowprops=dict(arrowstyle="->"))
ax.annotate("",xytext=(xh,yh-5+.1),xy=(xh,yh-5), arrowprops=dict(arrowstyle="->"))

ax.text(xh-.02,yh-2.5,'$H$',bbox=dict(facecolor='white', alpha=1,edgecolor='none'))


# xr,yr = [x[-1],-y2[-1]-2.5]
# ax.plot(xr,yr,'ro')
#
# # ax.annotate("", xytext=(0,y2[0]-2.5), xy=(.2, y2[0]-2.5),arrowprops=dict(arrowstyle="->"))
#
#
# dx = -.2

#ax.annotate("", xytext=(xr,yr), xy=(xr+dx, yr+dx*diry),arrowprops=dict(arrowstyle="->",color='tab:red'))

# ax.annotate("",xytext=(0,0),xy=(5,0), arrowprops=dict(arrowstyle="->",color='tab:red'))
# ax.plot([0,2],[-2.5,-2.5],linestyle='dashed',color='tab:red')
# ax.plot([xr,xr+.18],[yr,yr],linestyle='dashed',color='tab:red')
# ax.plot([xr+.18,xr+dx],[yr,yr+dx*diry],linestyle='dashed',color='tab:red')
# plt.plot(x, y3, 'g-', label='Modo 3')
ax.grid(True)
ax.axis('off')
ax.set_xlim([-.7,1.2])

delta_u = .5-.22
x_u = -.5
cx = 0
cy = 0
for yi in np.linspace(-6.5,1.5,7):
    xi =x_u + yi*cx
    ax.plot([xi,xi+delta_u],[yi,yi-delta_u*cy],'k',linewidth=0.7)
    ax.annotate("",xytext=(xi+delta_u-.02,yi),xy=(xi+delta_u-.008,yi), arrowprops=dict(arrowstyle="->"))


ax.text(-.24,yi+.3,r'$u_\infty$',bbox=dict(facecolor='white', alpha=1,edgecolor='none'))
fig.tight_layout()
savetikz('/home/juan/Documents/Publicaciones/2025_euromech/flag_shear/article/tikzs/esquema.tikz')
