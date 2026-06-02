import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
nfont = 20
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": nfont,
    "axes.titlesize": nfont - 2,
    "axes.labelsize": nfont,
    "xtick.labelsize": nfont - 2,
    "ytick.labelsize": nfont - 2,
    "legend.fontsize": nfont - 1,
})
dirdata= 'plot_stability/'
plt.close('all')
R1 = pd.read_csv(dirdata+'raynaud_black.csv',header=None).to_numpy().T
R2 = pd.read_csv(dirdata+'raynaud_orange.csv',header=None).to_numpy().T
R3 = pd.read_csv(dirdata+'raynaud_yellow.csv',header=None).to_numpy().T

A1 = pd.read_csv(dirdata+'argentina.csv',header=None).to_numpy().T
S1 = pd.read_csv(dirdata+'shelley.csv',header=None).to_numpy().T
E1 = pd.read_csv(dirdata+'eloy_black.csv',header=None).to_numpy().T
E2 = pd.read_csv(dirdata+'eloy_gray1.csv',header=None).to_numpy().T
E3 = pd.read_csv(dirdata+'eloy_gray2.csv',header=None).to_numpy().T

fig, ax = plt.subplots(figsize=(6.75,5.5))
ax.plot(R1[0],R1[1],marker='o',linestyle='none',markersize=10,fillstyle='none',label='0.5',markeredgewidth=2)
ax.plot(R2[0],R2[1],marker='s',linestyle='none',markersize=10,fillstyle='none',label='1.0',markeredgewidth=2)
ax.plot(R3[0],R3[1],marker='d',linestyle='none',markersize=10,fillstyle='none',label='1.5',markeredgewidth=2)

i1 = np.polyfit(A1[0],1/A1[1],2)
p1 = np.poly1d(i1)
i2 = np.polyfit(S1[0],1/S1[1],2)
p2 = np.poly1d(i2)
ms = np.linspace(1e-3,4,100)

ax.plot(ms,1/p1(ms),color='black',linestyle='dashed',linewidth=3)
ax.plot(ms,1/p2(ms),color='gray',linestyle='dotted',linewidth=3)



#ax.plot(E3[0],E3[1],color='gray',linestyle='none',marker='.')



ax.set_ylim([0,30])
ax.set_xlim([0,4])

dE1 = np.diff(E1[0])
i0 = np.nonzero(dE1<0)[0][0]+1
id0 = np.polyfit(E1[0][:i0],1/E1[1][:i0],2)
p0 = np.poly1d(id0)
i1 = np.nonzero(dE1[i0:]>0)[0][0]-1+i0
id1 = np.polyfit(E1[0][i0:i1+1],E1[1][i0:i1+1],2)
p1 = np.poly1d(id1)
id2 = np.polyfit(E1[0][i1:],1/E1[1][i1:],2)
p2 = np.poly1d(id2)



#ax.plot(E1[0][:i0],E1[1][:i0],color='gray',linestyle='-',marker='.')
#ax.plot(E1[0][i1:],E1[1][i1:],color='gray',linestyle='-',marker='.')
ms0 = ms[ms<3.28]
ms1 = ms[np.logical_and(ms<3.28,ms>2.673)]
ms2 = ms[ms>2.673]
ax.plot(ms0,1/p0(ms0),'k',linewidth=3)
ax.plot(ms1,p1(ms1),'k',linewidth=3)
ax.plot(ms2,1/p2(ms2),'k',linewidth=3)


ustar = np.linspace(1e-3,30,200)
dE2 = np.diff(E2[1])
dE3 = np.diff(E3[1])
ie2_0 = np.nonzero(dE2<0)[0][0]+1
ie2_d0 = np.polyfit(E2[1][:ie2_0],E2[0][:ie2_0],2)
ie2_d1 = np.polyfit(E2[0][ie2_0-1:],E2[1][ie2_0-1:],3)
pe2_0 = np.poly1d(ie2_d0)
pe2_1 = np.poly1d(ie2_d1)
ustar0 = ustar[ustar<9.5767]
ms3 = ms[ms>1.27845]
ax.plot(pe2_0(ustar0),ustar0,color='.6',linewidth=3)
ax.plot(ms3,pe2_1(ms3),color='.6',linewidth=3)


ie3_0 = np.argmax(E3[1])+1
ie3_d0 = np.polyfit(E3[1][:ie3_0],E3[0][:ie3_0],2)
ie3_d1 = np.polyfit(E3[0][ie3_0-1:],E3[1][ie3_0-1:],3)
pe3_0 = np.poly1d(ie3_d0)
pe3_1 = np.poly1d(ie3_d1)
ustar0 = ustar[ustar<11.768]
ms4 = ms[ms>1.3517]
ax.plot(pe3_0(ustar0),ustar0,color='0.3',linewidth=3)
ax.plot(ms4,pe3_1(ms4),color='0.3',linewidth=3)

ax.legend(title=r'$H/L$',fontsize=14)
pickle.dump(fig, open('fig_stability.pickle', 'wb'))
# ax.plot(E3[0],E3[1],'o')


# with open('fig_stability.pickle', 'rb') as f:
#     fig2 = pickle.load(f)
