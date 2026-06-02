import numpy as np
import matplotlib.pyplot as plt

from skimage import exposure
import glob
from matplotlib import rcParams
from funciones_flag import *
import tifffile as tif
from skimage import filters, measure


plt.close('all')

data = np.load('/home/juan/Documents/script_python/2026_fluttering_energy/sandbox/V3/full_V_45.5_SINDy_ready.npz')



A = np.load('full_uniform/velocidad_41.npz')
YT = A['YT']
# YT[np.isnan(YT)] = 0
# fig,ax = plt.subplots()
ymin = 600
ymax = 600

nxorigin, nyorigin = [1151, 559]
nxorigin, nyorigin = [0,0]


YT[np.isnan(YT)] = 0
YT = YT  - nyorigin
x_s = np.arange(len(YT[0]))
x_s = -(x_s - nxorigin)
for yi in YT[:,243:1151]:
#     #ax.plot(yi,linestyle='none',marker='.',color='tab:blue')
     yi_0 = yi[yi!=0]
     ymin = np.min((ymin,yi_0.min()))
     ymax = np.max((ymax,yi_0.max()))
# for yi in YT[:]:
#      ax.plot(yi,linestyle='none',marker='.',color='tab:blue')

YT1 = YT[:,243:1151]
[U,s,Vh] = np.linalg.svd(YT1-YT1.mean(0),full_matrices=False)
# ax.plot([0,1000],[ymin,ymin],'r')
# ax.plot([0,1000],[ymax,ymax],'r')
ymean = (ymin+ymax)/2
ymean = 559
# ax.plot([0,1300],[ymean,ymean],'r')
# ax.plot((0,1300),[559,559],'tab:orange')
YT = YT-ymean
fig,ax = plt.subplots()
for yi in YT1:
     ax.plot(yi,'o')




