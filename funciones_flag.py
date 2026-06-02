 
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interact_manual,interactive,widgets,Layout
from IPython.display import Latex
import scipy as sc
import sympy as sp
import pandas as pd
import skimage as sk
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.morphology import skeletonize, thin, remove_small_objects,closing, square, disk, medial_axis,binary_opening,binary_closing
from skimage.measure import label, regionprops
from scipy.signal import find_peaks
from skimage import feature, morphology
from scipy.interpolate import UnivariateSpline
#Determinacion de los modos elásticos
#en una viga empotrada
from scipy.signal import savgol_filter
x, y, z = sp.symbols('x y z')
eq1 = sp.diff(sp.cos(x)*sp.cosh(x)+1, x)
def fun_modn (x):
    return np.cosh(x)*np.cos(x)+1
    
callable_fct = sp.lambdify(x, eq1)

x_s = np.linspace(1,15,200)
y_s = np.zeros_like(x_s)
for i,xi in enumerate(x_s):
    y_s[i] = fun_modn(xi)
signo_s = np.sign(y_s)
diff_signo_s = np.diff(signo_s)
x0s = x_s[np.nonzero(diff_signo_s)]

BnL = np.zeros_like(x0s)
for i,x0i in enumerate(x0s):
    BnL[i] = sc.optimize.fsolve(fun_modn,x0i,fprime=callable_fct)[0]
#display(Latex(f'Las raíces son $\\beta_1 L={BnL[0]:.3f}$, $\\beta_2 L={BnL[1]:.3f},\\ldots$'  )    )

#deformacion elastica viga empotrada
def w_n (Bn,x,A1=1,L=1):
    wn = A1*((np.cosh(Bn*x)-np.cos(Bn*x))+(np.cos(Bn*L)+np.cosh(Bn*L))/(np.sin(Bn*L)+np.sinh(Bn*L))*(np.sin(Bn*x)-np.sinh(Bn*x)))
    return wn



def w_n_phase(Bn, x, phase=0, A1=1.0, L=1.0):
    spatial_part = (np.cosh(Bn*x) - np.cos(Bn*x)) + \
                  (np.cos(Bn*L) + np.cosh(Bn*L)) / \
                  (np.sin(Bn*L) + np.sinh(Bn*L)) * \
                  (np.sin(Bn*x) - np.sinh(Bn*x))
    return A1 * spatial_part * np.cos(phase)  # phase en radianes

# Define un objeto lamina flexible, con sus propiedades mecánicas
class material:
    def __init__(self, name,th,rho,lstuart):
        g = 9.8
        self.name = name
        self.thickness = th*1e-6
        self.rho = rho
        self.lstuart = lstuart*1e-3
        self.B =  ((lstuart*1e-3*1.103)**3)*rho*th*1e-6*g # rigidez a la flexion por unidad de longitud
        self.I = (th*1e-6)**3/12  #momento de inercia por unidad de longitud
        self.E = self.B / self.I
        self.L = 1

        self.mu = self.rho  * self.thickness
        #calculo modo empotrado

        x_s = np.linspace(1,15,200)
        y_s = np.zeros_like(x_s)
        for i,xi in enumerate(x_s):
            y_s[i] = fun_modn(xi)
        signo_s = np.sign(y_s)
        diff_signo_s = np.diff(signo_s)
        x0s = x_s[np.nonzero(diff_signo_s)]

        BnL = np.zeros_like(x0s)
        for i,x0i in enumerate(x0s):
            BnL[i] = sc.optimize.fsolve(fun_modn,x0i,fprime=callable_fct)[0]
        self.BnL = BnL
        self.BetaL = BnL[0]
    def freq_nat(self):
        self.beta,self.fn = np.tile(np.zeros_like(self.BnL),[2,1])
        for i,BnLi in enumerate(self.BnL):
            self.beta[i] = BnLi/self.L
            Bni = BnLi/self.L
            self.fn[i] = Bni**2*(self.B/self.mu)**0.5/2/np.pi
        
        
    def update_modo(self,nselect):
        
        self.BetaL = BnL[nselect-1]
        print(nselect)   
        
        
def delta_turb(U):
    nu = 15e-6
    L_tunel = 0.663
    Re = U*L_tunel/nu
    delta_x = 0.379*Re**(-1/5)
    return delta_x*L_tunel
def frec_kh(Um,theta):
    return 0.032*Um/theta


def veloc_tunel_ib(frec):
    rhoa = 1.2
    frecs = np.array([10,12,14,16,18,20,22,24])
    presiones = np.array([17.2,26.5, 39.5 , 54.5 , 73 , 93,115,141])
    rhoa = 1.2
    pbariloche = 91.92e3
    patm = 101.3e3

    pratio = pbariloche/patm

    rhoa_b = rhoa*pratio
    Veloc = np.sqrt(2*presiones/rhoa_b)
    p1 = np.polyfit(frecs,Veloc,1)
    f_vel = np.poly1d(p1)
    return f_vel(frec)
        
   
def delta_turb(x, U_inf, nu):
    Re_x = (U_inf * x) / nu
    delta = 0.37 * x * (Re_x)**(-0.2)
    return delta
import numpy as np

def longitud_equivalente_capa_limite_turbulenta(delta, Uinf, nu):
    """
    Calcula la longitud x necesaria para alcanzar un espesor de capa límite turbulenta delta.

    Parámetros:
        delta (float): Espesor de capa límite turbulenta [m].
        Uinf (float): Velocidad del flujo libre [m/s].
        nu (float): Viscosidad cinemática del fluido [m²/s].

    Retorna:
        x (float): Longitud característica de la placa [m].
    """
    x = (delta / 0.37)**(5/4) * (Uinf / nu)**(1/4)
    return x


Ym = 50
t = 75
dens_sup = 80
rho_papel = dens_sup/(t*1e-6)*1e-3
Papel_80 = material('Papel 80gr/m2',t,rho_papel,Ym)
B = Papel_80.B
rho = rho_papel
k1L = BnL[0]
k2L = BnL[1]
L = 130e-3
Papel_80.L = L
Papel_80.freq_nat()


def lee_datos_foil(lista_caso_2d,Velocidad,Amplitud,Frecuencia):
    fsampling = 1000 # Hz
    escalax = 1/0.138 # px/mm
    Lbandera = 138.5 # mm
    dx = 1/escalax
    caso = lista_caso_2d[0].split('/')[1].split('_')[0]
    print(caso)
    for j, filej in enumerate(lista_caso_2d[:]):
        A1 = np.load(filej)
        Asum = A1['Imagen_sum']

        YT = A1['A_curva_i']
        frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
        Velocidad[j] =  veloc_tunel_ib(frec_j)
        if caso == 'triang':
            factor_thresh = 1
            image = Asum**(.12)
            edges = feature.canny(image, sigma=4)
            closed_edges = morphology.closing(edges, morphology.disk(radius=5))
            image[closed_edges] = 1
            image[np.logical_not(closed_edges)] = 0
            lim_superior =np.nonzero(image==1)[0].max()
            lim_inferior = np.nonzero(image==1)[0].min()
            delta_coord = lim_superior - lim_inferior
            image = Asum**.15
            thresh = np.percentile(image, 65)  # o un valor fijo ej. 0.3
            binary = image > thresh


    # 2. Proyección vertical: suma de píxeles por fila
            vertical_projection = binary.sum(axis=1)
            n_x = len(vertical_projection)
            YT = A1['A_curva_i']
            vertical_projection_s = savgol_filter(vertical_projection, 101, 5)
            lim_superior = vertical_projection_s[:100].mean()
            # print(lim_superior)
            lim_inferior = vertical_projection_s[-100:].mean()
            coord_0 = np.nonzero(vertical_projection_s[:int(n_x/2)][::-1]<lim_superior)[0][0]*-1+int(n_x/2)
            coord_1 = np.nonzero(vertical_projection_s[int(n_x/2):-10][:]<lim_inferior)[0][0]+int(n_x/2)
            delta_coord = coord_1-coord_0

            x_n = np.arange(0,len(vertical_projection))

            # vertical_projection_s = savgol_filter(vertical_projection, 101, 5)
            # print(frec_j)
            # ax.plot(np.abs(np.diff(vertical_projection_s)*100),x_n[:-1])
            if frec_j == 11.5:
                delta_coord = 638-233
            elif frec_j == 12:
                delta_coord = 650-197
            elif frec_j == 13:
                delta_coord = 707-139
            elif frec_j == 14:
                delta_coord = 721-87
            elif frec_j == 15:
                delta_coord = 682-81
            elif frec_j == 16:
                delta_coord = 704-39
            elif frec_j == 17:
                delta_coord = 726 - 34
            elif frec_j == 18:
                delta_coord = 726 - 37
            elif frec_j == 19:
                delta_coord = 735  - 35
            elif frec_j == 20:
                delta_coord = 741 - 46



            Amplitud[j]  = delta_coord*1.0/ escalax  # mm



        elif caso == 'rect':
            frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
            Velocidad[j] =  veloc_tunel_ib(frec_j)
            image = Asum**.1
            thresh = np.percentile(image, 35)  # o un valor fijo ej. 0.3
            min_image = image.min()
            max_image = image.max()
            mean_image = image.mean()
            std_image = image.std()
            binary = image > thresh
            # print(min_image,mean_image,std_image,max_image)


    # 2. Proyección vertical: suma de píxeles por fila
            vertical_projection = binary.sum(axis=1)
            n_x = len(vertical_projection)
            x_n = np.arange(0,len(vertical_projection))
            p1 = np.polyfit(x_n,vertical_projection,9)
            fp1 = np.poly1d(p1)
            spline_v = UnivariateSpline(x_n, vertical_projection, s=50,k=2)
            senal_sg = savgol_filter(vertical_projection, 101, 5)


            vertical_projection_s = spline_v(x_n)

            YT = A1['A_curva_i']
            # raise ValueError()
            lim_superior = vertical_projection[10:100].mean()
            lim_inferior = vertical_projection[-100:].mean()+vertical_projection[-100:].std()*4
            coord_0 = np.nonzero(vertical_projection[:int(n_x/2)][::-1]<lim_superior)[0][0]*-1+int(n_x/2)
            coord_1 = np.nonzero(vertical_projection[int(n_x/2):-10][:]<lim_inferior)[0][0]+int(n_x/2)
            delta_coord = coord_1-coord_0

            if frec_j ==18:
                delta_coord = 866-246
            elif frec_j == 19:
                delta_coord = 854 - 246
            elif frec_j == 20:
                delta_coord = 876-250



            Amplitud[j]  = delta_coord*1.0/ escalax  # mm
            # fig,ax = plt.subplots()
            # ax.imshow(image)
            # ax.plot(vertical_projection,x_n,'wo')
            # ax.plot(vertical_projection_s,x_n,color='tab:orange',linewidth=3)
            # ax.plot(senal_sg,x_n,color='tab:green',linewidth=3)
            # ax.plot([0,len(Asum.T)],[coord_0,coord_0],'r')
            # ax.plot([0,len(Asum.T)],[coord_1,coord_1],'y')
            # ax.plot(np.abs(np.diff(senal_sg)*100),x_n[:-1])
            # raise ValueError()





        else:
            factor_thresh = 1.5


            umbral_intensidad = sk.filters.threshold_otsu(Asum)/factor_thresh
            A2 = Asum>= umbral_intensidad
            A_clean = binary_closing(A2, square(3))  # Elimina píxeles aislados
            A_clean = binary_opening(A_clean, square(3))  # Suaviza bordes
            label_image = label(A_clean)
            image = Asum.copy()

            # raise ValueError()
            aux = []
            for region in regionprops(label_image):
                if region.area>1020:
                    coord_amp = region.coords
                    aux.append((coord_amp[:,0].max(),coord_amp[:,0].min()))

            aux = np.asarray(aux)

            delta_coord = np.abs(coord_amp[:,0].max()-coord_amp[:,0].min())
            delta_coord = np.abs(aux[:,0].max()-aux[:,1].min())

            image = Asum*.25
            thresh = np.percentile(image, 65)  # o un valor fijo ej. 0.3
            binary = image > thresh

    # 2. Proyección vertical: suma de píxeles por fila
            vertical_projection = binary.sum(axis=1)
            vertical_projection_s = savgol_filter(vertical_projection, 101, 5)
            n_x = len(vertical_projection)
            YT = A1['A_curva_i']

            lim_superior = vertical_projection_s[:100].mean()+ vertical_projection[:100].std()
            # print(lim_superior)
            lim_inferior = vertical_projection_s[-100:].mean()+ vertical_projection[-100:].std()
            coord_0 = np.nonzero(vertical_projection_s[:int(n_x/2)][::-1]<lim_superior)[0][0]*-1+int(n_x/2)
            coord_1 = np.nonzero(vertical_projection_s[int(n_x/2):-10][:]<lim_inferior)[0][0]+int(n_x/2)
            delta_coord = coord_1-coord_0


            x_n = np.arange(0,len(vertical_projection))
            # fig,ax = plt.subplots()
            # ax.imshow(image)
            #
            # ax.plot(vertical_projection,x_n,'wo')
            # # raise ValueError()
            # ax.plot(vertical_projection_s,x_n,color='tab:orange',linewidth=3)
            # # ax.plot(senal_sg,x_n,color='tab:green',linewidth=3)
            # ax.plot([0,len(Asum.T)],[coord_0,coord_0],'r')
            # ax.plot([0,len(Asum.T)],[coord_1,coord_1],'y')



            # raise ValueError()
            Amplitud[j]  = delta_coord*1.0/ escalax  # mm




            # raise ValueError()
        print(f"Velocidad del túnel de viento: {Velocidad[j]:.2f} m/s")
        Fourier_YT = np.fft.fft(YT.T,axis=1)
        Fourier_YT_x= np.fft.fft(YT.T,axis=0)
        FYT = np.abs(Fourier_YT).sum(axis=0)
        FYX = np.abs(Fourier_YT_x).sum(axis=1)
        freq_YT = np.fft.fftfreq(len(YT), d=1/fsampling)
        k_YT = np.fft.fftfreq(len(YT.T), d=dx)
        peak_freqs, _ = find_peaks(FYT, height=0.1*np.max(FYT))
        Frecuencia[j] = np.abs(freq_YT[peak_freqs][FYT[peak_freqs].argmax()])
        print(f"Frecuencia de la señal: {Frecuencia[j]:.2f} Hz")

    return Velocidad, Amplitud, Frecuencia
        # plt.subplots()
        #
        # plt.semilogy(freq_YT[freq_YT>0], FYT[freq_YT>0], 'k-')
        # plt.grid()
        # plt.xlabel('Frecuency (Hz)')
        # plt.ylabel('PSD')
        # plt.plot(freq_YT[peak_freqs], FYT[peak_freqs], 'ro')
        # plt.xlim([0, 100])

