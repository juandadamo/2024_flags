 
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, interact_manual,interactive,widgets,Layout
from IPython.display import Latex
import scipy as sc
import sympy as sp
import pandas as pd

     
#Determinacion de los modos elásticos
#en una viga empotrada

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
